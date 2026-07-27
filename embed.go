// Embedding-based semantic search backed by a local Ollama server (free, no
// API key). We embed every code chunk once, cache the vectors on disk, and
// score queries with cosine similarity. Subsequent runs re-use the cache and
// only re-embed files whose (size, mtime) changed.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	cacheVersion = 1
	// Chunk config: split each file into ~40-line windows with a 5-line
	// overlap so a match near a chunk boundary still surfaces. Numbers are
	// deliberately conservative — small enough for a 512-dim embedding to
	// carry meaning, big enough to hold a full function.
	chunkLines   = 40
	chunkOverlap = 5
	// Ollama's /api/embed batches embeddings in one request. Keep the
	// batches small enough to stay under any default request-size limits.
	embedBatchSize = 32
)

// embeddedChunk is one entry in the on-disk cache.
type embeddedChunk struct {
	Path      string    `json:"path"`
	StartLine int       `json:"start"` // 1-indexed
	EndLine   int       `json:"end"`
	Content   string    `json:"body"`  // trimmed for display
	Embedding []float32 `json:"vec"`
	Norm      float32   `json:"norm"`
}

type cacheEntry struct {
	Size    int64           `json:"size"`
	ModTime int64           `json:"mtime"`
	Chunks  []embeddedChunk `json:"chunks"`
}

type embedCache struct {
	Version int                   `json:"version"`
	Model   string                `json:"model"`
	Files   map[string]cacheEntry `json:"files"`
}

// embedSearch is the entry point for -mode embed. It walks `root`, embeds
// every chunk not already cached with matching (size, mtime), embeds the
// query, and returns the top-K chunks by cosine similarity.
func embedSearch(
	ctx context.Context, root, cachePath, embedModel, query string,
	exts, skipDirs map[string]bool, top int, verbose bool,
) ([]result, error) {
	if cachePath == "" {
		cachePath = filepath.Join(root, ".semsearch-cache.json")
	}
	cache, err := loadCache(cachePath, embedModel)
	if err != nil {
		return nil, err
	}

	// First pass: figure out which files need (re)embedding.
	var toEmbed []indexedFile
	var kept int
	err = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if !exts[strings.ToLower(filepath.Ext(p))] {
			return nil
		}
		info, err := d.Info()
		if err != nil || info.Size() > 512*1024 {
			return nil
		}
		rel, e := filepath.Rel(root, p)
		if e != nil {
			rel = p
		}
		rel = filepath.ToSlash(rel)

		if e, ok := cache.Files[rel]; ok && e.Size == info.Size() && e.ModTime == info.ModTime().Unix() {
			kept++
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil || isBinary(data) {
			return nil
		}
		toEmbed = append(toEmbed, indexedFile{path: rel, snippet: string(data)})
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Drop cache entries for files that have vanished.
	current := map[string]bool{}
	for _, f := range toEmbed {
		current[f.path] = true
	}
	for path := range cache.Files {
		if !current[path] {
			// Only drop if the file is really gone (not just filtered out
			// this run by a different -ext / -no-gitignore setting).
			if _, err := os.Stat(filepath.Join(root, path)); os.IsNotExist(err) {
				delete(cache.Files, path)
			}
		}
	}

	// Second pass: chunk + embed the new/changed files.
	if verbose {
		fmt.Fprintf(os.Stderr, "index: %d cached, %d to embed\n", kept, len(toEmbed))
	}
	for _, f := range toEmbed {
		chunks := chunkFile(f.snippet)
		if len(chunks) == 0 {
			continue
		}
		texts := make([]string, len(chunks))
		for i, c := range chunks {
			texts[i] = c.body
		}
		vecs, err := embedBatched(ctx, embedModel, texts)
		if err != nil {
			return nil, fmt.Errorf("embed %s: %w", f.path, err)
		}
		out := make([]embeddedChunk, len(chunks))
		for i, c := range chunks {
			v := vecs[i]
			out[i] = embeddedChunk{
				Path:      f.path,
				StartLine: c.start,
				EndLine:   c.end,
				Content:   truncate(c.body, 240),
				Embedding: v,
				Norm:      vecNorm(v),
			}
		}
		info, _ := os.Stat(filepath.Join(root, f.path))
		cache.Files[f.path] = cacheEntry{
			Size:    info.Size(),
			ModTime: info.ModTime().Unix(),
			Chunks:  out,
		}
		if verbose {
			fmt.Fprintf(os.Stderr, "  + %s (%d chunks)\n", f.path, len(out))
		}
	}
	if err := saveCache(cachePath, cache); err != nil {
		return nil, err
	}

	// Third pass: embed the query and rank all cached chunks.
	qVecs, err := embedBatched(ctx, embedModel, []string{query})
	if err != nil {
		return nil, err
	}
	qVec, qNorm := qVecs[0], vecNorm(qVecs[0])

	type scored struct {
		chunk embeddedChunk
		score float32
	}
	var all []scored
	for _, entry := range cache.Files {
		for _, c := range entry.Chunks {
			if len(c.Embedding) != len(qVec) {
				continue // model changed under us; the entry will get rebuilt next run
			}
			all = append(all, scored{c, cosineSim(c.Embedding, qVec, c.Norm, qNorm)})
		}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].score > all[j].score })
	if top < 1 {
		top = 1
	}
	if top > len(all) {
		top = len(all)
	}

	results := make([]result, top)
	for i := 0; i < top; i++ {
		c := all[i].chunk
		results[i] = result{
			Path:   fmt.Sprintf("%s:%d-%d", c.Path, c.StartLine, c.EndLine),
			Score:  math.Round(float64(all[i].score)*100) / 1, // scale to 0..100
			Reason: c.Content,
		}
	}
	return results, nil
}

// ── on-disk cache ───────────────────────────────────────────────────────────

func loadCache(path, model string) (*embedCache, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return &embedCache{Version: cacheVersion, Model: model, Files: map[string]cacheEntry{}}, nil
	}
	var c embedCache
	if err := json.Unmarshal(data, &c); err != nil {
		// Malformed cache → start fresh rather than fail.
		return &embedCache{Version: cacheVersion, Model: model, Files: map[string]cacheEntry{}}, nil
	}
	// If the model changed, discard the vectors (dimensions likely differ).
	if c.Model != model || c.Version != cacheVersion {
		return &embedCache{Version: cacheVersion, Model: model, Files: map[string]cacheEntry{}}, nil
	}
	if c.Files == nil {
		c.Files = map[string]cacheEntry{}
	}
	return &c, nil
}

func saveCache(path string, c *embedCache) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	if err := enc.Encode(c); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	f.Close()
	return os.Rename(tmp, path)
}

// ── chunking & similarity ───────────────────────────────────────────────────

type textChunk struct {
	start, end int
	body       string
}

func chunkFile(text string) []textChunk {
	lines := strings.Split(text, "\n")
	var out []textChunk
	step := chunkLines - chunkOverlap
	if step < 1 {
		step = 1
	}
	for i := 0; i < len(lines); i += step {
		end := i + chunkLines
		if end > len(lines) {
			end = len(lines)
		}
		body := strings.TrimSpace(strings.Join(lines[i:end], "\n"))
		if body != "" {
			out = append(out, textChunk{start: i + 1, end: end, body: body})
		}
		if end == len(lines) {
			break
		}
	}
	return out
}

func vecNorm(v []float32) float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return float32(math.Sqrt(s))
}

func cosineSim(a, b []float32, aNorm, bNorm float32) float32 {
	if aNorm == 0 || bNorm == 0 {
		return 0
	}
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return float32(dot / (float64(aNorm) * float64(bNorm)))
}

func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	// Only the first line of the chunk — enough to eyeball what matched.
	if nl := strings.IndexByte(s, '\n'); nl >= 0 {
		s = s[:nl]
	}
	if len(s) > n {
		s = s[:n] + "…"
	}
	return s
}

// ── Ollama /api/embed client ────────────────────────────────────────────────

func embedBatched(ctx context.Context, model string, texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += embedBatchSize {
		end := i + embedBatchSize
		if end > len(texts) {
			end = len(texts)
		}
		vecs, err := embedOllama(ctx, model, texts[i:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func embedOllama(ctx context.Context, model string, inputs []string) ([][]float32, error) {
	host := envOr("OLLAMA_HOST", "http://localhost:11434")
	body, _ := json.Marshal(map[string]any{"model": model, "input": inputs})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(host, "/")+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: 5 * time.Minute}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama /api/embed returned %s", resp.Status)
	}
	var out struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Embeddings) != len(inputs) {
		return nil, fmt.Errorf("ollama returned %d embeddings, want %d",
			len(out.Embeddings), len(inputs))
	}
	return out.Embeddings, nil
}

// cacheKeyForRoot is a stable hash of an absolute path — used when someone
// wants a per-project cache under, say, ~/.cache/semsearch/.
func cacheKeyForRoot(root string) string {
	abs, _ := filepath.Abs(root)
	sum := sha256.Sum256([]byte(abs))
	return hex.EncodeToString(sum[:8])
}
