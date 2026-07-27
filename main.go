// semsearch — semantic code search.
//
// Two modes:
//
//   - -mode rank   (default when -provider anthropic)
//       Sends a compact index of the codebase to a hosted LLM and asks it
//       to rank the top matches. Good for medium codebases, no local setup.
//
//   - -mode embed  (default when -provider ollama)
//       Embeds every code chunk locally with Ollama's nomic-embed-text (free,
//       no API key), caches the vectors on disk, and scores queries with
//       cosine similarity. Results are line-level. Second run reuses the
//       cache and only re-embeds changed files.
//
//	semsearch "where do we validate auth tokens?"
//	semsearch -provider ollama "the retry/backoff logic"     # embed mode by default
//	semsearch -mode embed -embed-model nomic-embed-text "database connection pool"
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
)

type result struct {
	Path   string  `json:"path"`
	Score  float64 `json:"score"`
	Reason string  `json:"reason"`
}

type indexedFile struct {
	path    string
	snippet string
}

// Baseline set of directories that are never worth searching. Extended at
// runtime by directory-name entries in a root-level .gitignore.
var defaultSkipDirs = []string{
	".git", ".hg", ".svn",
	"node_modules", "vendor", "bower_components",
	"bin", "obj", "target", "dist", "build", "out",
	".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
	".idea", ".vs", ".vscode",
	".next", ".nuxt", ".svelte-kit", ".turbo",
	"coverage", ".cache",
}

func main() {
	path := flag.String("path", ".", "directory to search")
	extCSV := flag.String("ext",
		".go,.py,.cs,.js,.ts,.tsx,.jsx,.java,.rb,.php,.rs,.c,.cpp,.h,.hpp,.md,.txt,.json,.yaml,.yml,.toml,.sh",
		"comma-separated file extensions to index")
	top := flag.Int("top", 5, "number of results to return")
	mode := flag.String("mode", "",
		"search mode: rank (LLM ranking) or embed (local embeddings). Default: rank for anthropic, embed for ollama.")
	provider := flag.String("provider", envOr("SEMSEARCH_PROVIDER", "anthropic"),
		"backend for rank mode: anthropic (default) or ollama (local, keyless)")
	model := flag.String("model", "", "model id for rank mode (defaults per provider)")
	embedModel := flag.String("embed-model", envOr("SEMSEARCH_EMBED_MODEL", "nomic-embed-text"),
		"embedding model id for embed mode (uses local Ollama; run `ollama pull nomic-embed-text` first)")
	cachePath := flag.String("cache", "",
		"embed-mode index cache path (default: <path>/.semsearch-cache.json)")
	snippetBytes := flag.Int("snippet", 1500, "rank mode: max bytes read from the top of each file")
	maxFiles := flag.Int("max-files", 300, "rank mode: max files to include in the LLM index")
	noIgnore := flag.Bool("no-gitignore", false, "don't read directory names from .gitignore")
	asJSON := flag.Bool("json", false, "output results as a JSON array (for scripting)")
	dryRun := flag.Bool("dry-run", false, "rank mode: build the index and print it; don't call the API")
	verbose := flag.Bool("v", false, "print an index summary to stderr")
	flag.Parse()

	query := strings.TrimSpace(strings.Join(flag.Args(), " "))
	if query == "" && !*dryRun {
		fmt.Fprintln(os.Stderr, "usage: semsearch [options] <natural-language query>")
		flag.PrintDefaults()
		os.Exit(2)
	}

	exts := map[string]bool{}
	for _, e := range strings.Split(*extCSV, ",") {
		if e = strings.TrimSpace(e); e != "" {
			exts[strings.ToLower(e)] = true
		}
	}
	skipDirs := buildSkipSet(defaultSkipDirs)
	if !*noIgnore {
		for _, d := range readGitignoreDirs(*path) {
			skipDirs[d] = true
		}
	}

	// Pick the default mode per provider unless overridden.
	if *mode == "" {
		if *provider == "ollama" {
			*mode = "embed"
		} else {
			*mode = "rank"
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	var results []result
	var err error

	switch *mode {
	case "embed":
		results, err = embedSearch(ctx, *path, *cachePath, *embedModel, query, exts, skipDirs, *top, *verbose)
	case "rank":
		results, err = rankSearch(
			ctx, *path, *provider, resolveRankModel(*model, *provider),
			query, exts, skipDirs, *snippetBytes, *maxFiles, *top, *dryRun, *verbose,
		)
	default:
		fmt.Fprintf(os.Stderr, "error: -mode must be rank or embed (got %q)\n", *mode)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	if *dryRun {
		return // rank-mode dry-run already printed its output
	}

	sort.SliceStable(results, func(i, j int) bool { return results[i].Score > results[j].Score })

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(results)
		return
	}
	if len(results) == 0 {
		fmt.Println("No relevant files found.")
		return
	}
	for i, r := range results {
		fmt.Printf("%d. %s  (%.0f)\n   %s\n", i+1, r.Path, r.Score, r.Reason)
	}
}

func resolveRankModel(model, provider string) string {
	if model != "" {
		return model
	}
	if provider == "ollama" {
		return envOr("OLLAMA_MODEL", "llama3.2")
	}
	return envOr("ANTHROPIC_MODEL", "claude-opus-4-8")
}

// ── LLM-ranking mode ─────────────────────────────────────────────────────────

func rankSearch(
	ctx context.Context, root, provider, model, query string,
	exts, skipDirs map[string]bool,
	snippetBytes, maxFiles, top int, dryRun, verbose bool,
) ([]result, error) {
	files, truncated, err := buildIndex(root, exts, skipDirs, snippetBytes, maxFiles)
	if err != nil {
		return nil, err
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no matching text files found under %s", root)
	}
	index := renderIndex(files)

	if verbose || dryRun {
		fmt.Fprintf(os.Stderr, "indexed %d files (%d bytes of context)%s\n",
			len(files), len(index), truncNote(truncated))
	}
	if dryRun {
		for _, f := range files {
			fmt.Println("  ", f.path)
		}
		return nil, nil
	}

	if provider == "anthropic" && os.Getenv("ANTHROPIC_API_KEY") == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY is not set (or use -mode embed, or -provider ollama)")
	}
	return llmRank(ctx, provider, model, query, index, top)
}

func llmRank(ctx context.Context, provider, model, query, index string, top int) ([]result, error) {
	system := "You are a semantic code search engine. You receive a natural-language QUERY " +
		"and an INDEX of files, each with a snippet from the top of the file. Rank files by how " +
		"well they match the intent and meaning of the query, not just keyword overlap. Respond " +
		"with ONLY a JSON array (no prose, no markdown fences): " +
		`[{"path": string, "score": number 0-100, "reason": short string explaining the match}]. ` +
		"Order by score descending. If nothing is relevant, respond with []."

	user := fmt.Sprintf("QUERY: %s\n\nReturn the top %d most relevant files.\n\n=== INDEX ===\n%s",
		query, top, index)

	raw, err := complete(ctx, provider, model, system, user)
	if err != nil {
		return nil, err
	}

	var results []result
	if err := json.Unmarshal([]byte(extractJSONArray(raw)), &results); err != nil {
		return nil, fmt.Errorf("could not parse model response as JSON: %w\nraw: %s", err, raw)
	}
	if len(results) > top {
		results = results[:top]
	}
	return results, nil
}

// ── Chat providers (rank mode) ──────────────────────────────────────────────

func complete(ctx context.Context, provider, model, system, user string) (string, error) {
	if provider == "ollama" {
		return ollamaChat(ctx, model, system, user)
	}
	return anthropicChat(ctx, model, system, user)
}

func anthropicChat(ctx context.Context, model, system, user string) (string, error) {
	client := anthropic.NewClient()
	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: 4096,
		System:    []anthropic.TextBlockParam{{Text: system}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(user)),
		},
	})
	if err != nil {
		return "", err
	}
	var text strings.Builder
	for _, block := range resp.Content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			text.WriteString(tb.Text)
		}
	}
	return text.String(), nil
}

func ollamaChat(ctx context.Context, model, system, user string) (string, error) {
	host := envOr("OLLAMA_HOST", "http://localhost:11434")
	reqBody, _ := json.Marshal(map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"stream": false,
		"format": "json",
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(host, "/")+"/api/chat", bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: 3 * time.Minute}).Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama returned %s", resp.Status)
	}
	var out struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	return out.Message.Content, nil
}

// ── Shared helpers (indexer, gitignore, extraction) ─────────────────────────

func buildSkipSet(names []string) map[string]bool {
	m := map[string]bool{}
	for _, n := range names {
		m[n] = true
	}
	return m
}

// readGitignoreDirs pulls plain directory-name entries out of the root
// .gitignore (`dist/`, `build`, `.turbo/`). Globs and paths are ignored.
// A full gitignore parser would be a whole dependency for modest additional
// value at this scale.
func readGitignoreDirs(root string) []string {
	data, err := os.ReadFile(joinPath(root, ".gitignore"))
	if err != nil {
		return nil
	}
	var out []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "!") {
			continue
		}
		if strings.ContainsAny(line, "*?[") || strings.Contains(line, "/") && !strings.HasSuffix(line, "/") {
			continue
		}
		line = strings.TrimSuffix(line, "/")
		if line == "" || strings.Contains(line, "/") {
			continue
		}
		out = append(out, line)
	}
	return out
}

func buildIndex(root string, exts, skipDirs map[string]bool, snippetBytes, maxFiles int) ([]indexedFile, bool, error) {
	var out []indexedFile
	truncated := false
	err := walkFiles(root, skipDirs, func(rel, full string, size int64) error {
		if len(out) >= maxFiles {
			truncated = true
			return errStopWalk
		}
		if !exts[strings.ToLower(extOf(rel))] || size > 512*1024 {
			return nil
		}
		data, err := os.ReadFile(full)
		if err != nil || isBinary(data) {
			return nil
		}
		snip := string(data)
		if len(snip) > snippetBytes {
			snip = snip[:snippetBytes]
		}
		out = append(out, indexedFile{path: rel, snippet: snip})
		return nil
	})
	if err == errStopWalk {
		err = nil
	}
	return out, truncated, err
}

func isBinary(b []byte) bool {
	n := len(b)
	if n > 8000 {
		n = 8000
	}
	for i := 0; i < n; i++ {
		if b[i] == 0 {
			return true
		}
	}
	return false
}

func renderIndex(files []indexedFile) string {
	var sb strings.Builder
	for _, f := range files {
		sb.WriteString("### ")
		sb.WriteString(f.path)
		sb.WriteString("\n")
		sb.WriteString(f.snippet)
		sb.WriteString("\n\n")
	}
	return sb.String()
}

func extractJSONArray(s string) string {
	start := strings.Index(s, "[")
	end := strings.LastIndex(s, "]")
	if start >= 0 && end > start {
		return s[start : end+1]
	}
	return s
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func truncNote(t bool) string {
	if t {
		return " [truncated: hit -max-files]"
	}
	return ""
}
