// semsearch — semantic code search.
//
// Ask a question about a codebase in plain English and get back the files that
// match by *meaning*, ranked and explained by the model — not just keyword grep.
//
//	semsearch "where do we validate auth tokens?"
//	semsearch -path ./src -top 3 "the retry/backoff logic"
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

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

// Directories that are never worth searching.
var skipDirs = map[string]bool{
	".git": true, "node_modules": true, "vendor": true, "bin": true,
	"obj": true, ".venv": true, "venv": true, "dist": true, "build": true,
	".idea": true, ".vs": true, "target": true, "__pycache__": true,
}

func main() {
	path := flag.String("path", ".", "directory to search")
	extCSV := flag.String("ext",
		".go,.py,.cs,.js,.ts,.tsx,.jsx,.java,.rb,.php,.rs,.c,.cpp,.h,.hpp,.md,.txt,.json,.yaml,.yml,.toml,.sh",
		"comma-separated file extensions to index")
	top := flag.Int("top", 5, "number of results to return")
	model := flag.String("model", envOr("SEMSEARCH_MODEL", "claude-opus-4-8"), "model id")
	snippetBytes := flag.Int("snippet", 1500, "max bytes read from the top of each file")
	maxFiles := flag.Int("max-files", 300, "max files to index")
	dryRun := flag.Bool("dry-run", false, "build the index and print it; do not call the API")
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

	files, truncated, err := buildIndex(*path, exts, *snippetBytes, *maxFiles)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "no matching text files found under", *path)
		os.Exit(1)
	}

	index := renderIndex(files)

	if *dryRun {
		fmt.Printf("indexed %d files (%d bytes of context)%s\n", len(files), len(index), truncNote(truncated))
		for _, f := range files {
			fmt.Println("  ", f.path)
		}
		return
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		fmt.Fprintln(os.Stderr, "error: ANTHROPIC_API_KEY is not set")
		os.Exit(1)
	}

	results, err := search(context.Background(), *model, query, index, *top)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	if len(results) == 0 {
		fmt.Println("No relevant files found.")
		return
	}
	sort.SliceStable(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	for i, r := range results {
		fmt.Printf("%d. %s  (%.0f)\n   %s\n", i+1, r.Path, r.Score, r.Reason)
	}
}

func buildIndex(root string, exts map[string]bool, snippetBytes, maxFiles int) ([]indexedFile, bool, error) {
	var out []indexedFile
	truncated := false
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries
		}
		if d.IsDir() {
			if skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if len(out) >= maxFiles {
			truncated = true
			return filepath.SkipAll
		}
		if !exts[strings.ToLower(filepath.Ext(p))] {
			return nil
		}
		info, err := d.Info()
		if err != nil || info.Size() > 512*1024 {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil || isBinary(data) {
			return nil
		}
		snip := string(data)
		if len(snip) > snippetBytes {
			snip = snip[:snippetBytes]
		}
		rel, e := filepath.Rel(root, p)
		if e != nil {
			rel = p
		}
		out = append(out, indexedFile{path: filepath.ToSlash(rel), snippet: snip})
		return nil
	})
	return out, truncated, err
}

// isBinary reports whether data looks non-textual (contains a NUL byte).
func isBinary(b []byte) bool {
	n := min(len(b), 8000)
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

func search(ctx context.Context, model, query, index string, top int) ([]result, error) {
	client := anthropic.NewClient()

	system := "You are a semantic code search engine. You receive a natural-language QUERY " +
		"and an INDEX of files, each with a snippet from the top of the file. Rank files by how " +
		"well they match the intent and meaning of the query, not just keyword overlap. Respond " +
		"with ONLY a JSON array (no prose, no markdown fences): " +
		`[{"path": string, "score": number 0-100, "reason": short string explaining the match}]. ` +
		"Order by score descending. If nothing is relevant, respond with []."

	user := fmt.Sprintf("QUERY: %s\n\nReturn the top %d most relevant files.\n\n=== INDEX ===\n%s",
		query, top, index)

	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: 4096,
		System:    []anthropic.TextBlockParam{{Text: system}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(user)),
		},
	})
	if err != nil {
		return nil, err
	}

	var text strings.Builder
	for _, block := range resp.Content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			text.WriteString(tb.Text)
		}
	}

	js := extractJSONArray(text.String())
	var results []result
	if err := json.Unmarshal([]byte(js), &results); err != nil {
		return nil, fmt.Errorf("could not parse model response as JSON: %w\nraw: %s", err, text.String())
	}
	if len(results) > top {
		results = results[:top]
	}
	return results, nil
}

// extractJSONArray pulls the outermost [...] out of a response, tolerating any
// stray prose or code fences the model might wrap around it.
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
