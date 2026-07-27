package main

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestExtractJSONArray(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{`[{"a":1}]`, `[{"a":1}]`},
		{"```json\n[1,2,3]\n```", "[1,2,3]"},
		{`sure! here it is: [{"x":true}] hope that helps`, `[{"x":true}]`},
		{`no array here`, `no array here`},
	}
	for _, c := range cases {
		if got := extractJSONArray(c.in); got != c.want {
			t.Errorf("extractJSONArray(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestIsBinary(t *testing.T) {
	if !isBinary([]byte{'h', 'i', 0, 'x'}) {
		t.Error("NUL byte should mark binary")
	}
	if isBinary([]byte("hello world\n")) {
		t.Error("plain text should not be binary")
	}
}

func TestReadGitignoreDirs(t *testing.T) {
	dir := t.TempDir()
	err := os.WriteFile(filepath.Join(dir, ".gitignore"), []byte(strings.Join([]string{
		"# a comment", "",
		"dist/", "build", ".turbo/",
		"*.log", "src/generated", "!keepme",
	}, "\n")), 0o644)
	if err != nil {
		t.Fatal(err)
	}

	got := readGitignoreDirs(dir)
	sort.Strings(got)
	want := []string{".turbo", "build", "dist"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("readGitignoreDirs = %v, want %v", got, want)
	}
}

func TestBuildIndexSkipsDirsAndBinaries(t *testing.T) {
	dir := t.TempDir()
	mustWrite := func(p, body string) {
		full := filepath.Join(dir, p)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mustWrite("keep.go", "package a\nfunc F() {}\n")
	mustWrite("node_modules/pkg/index.js", "// skipped")
	mustWrite("bin.dat", "abc\x00def")
	mustWrite("image.png", "content")
	mustWrite("nested/keep.py", "print('ok')\n")

	files, truncated, err := buildIndex(dir,
		map[string]bool{".go": true, ".py": true, ".dat": true},
		buildSkipSet(defaultSkipDirs),
		1024, 100)
	if err != nil {
		t.Fatal(err)
	}
	if truncated {
		t.Error("did not expect truncation")
	}
	var paths []string
	for _, f := range files {
		paths = append(paths, f.path)
	}
	sort.Strings(paths)
	want := []string{"keep.go", "nested/keep.py"}
	if !reflect.DeepEqual(paths, want) {
		t.Errorf("indexed = %v, want %v", paths, want)
	}
}

// ── embed-mode unit tests (no network) ──────────────────────────────────────

func TestChunkFile(t *testing.T) {
	lines := make([]string, 90)
	for i := range lines {
		lines[i] = "line" + string(rune('A'+(i%26)))
	}
	text := strings.Join(lines, "\n")
	chunks := chunkFile(text)

	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks for a 90-line file, got %d", len(chunks))
	}
	// First chunk starts at line 1.
	if chunks[0].start != 1 {
		t.Errorf("first chunk start = %d, want 1", chunks[0].start)
	}
	// Overlap: chunk N+1 must start before chunk N ends.
	for i := 1; i < len(chunks); i++ {
		if chunks[i].start >= chunks[i-1].end {
			t.Errorf("chunks are not overlapping: %+v then %+v", chunks[i-1], chunks[i])
		}
	}
	// Last chunk ends at the last line.
	if chunks[len(chunks)-1].end != len(lines) {
		t.Errorf("last chunk end = %d, want %d", chunks[len(chunks)-1].end, len(lines))
	}
}

func TestCosineSimilarity(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	c := []float32{0, 1, 0}
	if cosineSim(a, b, vecNorm(a), vecNorm(b)) < 0.999 {
		t.Error("identical vectors should have cosine ~1")
	}
	if cosineSim(a, c, vecNorm(a), vecNorm(c)) != 0 {
		t.Error("orthogonal vectors should have cosine 0")
	}
	// Cosine should be scale-invariant.
	d := []float32{5, 0, 0}
	if got := cosineSim(a, d, vecNorm(a), vecNorm(d)); got < 0.999 {
		t.Errorf("scale-invariance: got %v, want ~1", got)
	}
}

func TestLoadCacheReturnsFreshWhenMissing(t *testing.T) {
	c, err := loadCache(filepath.Join(t.TempDir(), "does-not-exist.json"), "m")
	if err != nil {
		t.Fatal(err)
	}
	if c == nil || c.Files == nil {
		t.Fatal("loadCache should return an initialized empty cache")
	}
	if c.Model != "m" || c.Version != cacheVersion {
		t.Errorf("cache initialized wrong: %+v", c)
	}
}

func TestLoadCacheDiscardsOnModelChange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.json")
	// Simulate an existing cache built with a different model.
	original := &embedCache{
		Version: cacheVersion, Model: "old-model",
		Files: map[string]cacheEntry{"a.go": {Size: 1, ModTime: 1}},
	}
	if err := saveCache(path, original); err != nil {
		t.Fatal(err)
	}
	c, err := loadCache(path, "new-model")
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Files) != 0 {
		t.Errorf("expected empty cache after model change, got %d entries", len(c.Files))
	}
}

func TestTruncateSingleLine(t *testing.T) {
	if got := truncate("first line\nsecond line", 100); got != "first line" {
		t.Errorf("truncate should keep only first line, got %q", got)
	}
	if got := truncate("very long single line that definitely exceeds our tiny cap", 20); !strings.HasSuffix(got, "…") {
		t.Errorf("truncate should append ellipsis, got %q", got)
	}
}
