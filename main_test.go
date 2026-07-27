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
		{`no array here`, `no array here`}, // pass-through when nothing to extract
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
		"# a comment",
		"",
		"dist/",
		"build",
		".turbo/",
		"*.log",         // glob → ignored
		"src/generated", // path → ignored
		"!keepme",       // negation → ignored
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
	mustWrite("node_modules/pkg/index.js", "// should be skipped")
	mustWrite("bin.dat", "abc\x00def") // NUL → binary → skipped
	mustWrite("image.png", "content")  // wrong extension → skipped
	mustWrite("nested/keep.py", "print('ok')\n")

	files, truncated, err := buildIndex(dir,
		map[string]bool{".go": true, ".py": true, ".dat": true}, // .png excluded → skipped by ext
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
