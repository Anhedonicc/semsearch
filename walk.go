package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
)

// errStopWalk is used internally to short-circuit a WalkDir when we've collected
// enough entries. Wrap-only sentinel — not exported.
var errStopWalk = errors.New("semsearch: stop walk")

// walkFiles walks `root`, skipping any directory whose name is in skipDirs,
// calling fn once per plain file with its slash-normalized path relative to
// root and the absolute path.
func walkFiles(root string, skipDirs map[string]bool, fn func(rel, full string, size int64) error) error {
	return filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		rel, e := filepath.Rel(root, p)
		if e != nil {
			rel = p
		}
		return fn(filepath.ToSlash(rel), p, info.Size())
	})
}

func joinPath(elem ...string) string {
	return filepath.Join(elem...)
}

func extOf(p string) string {
	if i := strings.LastIndexByte(p, '.'); i >= 0 {
		return p[i:]
	}
	return ""
}
