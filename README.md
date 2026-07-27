# semsearch

**Semantic code search.** Ask a codebase a question in plain English and get
back the code that matches by *meaning* — not just keyword grep.

Two modes:

- **`embed` — real embedding search**, powered by a local Ollama server with
  the free `nomic-embed-text` model. **No API key required.** The index is
  cached on disk; re-runs only re-embed files whose (size, mtime) changed.
  Results are **line-level**, showing which chunk of a file matched.
- **`rank` — LLM ranking** via a hosted model. Sends a compact index to
  Claude and asks it to rank matches. Good for medium codebases with no
  local setup.

Default: `embed` when `-provider ollama`, `rank` when `-provider anthropic`.

```bash
# Free/local: embed once, then query
ollama pull nomic-embed-text
semsearch -provider ollama "where do we validate auth tokens?"
# 1. internal/auth/verify.go:23-56  (78)
#    func VerifyToken(tok string) (Claims, error) {
# 2. middleware/session.go:18-52  (65)
#    signed := r.Header.Get("Authorization")
# ...

# Hosted: LLM-ranking mode
semsearch "the retry/backoff logic"

# Explicit mode override
semsearch -mode embed -embed-model nomic-embed-text "database connection pool"
```

## Install

```bash
go install github.com/Anhedonicc/semsearch@latest   # once published
# or, from this folder:
go build -o semsearch .
```

Then pick a backend:

```bash
# Option A — free & local (no API key)  ← the exciting one
ollama pull nomic-embed-text   # or any other embedding model on the Ollama registry
semsearch -provider ollama "some plain-english query"

# Option B — hosted (Claude does the ranking on a text index)
export ANTHROPIC_API_KEY="sk-ant-..."
semsearch "some plain-english query"
```

## Flags

| Flag | Default | Description |
| --- | --- | --- |
| `-mode` | auto | `embed` (local embeddings) or `rank` (LLM ranking). Defaults per provider. |
| `-provider` | `anthropic` | Backend for `rank` mode: `anthropic` or `ollama` (both keyless in embed mode) |
| `-embed-model` | `nomic-embed-text` | Embedding model id for `embed` mode. Or `SEMSEARCH_EMBED_MODEL` |
| `-model` | per-provider | Chat model id for `rank` mode. Or `ANTHROPIC_MODEL` / `OLLAMA_MODEL` |
| `-cache` | `<path>/.semsearch-cache.json` | Where the embed index is stored |
| `-path` | `.` | Directory to search |
| `-top` | `5` | Number of results |
| `-ext` | (many) | Comma-separated extensions to index |
| `-json` | `false` | Emit results as a JSON array (for scripting) |
| `-no-gitignore` | `false` | Don't read directory names from the root `.gitignore` |
| `-v` | `false` | Print an index summary to stderr (which files were cached vs. embedded) |
| `-snippet` | `1500` | Rank mode: bytes read from the top of each file |
| `-max-files` | `300` | Rank mode: max files in the LLM index |
| `-dry-run` | `false` | Rank mode: print the index without calling the model |

The embed cache defaults to `.semsearch-cache.json` in the searched directory —
add that line to your `.gitignore`. Rebuild from scratch by deleting it.

## How embed mode works

1. Walks the directory, honoring your `.gitignore` for skip-dirs.
2. Splits each text file into overlapping ~40-line chunks.
3. For any chunk not in the cache (new file, or file whose size/mtime changed),
   calls Ollama's `/api/embed` in batches to get a vector per chunk.
4. Saves the cache atomically.
5. Embeds the query, scores every cached chunk by cosine similarity, and
   returns the top-K with their line ranges.

First run over ~2k files takes a few minutes (all-embedding); subsequent
runs are instant unless files have changed.

## Tests

```bash
go test ./...
```

## License

MIT
