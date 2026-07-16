# semsearch

**Semantic code search.** Ask a question about a codebase in plain
English and get the files that match by *meaning* — ranked and explained by
the model, not just keyword grep.

Two backends: a hosted default, or **Ollama for free local models with no API
key** (`-provider ollama`) — so anyone can run it.

```console
$ semsearch "where do we validate auth tokens?"
1. internal/auth/verify.go  (95)
   Contains the token signature + expiry checks used by the middleware.
2. middleware/session.go  (70)
   Calls the verifier on every request and rejects invalid tokens.
```

## Why it's different from grep

`grep "token"` finds the word "token". `semsearch` understands *intent* — a
query like "the retry/backoff logic" finds the exponential-backoff function even
if it never uses the word "retry".

## Install

Requires Go 1.21+.

```bash
go install github.com/Anhedonicc/semsearch@latest   # once published
# or, from this folder:
go build -o semsearch .
```

Then pick a backend:

```bash
# Option A — hosted (default)
export ANTHROPIC_API_KEY="sk-ant-..."      # key from console.anthropic.com

# Option B — free & local, no API key (https://ollama.com)
ollama pull llama3.2
semsearch -provider ollama "..."
```

## Usage

```bash
semsearch "how is rate limiting implemented?"
semsearch -path ./src -top 3 "the database connection pool"
semsearch -ext .go,.md "where is the config parsed?"
semsearch -dry-run            # show what would be indexed, no API call
```

| Flag | Default | Description |
| --- | --- | --- |
| `-provider` | `anthropic` | Backend: `anthropic` or `ollama` (local, keyless). Or `SEMSEARCH_PROVIDER` |
| `-path` | `.` | Directory to search |
| `-ext` | common code/text types | Comma-separated extensions to index |
| `-top` | `5` | Number of results |
| `-model` | per-provider | Model id (or `ANTHROPIC_MODEL` / `OLLAMA_MODEL`) |
| `-snippet` | `1500` | Bytes read from the top of each file |
| `-max-files` | `300` | Cap on files indexed |
| `-dry-run` | `false` | Build the index and print it; skip the API call |

## How it works

1. Walks the directory, skipping `.git`, `node_modules`, `bin`, etc.
2. Reads a snippet from the top of each text file into a compact index.
3. Sends the query + index to the model, which ranks files by semantic relevance
   and returns structured JSON with a score and reason per file.

**Limitations (v1):** ranking uses a snippet from the top of each file, so a
match buried deep in a large file may be missed. A future version can chunk
files and/or add a true embedding index for larger codebases.

## License

MIT
