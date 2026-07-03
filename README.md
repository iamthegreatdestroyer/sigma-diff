# sigma-diff

A semantic and structural code-diffing engine for Go source. It goes beyond
line-based diff by parsing files into their AST and comparing declarations
(functions, types, consts, vars, imports) directly, then optionally using
embeddings to detect code that was refactored — moved or rewritten but still
semantically equivalent.

## What it actually does today

- **`pkg/diff`** — parses two versions of a Go file with the standard
  `go/parser`/`go/ast` packages, extracts top-level symbols, and reports
  `added` / `removed` / `modified` / `moved` changes per symbol instead of
  per line. Real, tested logic — not a stub.
- **`pkg/semantic`** — an HTTP client that calls a `/v1/embeddings` endpoint
  (configurable via `RYZANSTEIN_URL`, defaults to `http://localhost:8000`)
  to embed function bodies and compare them with cosine similarity, so a
  renamed/refactored function can be matched to its predecessor even when
  the text differs. This currently depends on an external embeddings
  service being reachable; it is not a local/offline feature.
- **`cmd/sigma-diff`** — a working CLI built on `pkg/diff`:
  - `sigma-diff <file1> <file2>` — structural diff between two Go files
  - `sigma-diff --git HEAD~1` — diff every changed `.go` file against a git ref
  - set `SIGMA_DIFF_JSON=1` to also emit the diff result as JSON

## What it is not (yet)

- **Go only.** Despite earlier documentation claiming Rust/TypeScript/multi-language
  support via tree-sitter, only Go AST parsing is implemented. There is no
  `pkg/ast` package and no tree-sitter dependency in this module.
- Not wired into sigma-pipeline, sigma-index, or any HTTP/gRPC API — the CLI
  is the only integration point today.

## Build & test

```bash
export PATH=$PATH:/usr/local/go/bin   # if go isn't already on PATH
go build ./...
go test ./...
```

Both commands are expected to pass cleanly. `pkg/diff` has real unit tests
(`TestExtractGoSymbols`, `TestDiffGoFiles`, `TestIdenticalFiles`); `pkg/semantic`
and `cmd/sigma-diff` currently have no test files.

## Repo hygiene note

This repo originated as a fork/mirror of the Ryzanstein LLM project. As of
the cleanup on 2026-07-03, all Ryzanstein/RYZEN-LLM-named files and
directories (the `RYZEN-LLM/` mirror, `*ryzanstein*`/`*ryzen*`-named source
files, and a handful of unnamed leftover Go files at the repo root that
collided on `package main`) have been removed, restoring a clean
`go build ./...`. Some non-Go mirror content (Python scripts, sprint/phase
reports, observability configs, etc.) that don't share the Ryzanstein name
pattern may still be present at the repo root and have not yet been
triaged — see `CLAUDE.md` for the current cleanup scope.
