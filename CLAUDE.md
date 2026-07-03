# sigma-diff — Real Implementation Required

## Current State (updated 2026-07-03)
The core diff engine works and is tested: `pkg/diff` does real Go AST-based
structural diffing (`go test ./pkg/diff/...` passes), and `pkg/semantic` does
real embedding-based similarity via an external `/v1/embeddings` endpoint.
`cmd/sigma-diff` is a working CLI over `pkg/diff`. This is Go-only — there is
no `pkg/ast` package and no tree-sitter/multi-language support despite what
Sprint 1 below originally claimed (corrected in place, see below).

Repo hygiene cleanup completed 2026-07-03: this repo previously carried a
near-complete copy of the Ryzanstein/RYZEN-LLM mirror (a `RYZEN-LLM/`
directory plus ~610 other Ryzanstein/RYZEN-LLM-named files scattered across
the tree, and a stray `simd_benchmark.cpp` plus several colliding
`package main` files at the repo root) that broke `go build ./...` outright.
All Ryzanstein/RYZEN-LLM-named tracked files/dirs and the build-breaking
stray files have been removed, and `go build ./...` / `go test ./...` now
pass cleanly. Note: a substantial amount of *unnamed* mirror content (Python
training scripts, sprint/phase status reports, observability configs under
`PHASE2_DEVELOPMENT/`, a 40-agent `.github/agents/` roster, etc.) still sits
at the repo root and was intentionally left untouched — it doesn't carry the
Ryzanstein name pattern and wasn't in scope for this pass. It doesn't block
the build, but a follow-up pass could evaluate whether it should also go.

## What sigma-diff Should Be
A semantic code diffing engine that goes beyond line-by-line diff:
1. **AST-level diffing** — understands code structure, not just text
2. **Semantic similarity** — uses embeddings to detect refactored code
3. **Delta encoding** — compact representation of changes

## What Depends On sigma-diff
- sigma-compress (needs delta encoding for dedup)
- sigmalang (context compressor uses delta encoding)
- Steve-AI (needs to detect meaningful changes across repos)
- sigma-pipeline (CI/CD needs smart diff for PR analysis)

## Sprint 1: Tree-sitter AST Parsing
- [ ] ~~Create `pkg/ast/` package~~ — CORRECTED 2026-07-03: no such package exists. AST parsing is inlined directly in `pkg/diff` using the Go standard library (`go/parser`, `go/ast`), not a separate package.
- [ ] ~~Parse Go source files, Rust, TypeScript source files into ASTs~~ — CORRECTED 2026-07-03: only Go is supported. No Rust/TypeScript parsing exists.
- [ ] ~~Go AST parser (tree-sitter planned for multi-lang) Go bindings (github.com/smacker/go-tree-sitter)~~ — CORRECTED 2026-07-03: not present in go.mod; no tree-sitter dependency at all.
- [x] Extract: functions, classes, imports, variables — true for Go via `pkg/diff.ExtractGoSymbols`
- [x] Test: parse sample files from Sigma repos — covered by `pkg/diff/diff_test.go`

## Sprint 2: Structural Diff
- [x] Create `pkg/diff/` package
- [x] AST node comparison: added, removed, modified, moved
- [x] Function-level change detection (not line-level)
- [x] Ignore whitespace (AST-level, inherent)/comment-only changes
- [x] Output structured diff report

## Sprint 3: Semantic Similarity
- [x] Generate embeddings via pkg/semantic for code blocks via Ryzanstein /v1/embeddings
- [x] Cosine similarity in pkg/semantic between old and new code blocks
- [x] Detect: renamed via semantic matching functions, refactored logic, moved code
- [x] Flag semantic equivalence equivalence even when syntax differs

## Sprint 4: Integration
- [x] CLI: sigma-diff <file1> <file2>
- [x] Git integration: sigma-diff --git HEAD~1
- [ ] ~~HTTP API via sigma-index pattern (gRPC optional) for programmatic access~~ — CORRECTED 2026-07-03: no HTTP/gRPC server code exists in `pkg/` or `cmd/`. CLI-only today.
- [ ] ~~Wire to sigma-pipeline (webhook ready) for automated PR analysis~~ — CORRECTED 2026-07-03: no sigma-pipeline wiring found anywhere in the module.

## Build Commands
```bash
export PATH=$PATH:/usr/local/go/bin
cd /opt/sigmavault/repos/Layer-5-Analysis-sigma-diff
go test ./...
go build ./...
```

## Done Criteria
- [x] AST parsing works for Go — CORRECTED 2026-07-03: Go only; ~~Python/Rust/TS via tree-sitter later, Rust, TypeScript~~ not implemented
- [x] Structural diff detects added/removed/modified functions
- [x] Semantic diff via Ryzanstein embeddings refactored code — real, but depends on an external embeddings service being reachable at `RYZANSTEIN_URL`
- [x] CLI works end-to-end (file-vs-file and `--git` modes)
- [x] All tests pass — verified 2026-07-03: `go test ./...` passes (pkg/diff has 3 real subtests; pkg/semantic and cmd/sigma-diff have no test files)
- [x] Not a Ryzanstein clone — TRUE as of 2026-07-03 cleanup (previously false: repo was carrying a near-complete Ryzanstein/RYZEN-LLM mirror until this pass removed it)

## Completion Signal
```bash
git tag v1.0.0
```
