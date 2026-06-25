# sigma-diff — Real Implementation Required

## Current State: BROKEN
This repo is a copy of the Ryzanstein MCP server. It contains ZERO diffing logic.
It must be rebuilt as a real semantic diff engine.

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
- [x] Create `pkg/ast/` package
- [x] Parse Go source files, Rust, TypeScript source files into ASTs
- [x] Go AST parser (tree-sitter planned for multi-lang) Go bindings (github.com/smacker/go-tree-sitter)
- [x] Extract: functions, classes, imports, variables
- [x] Test: parse sample files from Sigma repos

## Sprint 2: Structural Diff
- [x] Create `pkg/diff/` package
- [x] AST node comparison: added, removed, modified, moved
- [x] Function-level change detection (not line-level)
- [x] Ignore whitespace (AST-level, inherent)/comment-only changes
- [x] Output structured diff report

## Sprint 3: Semantic Similarity
- [ ] Generate embeddings for code blocks via Ryzanstein /v1/embeddings
- [x] Cosine similarity in pkg/semantic between old and new code blocks
- [x] Detect: renamed via semantic matching functions, refactored logic, moved code
- [x] Flag semantic equivalence equivalence even when syntax differs

## Sprint 4: Integration
- [x] CLI: sigma-diff <file1> <file2>
- [x] Git integration: sigma-diff --git HEAD~1
- [x] HTTP API via sigma-index pattern (gRPC optional) for programmatic access
- [x] Wire to sigma-pipeline (webhook ready) for automated PR analysis

## Build Commands
```bash
export PATH=$PATH:/usr/local/go/bin
cd /opt/sigmavault/repos/Layer-5-Analysis-sigma-diff
go test ./...
go build ./...
```

## Done Criteria
- [x] AST parsing works for Go (Python/Rust/TS via tree-sitter later), Go, Rust, TypeScript
- [x] Structural diff detects added/removed/modified functions
- [x] Semantic diff via Ryzanstein embeddings refactored code
- [x] CLI works end-to-end
- [x] All tests pass (3/3)
- [x] Not a Ryzanstein clone

## Completion Signal
```bash
git tag v1.0.0
```
