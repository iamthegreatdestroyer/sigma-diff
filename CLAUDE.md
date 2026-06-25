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
- [ ] Create `pkg/ast/` package
- [ ] Parse Python, Go, Rust, TypeScript source files into ASTs
- [ ] Use tree-sitter Go bindings (github.com/smacker/go-tree-sitter)
- [ ] Extract: functions, classes, imports, variables
- [ ] Test: parse sample files from Sigma repos

## Sprint 2: Structural Diff
- [ ] Create `pkg/diff/` package
- [ ] AST node comparison: added, removed, modified, moved
- [ ] Function-level change detection (not line-level)
- [ ] Ignore whitespace/comment-only changes
- [ ] Output structured diff report

## Sprint 3: Semantic Similarity
- [ ] Generate embeddings for code blocks via Ryzanstein /v1/embeddings
- [ ] Cosine similarity between old and new code blocks
- [ ] Detect: renamed functions, refactored logic, moved code
- [ ] Flag semantic equivalence even when syntax differs

## Sprint 4: Integration
- [ ] CLI: sigma-diff <file1> <file2>
- [ ] Git integration: sigma-diff --git HEAD~1
- [ ] gRPC API for programmatic access
- [ ] Wire to sigma-pipeline for automated PR analysis

## Build Commands
```bash
export PATH=$PATH:/usr/local/go/bin
cd /opt/sigmavault/repos/Layer-5-Analysis-sigma-diff
go test ./...
go build ./...
```

## Done Criteria
- [ ] AST parsing works for Python, Go, Rust, TypeScript
- [ ] Structural diff detects added/removed/modified functions
- [ ] Semantic diff detects refactored code
- [ ] CLI works end-to-end
- [ ] All tests pass
- [ ] Not a Ryzanstein clone

## Completion Signal
```bash
git tag v1.0.0
```
