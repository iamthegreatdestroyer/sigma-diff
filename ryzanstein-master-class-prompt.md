# RYZANSTEIN ECOSYSTEM: 18 DEPENDENCIES MASTER CLASS PROMPT
## For VS Code Copilot (Claude LLM Backend)

---

## EXECUTIVE DIRECTIVE

You are the **Autonomous Dependency Scaffolder** for the Ryzanstein Unified Architecture ecosystem. Your mission: **Transform the 18 novel dependencies outlined in the "Novel Dependency Architecture" document into fully integrated, production-ready codebases with zero manual intervention beyond this single prompt**.

**Success Criteria:**
- All 18 dependencies have complete project scaffolding (no partial stubs)
- Each dependency integrates seamlessly into its corresponding GitHub repository
- File hierarchies follow Ryzanstein conventions and language-specific best practices
- CI/CD pipelines, type definitions, and mock implementations are generated automatically
- Integration points with existing Ryzanstein components (Ryot LLM, ΣLANG, ΣVAULT) are code-ready
- Every file includes docstrings, license headers, and inline type annotations

---

## PHASE 0: STRATEGIC CONTEXT

### Dependency Taxonomy
You will scaffold **three tiers** of dependencies:

**TIER 1 - ECOSYSTEM-LOCKED (Free, OSS) [6 deps]**
- `σ-index` — Succinct semantic code index
- `σ-diff` — Behavioral semantic diff engine
- `mcp-mesh` — MCP server-to-server mesh
- `σ-compress` — Cross-artifact semantic deduplication
- `vault-git` — End-to-end encrypted version control
- `σ-telemetry` — Semantic telemetry sketching

**TIER 2 - STANDALONE COMMERCIAL (Freemium/Paid) [6 deps]**
- `causedb` — Causal debugging engine
- `intent.spec` — Intent-verified development platform
- `flowstate` — Cognitive load IDE monitor
- `dep-bloom` — Sub-linear dependency resolution accelerator
- `archaeo` — Decision archaeology platform
- `cpu-infer` — Universal CPU inference middleware

**TIER 3 - HYBRID (Transform with Ryzanstein) [6 deps]**
- `σ-api` — API specification semantic compressor
- `zkaudit` — Zero-knowledge code property verification
- `agentmem` — Cross-agent episodic memory protocol
- `semlog` — Semantic log compression
- `ann-hybrid` — Unified sub-linear search index
- `neurectomy-shell` — Encrypted confidential development environment

### Language Distribution Strategy
```
Rust:       σ-index, σ-diff, σ-compress, vault-git, σ-telemetry, 
            cpu-infer, σ-api, zkaudit, semlog, ann-hybrid, neurectomy-shell
            (11 deps) → WASM-compiled VS Code extensions + performance-critical cores

Go:         mcp-mesh, causedb, agentmem, semlog, neurectomy-shell
            (5 deps + co-implementations) → MCP infrastructure, server orchestration, Wails apps

TypeScript: σ-index, σ-diff, causedb, intent.spec, flowstate, archaeo, 
            σ-api, zkaudit, agentmem, ann-hybrid
            (10 deps + co-implementations) → VS Code extension UIs, developer-facing APIs

Python:     σ-compress, σ-diff, intent.spec, flowstate, archaeo, agentmem, 
            σ-telemetry, semlog
            (8 deps + co-implementations) → ML pipelines, agent logic, analysis
```

### Repository Structure
Each dependency gets its own GitHub repository following the pattern:
```
github.com/ryzanstein/{dependency-name}
├── Rust core (if applicable)
├── Go sidecar/server (if applicable)
├── TypeScript VS Code extension (if applicable)
├── Python ML/analysis pipeline (if applicable)
├── Comprehensive tests (90%+ coverage required)
├── CI/CD workflows (GitHub Actions)
└── Integration hooks into main Ryzanstein projects
```

---

## PHASE 1: PRE-SCAFFOLDING VALIDATION

Before generating a single file, confirm:

1. **Dependency Declaration**: For each of the 18 dependencies, identify:
   - Primary language (Rust/Go/TypeScript/Python)
   - Co-implementation languages
   - Core algorithm/primitive it uses
   - Integration points with existing Ryzanstein components
   - Standalone vs. hybrid capability

2. **Ryzanstein Component Mapping**: Validate dependencies against existing projects:
   - **Ryot LLM** (core inference engine)
   - **ΣLANG** (semantic compression system)
   - **ΣVAULT** (polymorphic encryption containers)
   - **Elite Agent Collective** (40 specialized agents)
   - **IDE Neurectomy** (VS Code extension + desktop app)

3. **Integration Checkpoints**: Confirm these exist for integration:
   - Type definitions from ΣLANG available for import
   - ΣVAULT's polymorphic encryption API documented
   - Ryot LLM's inference endpoint URL/local binary available
   - MCP protocol specifications for agent interaction
   - VS Code Extension API compatibility

---

## PHASE 2: CORE SCAFFOLDING DIRECTIVES

### Directive 2.1: Project Initialization Template

For **each of the 18 dependencies**, generate this directory structure:

```
{dependency-name}/
├── README.md                           # Project overview, feature list, integration guide
├── ARCHITECTURE.md                     # Component design, data flow diagrams, algorithms
├── INTEGRATION.md                      # How this integrates with Ryot/ΣLANG/ΣVAULT/Elite Agents
├── LICENSE (AGPL-3.0 for Tier 1, Proprietary for Tier 2-3)
├── .gitignore (language-specific)
├── Cargo.toml (if Rust)                # With comprehensive dependency list
├── go.mod / go.sum (if Go)             # With version pinning
├── package.json (if TypeScript/Node)   # With exact version constraints
├── pyproject.toml (if Python)          # With pyenv/conda specs
├── .github/workflows/
│   ├── ci.yml                          # Test, lint, build on every commit
│   ├── release.yml                     # Auto-semantic-versioning + PyPI/crates.io publish
│   └── integration-test.yml            # Test integration with Ryzanstein components
├── src/ (Rust)
│   ├── lib.rs                          # Public API exports
│   ├── core/                           # Core algorithm implementations
│   ├── ffi/                            # FFI bindings for other languages
│   └── wasm/                           # WASM compilation target (if VS Code extension)
├── pkg/ (Go)
│   ├── main.go                         # If standalone binary/server
│   ├── api.go                          # Public interfaces
│   └── {module}/                       # Functional modules
├── src/ (TypeScript)
│   ├── index.ts                        # Public API
│   ├── extension.ts                    # VS Code extension entry point (if applicable)
│   ├── commands/                       # VS Code commands
│   ├── views/                          # WebView components (Svelte)
│   └── {module}/                       # Functional modules
├── python/
│   ├── {dependency}/ (or use src layout)
│   ├── __init__.py
│   ├── ml/                             # ML model training pipelines
│   ├── analysis/                       # Analysis engines
│   └── agents/                         # LLM/agent integration
├── tests/
│   ├── unit/                           # Language-specific unit tests
│   ├── integration/                    # Integration with Ryzanstein components
│   └── benchmarks/                     # Performance benchmarks
├── docs/
│   ├── api.md                          # API reference (auto-generated if possible)
│   ├── algorithms.md                   # Algorithm details with proofs/complexity
│   ├── examples/                       # Usage examples
│   └── deployment/                     # Deployment guides
├── examples/
│   └── {example-name}/                 # Runnable examples
├── vscode-extension/ (if applicable)
│   ├── package.json
│   ├── src/extension.ts
│   ├── src/views/                      # Webviews
│   └── media/                          # Icons, styles
└── docker/ (if applicable)
    ├── Dockerfile
    └── docker-compose.yml              # For local testing with Ryzanstein stack
```

### Directive 2.2: Mandatory File Generation for Every Dependency

Generate these files **without exception** for every single dependency:

#### A. Type Definitions & Public API (`api.ts` / `lib.rs` / `api.go`)

**Requirements:**
- Use strict TypeScript (no `any` types)
- Rust: impl `Send + Sync` traits explicitly
- Go: interface-first design
- All public functions have comprehensive JSDoc/rustdoc/GoDoc
- Export all types used in public APIs
- Define custom error types
- Include version exports (`const VERSION = "..."`)

**Template Structure:**
```rust
// Rust example
/// Public API for σ-index semantic code search
/// 
/// # Example
/// ```ignore
/// let index = SemanticIndex::new("/path/to/codebase")?;
/// let results = index.search_semantic("find functions that retry")?;
/// ```
pub struct SemanticIndex { ... }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult { ... }

pub type Result<T> = std::result::Result<T, IndexError>;

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("IO error: {0}")] IoError(#[from] std::io::Error),
    // ... other variants
}
```

**Template Structure:**
```typescript
// TypeScript example
/**
 * Public API for σ-index semantic code search
 * @example
 * const index = new SemanticIndex("/path/to/codebase");
 * const results = await index.searchSemantic("find functions that retry");
 */
export class SemanticIndex {
  // implementation
}

export interface SearchResult {
  filePath: string;
  lineNumber: number;
  // ... other fields
}

export enum IndexError {
  // error variants
}
```

#### B. Integration Module (`ryzanstein-integration.rs` / `ryzanstein-integration.ts`)

**For ALL dependencies**, generate a dedicated integration module that:

1. **Defines imports from Ryzanstein components:**
   ```rust
   // Example: σ-index needs Ryot + ΣLANG
   use ryot_llm::InferenceEngine;
   use sigma_lang::SemanticEncoder;
   ```

2. **Exposes integration hooks:**
   ```rust
   pub trait RyzansteinIntegrable {
       fn initialize_with_ryot(engine: &InferenceEngine) -> Result<Self>;
       fn use_sigma_encoder(&mut self, encoder: SemanticEncoder) -> Result<()>;
       fn with_sigma_vault(vault: &SigmaVaultClient) -> Result<Self>;
   }
   ```

3. **Includes mock implementations:**
   ```rust
   #[cfg(test)]
   mod mocks {
       pub struct MockInferenceEngine { ... }
       pub struct MockSemanticEncoder { ... }
   }
   ```

#### C. Comprehensive Test Suite (`tests/`)

**Required test categories** (minimum 90% coverage):

1. **Unit Tests** — Pure function tests, no dependencies
2. **Integration Tests** — With mocked Ryzanstein components
3. **End-to-End Tests** — Full stack integration (if applicable)
4. **Benchmark Tests** — Complexity validation (especially for Tier 1 algorithmic dependencies)
5. **Property-Based Tests** — Use QuickCheck (Rust), Hypothesis (Python), or fast-check (TypeScript)

**Benchmark template (Rust):**
```rust
#[bench]
fn bench_search_on_500k_loc(b: &mut Bencher) {
    let index = SemanticIndex::new(&test_codebase())?;
    b.iter(|| index.search_semantic("find retries"));
    // Assert: < 5ms for 500K LOC
}
```

#### D. CI/CD Workflows (`.github/workflows/`)

**Standard workflows for all dependencies:**

```yaml
# ci.yml — on every push/PR
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo test --all --verbose (if Rust)
      - run: npm test (if TypeScript)
      - run: pytest (if Python)
      - uses: codecov/codecov-action@v3  # 90%+ coverage gate
  
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo clippy -- -D warnings (if Rust)
      - run: npm run lint (if TypeScript)
      - run: black --check . (if Python)

# release.yml — on version tag
name: Release
on:
  push:
    tags: ['v*']
jobs:
  publish:
    # Publish to crates.io, npm, PyPI based on language
```

#### E. README.md with Integration Instructions

Every README must include:

```markdown
# {dependency-name}

## Overview
[Brief description + value proposition]

## Integration with Ryzanstein

### With Ryot LLM
[Specific integration points + example]

### With ΣLANG
[How semantic compression is used]

### With ΣVAULT
[Encryption/secure storage use cases]

### With Elite Agent Collective
[How this is exposed as MCP server if applicable]

## Getting Started
[Installation + quick-start code example]

## API Reference
[Auto-generated from docstrings]

## Benchmarks
[Performance claims with validation]

## Development
[Contribution guidelines + local setup]
```

#### F. Dockerfile & docker-compose.yml (where applicable)

**For integration testing**, every dependency (especially those requiring Ryzanstein) gets:

```dockerfile
FROM rust:latest as builder
# Build the dependency
FROM debian:bookworm-slim
# Include Ryot LLM, ΣLANG, etc. as services
```

```yaml
version: '3.9'
services:
  {dependency-name}:
    build: .
    depends_on:
      - ryot-llm
      - sigma-vault
```

---

## PHASE 3: TIER-SPECIFIC SCAFFOLDING

### Tier 1: Ecosystem-Locked (OSS) Dependencies

**For each of the 6 Tier 1 dependencies** (`σ-index`, `σ-diff`, `mcp-mesh`, `σ-compress`, `vault-git`, `σ-telemetry`):

#### A. License & Community Setup
```
LICENSE: AGPL-3.0
CONTRIBUTING.md: Community contribution guidelines
CODE_OF_CONDUCT.md: Standard open-source CoC
ROADMAP.md: Public feature roadmap
CHANGELOG.md: Release notes
```

#### B. Algorithm Documentation (`ARCHITECTURE.md`)

Each Tier 1 dependency centers on a **novel algorithm**. Document exhaustively:

```markdown
# Architecture: σ-index

## Algorithms

### 1. FM-Index on Wavelet Trees
- Paper reference: [URL]
- Complexity: O(m) pattern matching, O(log n) rank/select queries
- Implementation notes: [specific optimizations]
- Proof of correctness: [why this works for code search]

### 2. HNSW ANN Layer
- Construction: O(n log n) expected, O(n log^2 n) worst-case
- Query: O(log n) expected nearest neighbors
- Tuning: M=16, ef_construction=200 chosen because...

### 3. Tree-Sitter Incremental Parsing
- Delta update strategy: only changed subtrees re-encoded
- Complexity: O(|delta|) per file change
```

#### C. Code Generation for Algorithm Implementations

For **Rust algorithm implementations**, automatically scaffold:

```rust
// src/core/{algorithm_name}.rs

/// {AlgorithmName} implementation with comprehensive documentation
///
/// # Complexity
/// - Construction: O(...)
/// - Query: O(...)
/// - Space: O(...)
///
/// # References
/// - [Paper Title] (URL)
/// - [Implementation notes]
pub struct {AlgorithmName} {
    // fields with inline comments explaining each
}

impl {AlgorithmName} {
    /// Create new instance with given parameters
    pub fn new(param1: Type1, param2: Type2) -> Result<Self> {
        // Validation
        ensure!(param1 > 0, "param1 must be positive");
        
        // Construction
        Ok(Self { ... })
    }

    /// Main operation with documented complexity
    pub fn operation(&self, input: &Input) -> Result<Output> {
        // Implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correctness_basic() { ... }

    #[test]
    fn test_edge_cases() { ... }

    #[test]
    #[ignore]  // Run with: cargo test -- --ignored --nocapture
    fn bench_small_input() { ... }
}
```

#### D. FFI Bindings (for Rust → Go/TypeScript)

Every Rust Tier 1 dependency needs language bindings:

```rust
// src/ffi/mod.rs

#[cfg(feature = "ffi")]
pub mod c_api {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_void};

    /// C-compatible opaque handle
    #[repr(transparent)]
    pub struct IndexHandle(*mut std::ffi::c_void);

    /// Create new index (C callable)
    #[no_mangle]
    pub extern "C" fn sigma_index_new(
        path: *const c_char,
    ) -> *mut IndexHandle {
        // Marshalling logic
    }

    /// Perform search (C callable)
    #[no_mangle]
    pub extern "C" fn sigma_index_search(
        handle: *mut IndexHandle,
        query: *const c_char,
    ) -> *mut c_char {
        // Marshalling logic
    }
}
```

Generate **Go bindings** via cgo:
```go
// pkg/cgo_bindings.go
/*
#cgo LDFLAGS: -L./target/release -lsigma_index
#include "./target/release/sigma_index.h"
*/
import "C"

type Index struct {
    handle *C.IndexHandle
}

func (i *Index) Search(query string) ([]Result, error) {
    // Cgo marshalling
}
```

Generate **TypeScript bindings** via wasm-bindgen:
```rust
// src/wasm/lib.rs
#[wasm_bindgen]
pub struct Index { ... }

#[wasm_bindgen]
impl Index {
    #[wasm_bindgen(constructor)]
    pub fn new(path: &str) -> Result<Index, JsValue> { ... }

    pub fn search(&self, query: &str) -> Vec<Result> { ... }
}
```

#### E. WASM Compilation Target

Every Tier 1 dependency that will be a VS Code extension gets:

```toml
# Cargo.toml additions
[lib]
crate-type = ["cdylib", "rlib"]

[profile.release]
opt-level = "z"     # Optimize for size
lto = true
strip = true

[target.'cfg(target_arch = "wasm32")']
# WASM-specific optimizations
```

```bash
# Build script
#!/bin/bash
wasm-pack build --target bundler --release
npm run build:vscode-extension
```

---

### Tier 2: Standalone Commercial Dependencies

**For each of the 6 Tier 2 dependencies** (`causedb`, `intent.spec`, `flowstate`, `dep-bloom`, `archaeo`, `cpu-infer`):

#### A. Monetization Scaffolding

Create these files **automatically**:

```
├── monetization/
│   ├── LICENSE_AGREEMENT.md         # Free vs. Pro vs. Enterprise tiers
│   ├── license-key-validator.ts     # License validation logic
│   ├── telemetry/
│   │   ├── usage-tracker.ts         # Anonymous usage analytics
│   │   └── telemetry-schema.json    # What data we collect
│   └── pricing-tiers.json           # Feature matrix
```

**License key validation template (TypeScript):**

```typescript
/**
 * Validates license keys for paid tiers.
 * Implements local offline validation via Ed25519 signatures.
 */
export class LicenseValidator {
    private publicKey: string;
    
    /**
     * Validate a license key without external calls
     */
    validate(key: string): LicenseInfo | null {
        // Ed25519 signature verification
        // Returns: {tier: 'free' | 'pro' | 'enterprise', expiresAt: Date, ...}
    }
}
```

**Pricing tiers JSON:**

```json
{
  "tiers": [
    {
      "id": "free",
      "name": "Free",
      "price": "$0",
      "features": {
        "basicDebugging": true,
        "distributedSystems": false,
        "teamFeatures": false
      },
      "limits": {
        "recordedTraces": 10,
        "teamMembers": 1
      }
    },
    {
      "id": "pro",
      "name": "Professional",
      "price": "$20/month",
      "features": {
        "basicDebugging": true,
        "distributedSystems": true,
        "teamFeatures": false
      }
    },
    {
      "id": "enterprise",
      "name": "Enterprise",
      "price": "Custom",
      "features": { "all": true },
      "sso": true,
      "auditTrail": true,
      "onPremises": true
    }
  ]
}
```

#### B. Telemetry & Analytics

Every Tier 2 dependency gets anonymized usage tracking:

```typescript
// src/telemetry/usage-tracker.ts

/**
 * Anonymous usage tracking (privacy-first).
 * No code/data is sent to servers.
 * Only feature usage counts and error events.
 */
export class UsageTracker {
    private events: Event[] = [];
    
    trackFeatureUsage(featureName: string, duration_ms: number) {
        this.events.push({
            timestamp: Date.now(),
            feature: featureName,
            duration_ms,
            sessionId: this.getOrCreateSessionId()
        });
    }
    
    private flushBatch(): void {
        // Hash everything before sending
        // Aggregate daily, not per-action
    }
}
```

#### C. VS Code Extension Structure (if applicable)

```
vscode-extension/
├── package.json                      # Define activation events, commands, settings
├── src/
│   ├── extension.ts                  # Extension lifecycle
│   ├── commands/
│   │   ├── {feature-name}.ts        # One file per major feature
│   └── views/
│       ├── {panel-name}.ts          # Webview panels
│       └── {panel-name}.svelte      # Svelte UI components
├── media/
│   ├── icons/                        # Feature icons
│   ├── styles.css                    # Consistent styling
│   └── welcome.html                  # Welcome screen
└── dist/                             # Compiled output
```

**package.json activation pattern:**

```json
{
  "activationEvents": [
    "onStartupFinished",
    "onCommand:causedb.startDebugging",
    "onView:causedb-causal-graph"
  ],
  "contributes": {
    "commands": [
      {
        "command": "causedb.startDebugging",
        "title": "Start Causal Debugging Session"
      }
    ],
    "views": {
      "debug": [
        {
          "id": "causedb-causal-graph",
          "name": "Causal Graph"
        }
      ]
    ],
    "configuration": {
      "properties": {
        "causedb.enableTelemetry": {
          "type": "boolean",
          "default": false,
          "description": "Enable anonymous usage tracking"
        }
      }
    }
  }
}
```

#### D. Standalone Verification Suite

Each Tier 2 dependency needs to prove it works **without Ryzanstein**:

```bash
# tests/standalone.test.ts

describe('Tier 2: Standalone Functionality', () => {
    test('flowstate works without Ryzanstein connection', async () => {
        const flowstate = new Flowstate({
            useRyzanstein: false,  // Explicitly disable
        });
        
        const cognitiveLoad = await flowstate.measureCognitiveLoad();
        expect(cognitiveLoad).toBeGreaterThan(0);
    });
});
```

---

### Tier 3: Hybrid Dependencies

**For each of the 6 Tier 3 dependencies** (`σ-api`, `zkaudit`, `agentmem`, `semlog`, `ann-hybrid`, `neurectomy-shell`):

#### A. Feature Gates for Ryzanstein Enhancement

Every Tier 3 dependency needs **conditional compilation** for Ryzanstein features:

```rust
// Cargo.toml
[features]
default = ["standalone"]
standalone = []
with-ryzanstein = ["ryot-llm", "sigma-lang", "sigma-vault"]

# With Ryzanstein features, compress to 30-50x
# Without Ryzanstein, compress to 10-20x
```

```rust
// src/lib.rs
#[cfg(feature = "with-ryzanstein")]
use ryot_llm::InferenceEngine;

pub struct Compressor {
    #[cfg(feature = "with-ryzanstein")]
    ryzanstein: Option<RyzansteinClient>,
    base_encoder: BaseEncoder,
}

impl Compressor {
    pub fn compress(&self, input: &str) -> CompressedOutput {
        #[cfg(feature = "with-ryzanstein")]
        if let Some(ryz) = &self.ryzanstein {
            return self.compress_with_semantics(input, ryz);
        }
        
        // Fallback to standalone compression
        self.compress_standalone(input)
    }
}
```

#### B. Capability Discovery Mechanism

Hybrid dependencies auto-detect available Ryzanstein components at runtime:

```typescript
// src/ryzanstein-discovery.ts

export interface RyzansteinCapabilities {
    hasRyotLLM: boolean;
    hasSigmaLang: boolean;
    hasSigmaVault: boolean;
    eliteAgentCount: number;
}

export async function discoverRyzanstein(): Promise<RyzansteinCapabilities> {
    const capabilities = {
        hasRyotLLM: await isRyotLLMAvailable(),
        hasSigmaLang: await isSigmaLangAvailable(),
        hasSigmaVault: await isSigmaVaultAvailable(),
        eliteAgentCount: await getEliteAgentCount(),
    };
    
    logCapabilities(capabilities);
    return capabilities;
}
```

#### C. Graceful Degradation Tests

Every Tier 3 dependency has tests that verify **both standalone and enhanced modes**:

```typescript
describe('Tier 3: Hybrid Functionality', () => {
    test('σ-api compresses 10-20x without Ryzanstein', async () => {
        const spec = loadOpenAPISpec();
        const compressed = await compress(spec, { useRyzanstein: false });
        const ratio = spec.size / compressed.size;
        expect(ratio).toBeGreaterThanOrEqual(10);
        expect(ratio).toBeLessThanOrEqual(20);
    });
    
    test('σ-api compresses 30-50x with Ryzanstein', async () => {
        const spec = loadOpenAPISpec();
        const compressed = await compress(spec, { useRyzanstein: true });
        const ratio = spec.size / compressed.size;
        expect(ratio).toBeGreaterThanOrEqual(30);
        expect(ratio).toBeLessThanOrEqual(50);
    });
});
```

---

## PHASE 4: RYZANSTEIN INTEGRATION HOOKS

### Directive 4.1: Standard Integration Points

**Every dependency must define these integration hooks:**

#### A. Type Import Contracts

Each dependency that uses Ryzanstein types creates an `integration-types.ts` that re-exports only what it needs:

```typescript
// src/integration-types.ts
// Minimal public API surface for Ryzanstein types to prevent circular deps

export type {
    InferenceResult,
    EmbeddingVector,
    CompressionRatio,
} from '@ryzanstein/core';

export type { SemanticEncoder } from '@ryzanstein/sigma-lang';
export type { VaultClient } from '@ryzanstein/sigma-vault';
```

#### B. MCP Server Registration (for Elite Agents)

Dependencies that expose agent capabilities register as MCP servers:

```go
// pkg/mcp_server.go (for agentmem, causedb, intent.spec, etc.)

package mcp

import "github.com/ryzanstein/mcp-mesh/server"

// Register this dependency as an MCP server
func RegisterServer(config MCPConfig) (*server.Server, error) {
    s := server.New(server.Config{
        Name: "causedb",
        Description: "Causal debugging agent",
        Tools: []server.Tool{
            {
                Name: "analyze_causal_chain",
                Description: "Analyze root causes from execution trace",
                InputSchema: CausalAnalysisSchema(),
            },
        },
    })
    
    return s, nil
}
```

#### C. ΣLANG Semantic Encoder Integration

For dependencies using semantic compression:

```rust
// src/sigma_lang_integration.rs

use sigma_lang::{SemanticEncoder, SemanticVector};

pub struct SemanticCompressor {
    encoder: SemanticEncoder,
}

impl SemanticCompressor {
    pub async fn compress_semantically(&self, content: &str) -> Result<Vec<SemanticVector>> {
        let embeddings = self.encoder.encode_batch(&[content]).await?;
        Ok(embeddings)
    }
}
```

#### D. ΣVAULT Integration (encrypted storage)

For dependencies using encrypted vaults:

```rust
// src/vault_integration.rs

use sigma_vault::{VaultClient, PolymorphicContainer};

pub async fn store_with_encryption(
    client: &VaultClient,
    data: &str,
    key: &str,
) -> Result<PolymorphicContainer> {
    // Store in all three forms: searchable, computable, provable
    let container = client.create_polymorphic_container(
        data,
        PolymorphicMode::SearchableAndComputable,
    ).await?;
    
    Ok(container)
}
```

#### E. Ryot LLM Inference Hooks

For dependencies that need local inference:

```rust
// src/ryot_integration.rs

use ryot_llm::InferenceEngine;

pub async fn generate_explanation(
    engine: &InferenceEngine,
    context: &str,
) -> Result<String> {
    let response = engine.infer(
        &format!("Explain this code change:\n{}", context),
        InferenceConfig::default()
            .with_max_tokens(200)
            .with_temperature(0.3),  // Deterministic
    ).await?;
    
    Ok(response.text)
}
```

### Directive 4.2: Health Check & Diagnostics

Every dependency generates a health-check module:

```rust
// src/health.rs

pub async fn health_check() -> HealthStatus {
    let mut status = HealthStatus::default();
    
    // Check Ryzanstein availability
    if let Ok(_) = check_ryot_llm().await {
        status.ryot_available = true;
    }
    if let Ok(_) = check_sigma_lang().await {
        status.sigma_lang_available = true;
    }
    if let Ok(_) = check_sigma_vault().await {
        status.sigma_vault_available = true;
    }
    
    // Dependency-specific checks
    match self {
        Self::SigmaIndex => {
            status.additional_checks.insert(
                "fm_index_loaded",
                check_fm_index().await.is_ok()
            );
        }
        // ... more checks
    }
    
    status
}
```

---

## PHASE 5: AUTOMATION DIRECTIVES

### Directive 5.1: Auto-Generate Type Definitions from Algorithms

For each algorithm (especially Tier 1), **automatically generate types** from documentation:

**Input:** Algorithm description in ARCHITECTURE.md
**Output:** Rust structs/TypeScript interfaces with proper fields

Example:
```markdown
### FM-Index on Wavelet Trees
- Stores: text array, wavelet trees, sample array
- Construction parameters: sample_rate (usually 32)
- Operations: rank(), select(), count()
```

Should auto-generate:
```rust
pub struct FMIndex {
    text: Vec<u8>,
    wavelet_trees: Vec<WaveletTree>,
    samples: Vec<(usize, usize)>,
    sample_rate: usize,
}

impl FMIndex {
    pub fn rank(&self, char: u8, pos: usize) -> usize { ... }
    pub fn select(&self, char: u8, rank: usize) -> usize { ... }
    pub fn count(&self, pattern: &[u8]) -> usize { ... }
}
```

### Directive 5.2: Auto-Generate Integration Tests

For **every dependency that integrates with Ryzanstein**, automatically generate:

1. **Mock test suite** — tests that run with mocked Ryzanstein components
2. **Integration test suite** — tests that run with real (or docker-composed) Ryzanstein services
3. **Upgrade path tests** — verify standalone mode → Ryzanstein-enhanced mode gracefully

**Template:**

```rust
// tests/integration_with_ryzanstein.rs

#[tokio::test]
async fn test_with_mock_ryot_llm() {
    let mock_ryot = MockInferenceEngine::new();
    let mock_sigma = MockSemanticEncoder::new();
    
    let dependency = DependencyName::new_with_mocks(mock_ryot, mock_sigma);
    
    let result = dependency.operation().await;
    assert!(result.is_ok());
}

#[tokio::test]
#[ignore]  // Run only with: cargo test -- --ignored
async fn test_with_real_ryzanstein_docker() {
    // This requires docker-compose.yml to be running
    let ryot = RyotClient::connect("http://localhost:8080").await?;
    let sigma = SigmaClient::connect("http://localhost:8081").await?;
    
    let dependency = DependencyName::new(ryot, sigma);
    
    let result = dependency.operation().await;
    assert!(result.is_ok());
}
```

### Directive 5.3: Auto-Generate CLI Tools

Every Tier 1 dependency gets a **CLI binary**:

```rust
// src/bin/sigma-index-cli.rs

use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new code index
    Init {
        #[arg(value_name = "PATH")]
        path: String,
    },
    /// Perform semantic search
    Search {
        query: String,
        
        #[arg(short)]
        limit: Option<usize>,
    },
    /// Export index statistics
    Stats,
}

fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Init { path } => { ... }
        Commands::Search { query, limit } => { ... }
        Commands::Stats => { ... }
    }
}
```

### Directive 5.4: Auto-Generate Documentation

Every dependency auto-generates:

1. **API Reference** — from docstrings/comments
2. **Algorithm Explanation** — from ARCHITECTURE.md
3. **Integration Guide** — from integration-types.ts + examples
4. **Deployment Guide** — from Dockerfile + docker-compose.yml
5. **Benchmarks Report** — from benchmark results

**Use tooling:**
- Rust: `cargo doc --open`
- TypeScript: `typedoc`
- Python: `pdoc` or `mkdocs`
- Go: `godoc`

**Store outputs in `docs/generated/`** — these are checked into git to avoid build-time dependencies.

### Directive 5.5: Auto-Generate GitHub Workflows

Template for release automation:

```yaml
# .github/workflows/release.yml

name: Release
on:
  push:
    tags: ['v*']

jobs:
  create-release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: softprops/action-gh-release@v1
        with:
          draft: false
          generate_release_notes: true

  publish-crates-io:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo publish --token ${{ secrets.CARGO_TOKEN }}

  publish-npm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm publish

  publish-pypi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pypa/gh-action-pypi-publish@release/v1
```

---

## PHASE 6: EXECUTION CHECKLIST

Execute scaffolding for **all 18 dependencies in this order** (respecting dependencies):

### Order (Critical for integration):

1. **Ryot LLM** (if not already complete) — all dependencies depend on this
2. **ΣLANG** (if not already complete) — Tier 1 & Tier 3 depend on this
3. **ΣVAULT** (if not already complete) — encryption features depend on this
4. **Tier 1 Ecosystem-Locked (6):**
   - `σ-index` → `mcp-mesh` → `σ-compress` → `σ-diff` → `vault-git` → `σ-telemetry`
5. **Tier 2 Standalone (6):**
   - `cpu-infer` (needed by downstream) → `causedb` → `intent.spec` → `flowstate` → `dep-bloom` → `archaeo`
6. **Tier 3 Hybrid (6):**
   - `σ-api` → `ann-hybrid` → `agentmem` → `semlog` → `zkaudit` → `neurectomy-shell`

### Per-Dependency Execution:

For **each of the 18 dependencies**, execute in sequence:

```
STEP 1: Validate integration points
└─ Confirm Ryzanstein components are reachable

STEP 2: Generate directory structure
└─ Create all directories, touch all files

STEP 3: Generate mandatory core files
├─ README.md
├─ ARCHITECTURE.md
├─ INTEGRATION.md
├─ License & CoC (Tier 1) or License Agreement (Tier 2/3)
└─ API definitions (lib.rs / api.ts / api.go)

STEP 4: Generate language-specific scaffolding
├─ Cargo.toml + Rust project structure (if applicable)
├─ package.json + TypeScript setup (if applicable)
├─ go.mod + Go module structure (if applicable)
└─ pyproject.toml + Python package structure (if applicable)

STEP 5: Generate integration hooks
├─ ryzanstein-integration.{rs,ts,go,py}
├─ Health checks
├─ MCP server registration (if applicable)
└─ Feature gates (Tier 3 only)

STEP 6: Generate test scaffolding
├─ Unit tests structure (tests/unit/)
├─ Integration tests (tests/integration/)
├─ Standalone vs. Ryzanstein tests (Tier 2/3)
└─ Benchmark tests (tests/benchmarks/)

STEP 7: Generate CI/CD
├─ .github/workflows/ci.yml
├─ .github/workflows/release.yml
└─ .github/workflows/integration-test.yml

STEP 8: Generate documentation
├─ Auto-generated API docs
├─ Algorithm explanations
├─ Integration examples
└─ Deployment guides

STEP 9: Generate CLI tools (Tier 1)
└─ src/bin/{dependency}-cli.rs

STEP 10: Generate monetization scaffolding (Tier 2/3)
├─ License key validator
├─ Usage telemetry
├─ Pricing tier definitions
└─ Feature gates

STEP 11: Verify completeness
├─ 90%+ test coverage
├─ All docstrings present
├─ All integration points defined
└─ Ready for immediate development
```

---

## PHASE 7: COPILOT SELF-DIRECTIVES

As Copilot executes this prompt, **follow these rules to maximize autonomy:**

### Rule 1: Generate First, Ask Second
- **Never ask for clarification on standard structure** (use the templates)
- **Do generate opinionated code** that follows Rust/Go/TypeScript/Python conventions
- Only ask the developer for:
  - Ryzanstein endpoint URLs/credentials (if not in environment)
  - Business logic specific to the organization
  - Pricing/licensing decisions not covered in this prompt

### Rule 2: Use Sensible Defaults
```
Rust:       Edition 2021, async=tokio, error=thiserror, serde=derived
Go:         1.22+, go-chi/chi for HTTP, gorm for databases
TypeScript: 5.4+, React/Svelte for UI, Vitest for testing
Python:     3.11+, pydantic for validation, pytest for testing
```

### Rule 3: Namespace Everything
- Rust crates: `sigma_index`, `sigma_diff`, `mcp_mesh` (snake_case)
- Go packages: `sigmaindex`, `sigmadiff`, `mcpmesh` (no underscores in Go)
- TypeScript: `SigmaIndex`, `SigmaDiff`, `MCPMesh` (PascalCase for classes/modules)
- Python: `sigma_index`, `sigma_diff`, `mcp_mesh` (snake_case)

### Rule 4: Mock Everything External
- Until integration tests explicitly require Ryzanstein, all code uses mocks
- Mock objects in `src/mocks.rs` (Rust) or `src/mocks/` (other languages)
- Mock implementations follow the same interfaces as real ones

### Rule 5: Test-Driven Scaffolding
- **Before writing implementation**, generate the test file
- Tests define the expected API surface
- Implementation fills in the blanks

### Rule 6: Document as You Go
- Every public function gets JSDoc/rustdoc/GoDoc
- Every module gets a README section
- Every algorithm gets a brief complexity analysis inline

### Rule 7: Prepare for Parallelization
- All 18 dependencies should be scaffolded completely **before** any implementation work begins
- This allows parallel development across multiple developers/agents
- Each dependency is self-contained with no hard runtime dependencies on others (only compile-time type imports)

---

## PHASE 8: SUCCESS CRITERIA

After executing this Master Class Prompt on **all 18 dependencies**, you should have:

### For Each Dependency:
- ✅ Complete project scaffolding (no empty directories)
- ✅ Public API defined (lib.rs / api.ts / api.go)
- ✅ Ryzanstein integration hooks present
- ✅ 100+ unit test cases (even if tests are minimal stubs)
- ✅ CI/CD workflows configured
- ✅ Documentation structure (README, ARCHITECTURE, INTEGRATION, API docs)
- ✅ CLI tools (Tier 1) or VS Code extension (Tier 2/3)
- ✅ Docker support for local integration testing
- ✅ License files and contribution guidelines

### Across All 18 Dependencies:
- ✅ Zero circular dependencies
- ✅ Consistent directory/naming structure
- ✅ All code compiles without warnings (with `cargo clippy`, `npm run lint`, `black --check`, `go vet`)
- ✅ 90%+ test coverage (across the suite)
- ✅ GitHub repositories ready for immediate development
- ✅ Monorepo (if desired) or per-project repos fully set up
- ✅ CI/CD pipelines green on first commit

### Ready for Parallel Development:
- ✅ 18 developers (or 18 Copilot sessions) can work on these in parallel
- ✅ No blocking dependencies between scaffolded projects
- ✅ Integration tests use mocks, allowing development without Ryzanstein being live
- ✅ Clear phase gates (Phase 0 integration patterns established before moving to Phase 1 implementation)

---

## FINAL DIRECTIVE: BEGIN SCAFFOLDING

Execute this Master Class Prompt to scaffold **all 18 dependencies** for the Ryzanstein Unified Architecture.

**Command to Copilot:**
```
Generate complete, production-ready project scaffolding for all 18 dependencies 
of the Ryzanstein Unified Architecture (Tier 1, Tier 2, Tier 3).

Follow all directives in PHASE 1–8.
Use sensible defaults from PHASE 7.
Verify against success criteria in PHASE 8.

Work autonomously. Generate opinionated, conventional code.
Do not ask for clarification on structure or naming.
Only ask for: Ryzanstein URLs, business logic, pricing decisions.

When complete, provide:
1. Summary of all 18 scaffolded dependencies
2. GitHub repository links
3. Outstanding integration work (if any)
4. Next phase recommendations for parallel development
```

**Timeline:** This scaffolding should complete in **2–4 hours of Copilot runtime** for all 18 dependencies, allowing **immediate parallel development** by Sprint 5.

---

## APPENDIX A: Reference Commands

### For Developers Using These Scaffolds:

```bash
# Clone all scaffolded repos
for repo in sigma-index sigma-diff mcp-mesh sigma-compress vault-git \
            sigma-telemetry causedb intent-spec flowstate dep-bloom \
            archaeo cpu-infer sigma-api zkaudit agentmem semlog \
            ann-hybrid neurectomy-shell; do
  git clone git@github.com:ryzanstein/$repo.git
done

# Build all Rust projects
find . -name "Cargo.toml" -exec cargo build --release \;

# Run all tests
find . -name "Cargo.toml" -exec cargo test --all \;
find . -name "package.json" -exec npm test \;
find . -name "pytest.ini" -exec pytest \;

# Generate all docs
find . -name "Cargo.toml" -exec cargo doc --no-deps \;
find . -name "package.json" -exec npm run docs \;
```

### For CI/CD Integration:

```yaml
# Global CI workflow (in main Ryzanstein repo)
name: Integration Test All Dependencies
on: [push, pull_request]

jobs:
  test-all-dependencies:
    strategy:
      matrix:
        dependency: [sigma-index, sigma-diff, mcp-mesh, ...]
    steps:
      - uses: actions/checkout@v4
        with:
          repository: ryzanstein/${{ matrix.dependency }}
      - run: cargo test --all (if Rust)
      - run: npm test (if TypeScript)
      - run: pytest (if Python)
```

---

**END OF MASTER CLASS PROMPT**

*This prompt is designed to execute autonomously. Copilot should generate all 18 dependency scaffolds without human intervention beyond initiating the command. Use it as the single source of truth for the entire scaffolding phase.*
