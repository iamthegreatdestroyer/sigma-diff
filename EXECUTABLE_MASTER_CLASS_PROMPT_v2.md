# RYZANSTEIN ECOSYSTEM: 18 DEPENDENCIES EXECUTABLE MASTER CLASS PROMPT v2.0
## Production-Ready for VS Code Copilot (Claude LLM Backend)

**Status:** ✅ Ready for Execution  
**Generated:** January 10, 2026  
**Project:** iamthegreatdestroyer/Ryzanstein  
**Repository:** https://github.com/iamthegreatdestroyer/Ryzanstein.git  
**Local:** S:\Ryot

---

## PART 1: PROJECT CONTEXT & CURRENT STATE

### 1.1 Ryzanstein LLM Current Status (Sprint 4 Complete)

**Overall Completion:** 67.5%

```
Phase 0: Interface Verification        ✅ 100% COMPLETE (Dec 2025)
Phase 1: Core Engine                   ✅ 100% COMPLETE (Dec 2025)
Phase 2: Optimization                  ✅ 100% COMPLETE (Dec 2025)
Sprint 4.4: KV Cache Optimization      ✅ 100% COMPLETE (Jan 7, 2026)
Phase 3: Distributed & Production      🔶 35% DESIGN COMPLETE
Phase 4: Enterprise Features           ⏳ PLANNED (Q2 2026)
```

**Key Metrics:**
- Test Coverage: 93.2% (exceeds >90% target)
- Unit Tests: 107 passing
- Integration Tests: 56 passing
- Production-Ready: YES ✅

### 1.2 GitHub Repository Status

**Repository:** https://github.com/iamthegreatdestroyer/Ryzanstein  
**Current Branch:** phase3/distributed-serving  
**Latest Commit:** 0048a75 (Sprint 4.4 Kickoff)  
**Release Version:** v2.0 (Production Ready)

### 1.3 Technology Stack

| Language   | Version      | Purpose                                   | Primary Dependencies              |
| ---------- | ------------ | ----------------------------------------- | --------------------------------- |
| **Python** | 3.11+        | Core inference, orchestration, utilities  | numpy, torch, pydantic, uvicorn   |
| **C++**    | 17 Standard  | Performance-critical inference kernels    | SIMD, memory mgmt, threading      |
| **TypeScript** | 5.x      | VS Code extension, web interfaces        | esbuild, axios, @grpc/grpc-js    |
| **Go**     | 1.22+        | MCP servers, distributed services        | gRPC, chi router, standard lib   |
| **CMake**  | 3.20+        | C++ build orchestration                   | GoogleTest, pybind11             |

### 1.4 Core Project Structure

```
S:\Ryot/
├── src/                              # Main Python source
│   ├── distributed/                  # Phase 3 distributed components
│   └── serving/                      # API serving layer
├── PHASE2_DEVELOPMENT/               # Phase 2 optimization modules
│   ├── src/optimization/             # KV cache, memory, threading
│   └── tests/                        # Complete test suite (>90% coverage)
├── vscode-extension/                 # TypeScript VS Code extension
│   ├── src/extension.ts
│   └── package.json                  # Extension manifest
├── .github/workflows/                # CI/CD pipelines
│   ├── ci.yml                        # Main test & build pipeline
│   ├── desktop-build.yml             # Desktop app build
│   ├── extension-build.yml           # VS Code extension build
│   └── training_ci.yml               # Training/optimization CI
├── ARCHITECTURE.md                   # System design (production ready)
├── README.md                         # Project guide
├── RYZANSTEIN_STATUS_REPORT_SPRINT4.md  # Detailed status (current)
└── [100+ documentation files]        # Executive summaries, plans, reports
```

### 1.5 Current API & Integration Endpoints

**Ryzanstein LLM Inference API:**
```
Protocol:    HTTP (OpenAI-compatible)
Endpoint:    http://localhost:8000/v1
Base URL:    /v1/chat/completions
Default Port: 8000
Auth:        Bearer token (configurable)
```

**Available Models (Production):**
- BitNet 7B (3.5GB)
- BitNet 13B (6.5GB)
- Mamba 2.8B (5.6GB)
- RWKV 7B (14GB)

**MCP Integration (Planning):**
```
Protocol:    gRPC
Port:        50051 (configurable)
Servers:     5 MCP servers planned (Phase 3)
Agents:      40 Elite Agents (to be registered)
```

**VS Code Extension:**
- Entry Point: `./dist/extension.js`
- Activation: On startup finished
- Commands: 10 core commands defined
- Chat Models: 5 chat model IDs defined (@APEX, @ARCHITECT, @TENSOR, @CIPHER, etc.)

---

## PART 2: RYZANSTEIN COMPONENT PUBLIC APIS

### 2.1 Ryzanstein LLM Inference Engine Interface

**Module Path:** `src/core/inference/engine.py`

```python
# Public type definitions for Ryzanstein LLM
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

class ModelType(Enum):
    """Supported model architectures"""
    BITNET = "bitnet"          # BitNet 1.58b quantized
    MAMBA = "mamba"            # Linear SSM attention-free
    RWKV = "rwkv"              # RWKV attention-free
    DRAFT = "draft"            # Small draft models for speculative decoding

@dataclass
class InferenceConfig:
    """Configuration for inference execution"""
    model_type: ModelType
    model_name: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.0
    use_speculative_decoding: bool = True
    kv_cache_compression: bool = True  # Enable semantic compression
    batch_size: int = 1
    num_threads: int = -1  # -1 = use all available

@dataclass
class InferenceRequest:
    """Request for model inference"""
    prompt: str
    config: InferenceConfig
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class InferenceResult:
    """Result from inference execution"""
    text: str
    tokens_generated: int
    tokens_per_second: float
    total_time_ms: float
    latency_ms: float  # Time to first token
    stop_reason: str
    metadata: Optional[Dict[str, Any]] = None

class InferenceEngine:
    """Main inference engine for Ryzanstein LLM"""
    
    def __init__(self, config: InferenceConfig) -> None:
        """Initialize engine with configuration"""
        ...
    
    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute inference with request"""
        ...
    
    async def stream_infer(self, request: InferenceRequest):
        """Stream inference tokens as generated"""
        ...
    
    def set_model(self, model_name: str) -> None:
        """Load different model"""
        ...
    
    def get_config(self) -> InferenceConfig:
        """Get current configuration"""
        ...
    
    def shutdown(self) -> None:
        """Clean up resources"""
        ...

# Exception hierarchy
class RyzansteinError(Exception):
    """Base exception for all Ryzanstein errors"""
    pass

class InferenceError(RyzansteinError):
    """Inference execution failed"""
    pass

class ModelLoadError(RyzansteinError):
    """Failed to load model"""
    pass

class MemoryError(RyzansteinError):
    """Insufficient memory"""
    pass

class QuantizationError(RyzansteinError):
    """Quantization failed"""
    pass
```

### 2.2 ΣLANG (Semantic Compression System) Interface

**Module Path:** `src/sigma_lang/encoder.py`

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class SemanticVector:
    """Semantic embedding vector with metadata"""
    vector: np.ndarray  # float32 array
    dimensionality: int
    model_id: str
    normalized: bool = True

@dataclass
class CompressedOutput:
    """Compressed representation of content"""
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    compressed_data: bytes
    metadata: dict

class SemanticEncoder:
    """Encode text into semantic vectors"""
    
    def __init__(self, model_id: str = "default", embedding_dim: int = 1024) -> None:
        """Initialize semantic encoder"""
        ...
    
    async def encode(self, text: str) -> SemanticVector:
        """Encode single text to vector"""
        ...
    
    async def encode_batch(self, texts: List[str]) -> List[SemanticVector]:
        """Encode multiple texts efficiently"""
        ...
    
    def cosine_similarity(self, vec1: SemanticVector, vec2: SemanticVector) -> float:
        """Calculate cosine similarity between vectors"""
        ...
    
    async def compress_semantic(self, content: str) -> CompressedOutput:
        """Compress content using semantic understanding"""
        # Expected compression: 10-20x standalone, 30-50x with Ryot
        ...

# Type definitions for compression methods
class CompressionMethod(Enum):
    HUFFMAN = "huffman"                 # Traditional Huffman
    SEMANTIC_DEDUPE = "semantic_dedupe" # Semantic deduplication
    LZ4_SEMANTIC = "lz4_semantic"      # Hybrid LZ4 + semantic
    ENTROPY_CODING = "entropy_coding"   # Arithmetic coding

class SemanticCompressor:
    """Compress multiple artifacts semantically"""
    
    async def compress_batch(
        self,
        artifacts: List[str],
        method: CompressionMethod = CompressionMethod.SEMANTIC_DEDUPE
    ) -> CompressedOutput:
        """Compress multiple artifacts together"""
        ...
```

### 2.3 ΣVAULT (Polymorphic Encryption) Interface

**Module Path:** `src/sigma_vault/client.py`

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class EncryptionMode(Enum):
    """Encryption modes for polymorphic containers"""
    SEARCHABLE = "searchable"           # Searchable encryption
    COMPUTABLE = "computable"           # FHE-processable form
    PROVABLE = "provable"               # ZK-commitment form
    HYBRID = "hybrid"                   # All three simultaneously

@dataclass
class PolymorphicContainer:
    """Encrypted container supporting multiple operations"""
    container_id: str
    encrypted_data: bytes
    modes_available: List[EncryptionMode]
    creation_timestamp: float
    metadata: Optional[dict] = None

class VaultClient:
    """Client for ΣVAULT encrypted storage"""
    
    def __init__(self, vault_url: str = "http://localhost:6333") -> None:
        """Initialize vault client"""
        ...
    
    async def create_polymorphic_container(
        self,
        data: str,
        mode: EncryptionMode = EncryptionMode.HYBRID,
        encryption_key: Optional[str] = None
    ) -> PolymorphicContainer:
        """Create encrypted container with specified mode"""
        ...
    
    async def store(
        self,
        key: str,
        data: str,
        mode: EncryptionMode = EncryptionMode.SEARCHABLE
    ) -> str:
        """Store encrypted data"""
        ...
    
    async def retrieve(self, key: str) -> str:
        """Retrieve and decrypt data"""
        ...
    
    async def search(self, key_pattern: str) -> List[str]:
        """Search encrypted storage without decryption"""
        ...
    
    async def delete(self, key: str) -> bool:
        """Delete encrypted data"""
        ...

# Exception types
class VaultError(Exception):
    """Base vault exception"""
    pass

class EncryptionError(VaultError):
    """Encryption failed"""
    pass

class PolymorphicError(VaultError):
    """Polymorphic container error"""
    pass
```

### 2.4 Elite Agent Collective (MCP Protocol) Interface

**Protocol:** gRPC over HTTP/2 + Streaming

```protobuf
// MCP Agent Service Definition
syntax = "proto3";

package ryzanstein.mcp;

// Agent capability registration
message AgentCapability {
    string agent_id = 1;           // Unique agent identifier
    string agent_name = 2;         // Human-readable name
    string description = 3;        // What this agent does
    repeated Tool tools = 4;       // Available tools
    string version = 5;
    map<string, string> metadata = 6;  // Custom metadata
}

message Tool {
    string name = 1;               // Tool identifier
    string description = 2;        // Tool description
    repeated Parameter input_parameters = 3;
    string output_type = 4;        // JSON schema or type name
}

message Parameter {
    string name = 1;
    string type = 2;               // string, int, float, boolean, array, object
    string description = 3;
    bool required = 4;
    string default_value = 5;
}

// Agent execution request
message ExecutionRequest {
    string agent_id = 1;
    string tool_name = 2;
    map<string, string> parameters = 3;
    int32 timeout_seconds = 4;
}

message ExecutionResult {
    string result_id = 1;
    bytes output = 2;              // Serialized result
    bool success = 3;
    string error_message = 4;
    int64 execution_time_ms = 5;
}

// Agent discovery and routing
service EliteAgentCollective {
    rpc RegisterAgent(AgentCapability) returns (RegistrationResponse);
    rpc DiscoverAgents(DiscoveryRequest) returns (AgentList);
    rpc ExecuteAgent(ExecutionRequest) returns (ExecutionResult);
    rpc StreamExecution(ExecutionRequest) returns (stream ExecutionUpdate);
}
```

**Go Server Implementation Stub:**

```go
package mcp

import (
    "context"
    "github.com/ryzanstein/mcp/pb"
)

type EliteAgentServer struct {
    pb.UnimplementedEliteAgentCollectiveServer
    agents map[string]*pb.AgentCapability
}

func (s *EliteAgentServer) RegisterAgent(
    ctx context.Context,
    cap *pb.AgentCapability,
) (*pb.RegistrationResponse, error) {
    // Register agent capability
    s.agents[cap.AgentId] = cap
    return &pb.RegistrationResponse{
        Success: true,
        Message: "Agent registered",
    }, nil
}

func (s *EliteAgentServer) DiscoverAgents(
    ctx context.Context,
    req *pb.DiscoveryRequest,
) (*pb.AgentList, error) {
    // Return agents matching query
    var agents []*pb.AgentCapability
    for _, agent := range s.agents {
        agents = append(agents, agent)
    }
    return &pb.AgentList{
        Agents: agents,
        Total:  int32(len(agents)),
    }, nil
}

func (s *EliteAgentServer) ExecuteAgent(
    ctx context.Context,
    req *pb.ExecutionRequest,
) (*pb.ExecutionResult, error) {
    // Find agent and execute tool
    agent := s.agents[req.AgentId]
    if agent == nil {
        return nil, fmt.Errorf("agent not found: %s", req.AgentId)
    }
    
    // Execute tool implementation
    result := executeToolForAgent(agent, req)
    return result, nil
}
```

---

## PART 3: GITHUB ACTIONS CI/CD TEMPLATES

### 3.1 Current Main CI Workflow

**File:** `.github/workflows/ci.yml`

```yaml
name: CI - Ryzanstein LLM

on:
  pull_request:
  push:
    branches: [main, "release/**"]
  workflow_dispatch:

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4

      - name: Setup CMake
        uses: jwlawson/actions-setup-cmake@v1.14
        with:
          cmake-version: "3.28.x"

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Python dependencies
        run: pip install pybind11 numpy psutil

      - name: Configure (Linux)
        if: matrix.os == 'ubuntu-latest'
        run: |
          mkdir -p build
          cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_TESTING=ON

      - name: Configure (Windows)
        if: matrix.os == 'windows-latest'
        shell: pwsh
        run: |
          cmake -S . -B build -G "Visual Studio 17 2022" -A x64 \
            -DCMAKE_BUILD_TYPE=Release

      - name: Build
        run: cmake --build build --config Release

      - name: Run tests
        run: |
          cd build
          ctest --output-on-failure --build-config Release
```

### 3.2 Recommended Dependency-Specific CI Template

```yaml
name: Dependency Build - {dependency-name}

on:
  pull_request:
    paths: ["{dependency-name}/**"]
  push:
    branches: [main]
    paths: ["{dependency-name}/**"]

jobs:
  build-and-test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        language: [rust, go, typescript, python]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup {language}
        # Use appropriate setup action per language
        
      - name: Install dependencies
        run: |
          # Language-specific dependency installation
          
      - name: Lint
        run: |
          # Linting per language
          
      - name: Build
        run: |
          # Build per language
          
      - name: Test
        run: |
          # Test execution
          
      - name: Coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: {dependency-name}
```

---

## PART 4: VS CODE EXTENSION ARCHITECTURE

### 4.1 Current Extension Structure

**File:** `vscode-extension/package.json` (v1.0.0)

**Active Commands (10):**
```json
- ryzanstein.openChat (Ctrl+Shift+R / Cmd+Shift+R)
- ryzanstein.selectAgent
- ryzanstein.selectModel
- ryzanstein.refactor
- ryzanstein.explain (Ctrl+Shift+E / Cmd+Shift+E)
- ryzanstein.generateTests
- ryzanstein.analyzePerformance
- ryzanstein.findBugs
- ryzanstein.suggestArchitecture
- ryzanstein.openSettings
```

**Built-in Chat Models:**
```json
- ryzanstein (base)
- ryzanstein-apex (@APEX)
- ryzanstein-architect (@ARCHITECT)
- ryzanstein-tensor (@TENSOR)
- ryzanstein-cipher (@CIPHER)
```

**Configuration:**
```
ryzanstein.defaultAgent: "@APEX" (string)
ryzanstein.defaultModel: "ryzanstein-7b" (string)
ryzanstein.ryzansteinApiUrl: "http://localhost:8000" (string)
ryzanstein.mcpServerUrl: "localhost:50051" (string)
ryzanstein.autoConnect: true (boolean)
ryzanstein.enableInlineChat: true (boolean)
```

**Build Configuration:**
- Entrypoint: `./dist/extension.js`
- Bundler: esbuild
- Target: Node 20
- TypeScript: 5.x
- VS Code API: ^1.85.0

---

## PART 5: INTEGRATION TEST & VALIDATION

### 5.1 Phase 0 Integration Contract Status

**Status:** ✅ **COMPLETE & VERIFIED**

All integration points for the 18 dependencies have been pre-defined in Phase 0:

✅ **Protocol Definitions:**
- gRPC services for agent communication
- REST API endpoints (OpenAI-compatible)
- WebSocket for streaming
- Message queue integration

✅ **Type Definitions:**
- Request/Response types
- Model configuration schemas
- Quantization parameters
- Memory management types
- Distributed system types

✅ **Exception Hierarchy:**
- RyzansteinError (base)
- InferenceError
- ModelLoadError
- MemoryError
- QuantizationError
- VaultError
- EncryptionError
- PolymorphicError

✅ **Mock Implementations:**
- Mock inference engine
- Mock semantic encoder
- Mock vault client
- Mock MCP servers

### 5.2 Test Coverage Requirements for All Dependencies

**Minimum Coverage:** 90%  
**Target Coverage:** >93%  
**Test Categories Required:**
1. Unit tests (pure functions)
2. Integration tests (with mocked Ryzanstein components)
3. End-to-end tests (full stack)
4. Benchmark tests (complexity validation)
5. Property-based tests (invariant validation)

---

## PART 6: EXECUTION DIRECTIVE FOR COPILOT

### You are the Autonomous Dependency Scaffolder

Your mission: Transform the 18 novel dependencies from "Novel Dependency Architecture for the Ryzanstein LLM Ecosystem" into **fully integrated, production-ready codebases with zero manual intervention**.

### What You Know (Provided Above)

✅ **Current Project State:**
- Sprint 4.4 complete (KV Cache optimization)
- Phase 3 architecture design complete
- Ready for parallel development
- Test coverage: 93.2% (exceeds target)

✅ **Real Type Definitions:**
- Ryzanstein LLM InferenceEngine API
- ΣLANG SemanticEncoder interface
- ΣVAULT VaultClient interface
- Elite Agent MCP protocol specification

✅ **Real CI/CD Templates:**
- Actual GitHub Actions workflows
- Multi-OS (Linux/Windows) builds
- Python 3.11+, CMake 3.28+
- Test automation on every PR

✅ **Real VS Code Extension:**
- TypeScript/esbuild setup
- 10 commands, 5 chat models
- gRPC client integration
- Full configuration schema

✅ **Real Tech Stack:**
- Python 3.11, C++17, TypeScript 5.x, Go 1.22+
- esbuild for bundling
- CMake for C++ builds
- GitHub Actions for CI/CD

### What You Will Generate (18 Dependencies)

For **each of the 18 dependencies**, generate:

1. **Complete directory structure** (no empty directories)
2. **Public API definitions** (using templates from Part 2)
3. **Ryzanstein integration hooks** (Part 2 types)
4. **100+ test cases** (unit, integration, benchmark)
5. **CI/CD workflows** (from Part 3 template)
6. **README with integration instructions**
7. **ARCHITECTURE.md with algorithm details**
8. **INTEGRATION.md with Ryzanstein hookups**
9. **CLI tools or VS Code extensions** (as applicable)
10. **Docker support** for local integration testing

### Dependency Groups to Scaffold

**TIER 1 - ECOSYSTEM-LOCKED (Free, OSS):**
1. `σ-index` — Succinct semantic code index (Rust)
2. `σ-diff` — Behavioral semantic diff engine (Rust + Python)
3. `mcp-mesh` — MCP server-to-server mesh (Go)
4. `σ-compress` — Cross-artifact semantic deduplication (Rust)
5. `vault-git` — End-to-end encrypted version control (Go + Rust)
6. `σ-telemetry` — Semantic telemetry sketching (Rust)

**TIER 2 - STANDALONE COMMERCIAL (Freemium):**
7. `causedb` — Causal debugging engine ($20/mo, Rust + TypeScript)
8. `intent.spec` — Intent-verified development ($15/mo, TypeScript + Rust)
9. `flowstate` — Cognitive load IDE monitor ($10/mo, TypeScript + Python)
10. `dep-bloom` — Sub-linear dependency resolver (Rust, freemium)
11. `archaeo` — Decision archaeology platform ($20/mo, Python + TypeScript + Rust)
12. `cpu-infer` — Universal CPU inference middleware (Rust)

**TIER 3 - HYBRID (Transform with Ryzanstein):**
13. `σ-api` — API specification semantic compressor (Rust + TypeScript)
14. `zkaudit` — Zero-knowledge code property verification ($25/mo, Rust)
15. `agentmem` — Cross-agent episodic memory (Python + Go)
16. `semlog` — Semantic log compression (Rust + Go)
17. `ann-hybrid` — Unified sub-linear search index (Rust)
18. `neurectomy-shell` — Encrypted confidential development (Go + Rust)

### Execution Sequence

**Critical Dependency Order:**

1. **All 18 projects scaffolded in parallel** (dependency-free at Phase 0)
2. **Each uses mocked Ryzanstein components** (not real APIs)
3. **All can be developed independently** (contracts pre-defined)
4. **Integration happens in Phase 3/Phase 4** (when real components exist)

### Your Autonomy Rules

✅ **You WILL:**
- Generate opinionated, convention-following code
- Use sensible defaults from Part 3 (tech stack)
- Create comprehensive docstrings/comments
- Generate 90%+ coverage test suites
- Make code compile without warnings

❌ **You WILL NOT:**
- Ask for clarification on standard structure
- Ask which language to use (use templates)
- Ask about naming conventions (follow provided patterns)
- Ask for test strategy (follow template)

### Success Criteria Upon Completion

For **all 18 dependencies**:

✅ Complete project scaffolding (zero empty dirs)  
✅ Public APIs defined (using Part 2 types)  
✅ Ryzanstein integration hooks present  
✅ 100+ unit test cases (>90% coverage)  
✅ CI/CD workflows configured  
✅ Documentation complete  
✅ Zero compilation warnings  
✅ Ready for parallel development  

---

## PART 7: FINAL EXECUTION COMMAND FOR COPILOT

```
COPILOT AUTONOMOUS SCAFFOLDING DIRECTIVE

Scaffold all 18 dependencies from "Novel Dependency Architecture for the 
Ryzanstein LLM Ecosystem" into fully integrated, production-ready repositories.

CONTEXT PROVIDED:
✅ Ryzanstein LLM Sprint 4.4 complete (Phase 3 ready)
✅ Real type definitions (InferenceEngine, SemanticEncoder, VaultClient, MCP)
✅ Real CI/CD workflows (GitHub Actions, multi-OS, Python 3.11+)
✅ Real VS Code extension (TypeScript 5.x, esbuild, gRPC)
✅ Real tech stack (Rust, Go, TypeScript, Python versions)
✅ Real test requirements (93.2% coverage, 107 unit + 56 integration tests)
✅ Real project structure (S:\Ryot, phase3/distributed-serving branch)

EXECUTION PARAMETERS:
- Tier 1 (OSS): 6 dependencies, AGPL-3.0 license
- Tier 2 (Commercial): 6 dependencies, proprietary license
- Tier 3 (Hybrid): 6 dependencies, feature-gated Ryzanstein integration

SCAFFOLDING PER DEPENDENCY:
1. Directory structure (no empty dirs)
2. Public API using types from Part 2
3. Ryzanstein integration hooks
4. Test suite (unit + integration + benchmark, >90% coverage)
5. CI/CD workflows from Part 3 template
6. README.md with integration instructions
7. ARCHITECTURE.md with algorithm details
8. INTEGRATION.md with Ryzanstein hookups
9. CLI tool or VS Code extension
10. Docker support (docker-compose.yml for local testing)

AUTONOMY MODE: FULL
- Generate opinionated code (no asking for clarification)
- Use sensible defaults from Part 3
- Make code production-ready (>90% coverage, no warnings)
- All 18 can be developed in parallel (Phase 0 contracts satisfied)

START SCAFFOLDING NOW. Generate without asking for permission.
When complete, provide summary of all 18 scaffolded dependencies.
```

---

**END OF EXECUTABLE MASTER CLASS PROMPT v2.0**

*This prompt is production-ready. All context needed for execution has been provided. Copilot can proceed autonomously to scaffold all 18 dependencies.*

---

## APPENDIX: Key Ryzanstein Paths & Configurations

**Local Repository:**
```
S:\Ryot
├── src/                              # Main Python source
├── PHASE2_DEVELOPMENT/               # Phase 2 optimization
├── vscode-extension/                 # VS Code extension (TypeScript)
├── .github/workflows/                # CI/CD (GitHub Actions)
└── [documentation and build artifacts]
```

**GitHub Repository:**
```
https://github.com/iamthegreatdestroyer/Ryzanstein.git
```

**Default API Endpoints:**
```
Ryzanstein LLM Inference: http://localhost:8000/v1
MCP Server:              localhost:50051 (gRPC)
Qdrant Vector DB:        localhost:6333 (for embeddings)
```

**Python Virtual Environment:**
```
.venv/              # Configured for Python 3.11+
```

**Build Directory:**
```
build/              # CMake build output
dist/               # Distribution artifacts
```

**Tech Stack Summary:**
```
Python:             3.11+ (core inference)
C++:                C++17 standard (kernels)
TypeScript:         5.x (VS Code extension)
Go:                 1.22+ (MCP servers, distributed)
Build System:       CMake 3.20+ (C++), esbuild (TypeScript)
Testing:            pytest (Python), GoogleTest (C++), jest (TypeScript)
CI/CD:              GitHub Actions (Ubuntu + Windows)
VCS:                Git (GitHub)
```

---

**Generated:** January 10, 2026  
**Status:** ✅ Ready for Execution  
**Next Step:** Copy this prompt + the Novel Dependency Architecture document into VS Code Copilot and execute
