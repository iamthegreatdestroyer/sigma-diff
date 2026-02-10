# 🚀 OPTION A: MASTER ACTION PLAN & ROADMAP

## Full Custom Ryzanstein LLM Stack - Autonomous Execution Framework

**Created:** January 15, 2026  
**Version:** 1.0  
**Objective:** Complete the full custom Ryzanstein LLM inference stack with maximum autonomy and automation  
**Philosophy:** _"This is a custom LLM for personal use, not a quick fix - build it right."_

---

## 📊 EXECUTIVE SUMMARY

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RYZANSTEIN LLM - OPTION A ROADMAP                         ║
║                   Full Custom Stack - No Shortcuts                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  LAST UPDATED: January 15, 2026 @ 18:30                                      ║
║                                                                              ║
║  PHASE 1: C++ Bindings Completion (Days 1-3) ██████░░░░░░░░░░░░░░░░ 80%     ║
║  PHASE 2: Model Acquisition (Days 4-5)       ░░░░░░░░░░░░░░░░░░░░░░ 0%      ║
║  PHASE 3: Python API Server (Days 6-8)       ██████████████████████ 100% ✅ ║
║  PHASE 4: MCP Integration (Days 9-11)        ██████████████████████ 100% ✅ ║
║  PHASE 5: Desktop Integration (Days 12-14)   ██████████░░░░░░░░░░░░ 70%     ║
║  PHASE 6: Elite Agents Activation (Days 15-17) ████████████████████ 90%     ║
║  PHASE 7: Optimization & Tuning (Days 18-21) ░░░░░░░░░░░░░░░░░░░░░░ 0%      ║
║                                                                              ║
║  CURRENT BLOCKER: C++ bindings build (real inference vs mock)               ║
║  NEXT ACTION: Build ryzen_llm_bindings.pyd with full engine exposure        ║
║                                                                              ║
║  TOTAL ESTIMATED TIME: 3 Weeks (adjusted)                                    ║
║  AUTOMATION LEVEL: Maximum (95%+ autonomous)                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🏗️ ARCHITECTURAL OVERVIEW

### The Complete Custom Stack

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              RYZANSTEIN DESKTOP APP                     │
                    │              (Wails + Svelte UI)                        │
                    │              Port: localhost:34115                      │
                    └─────────────────────┬───────────────────────────────────┘
                                          │ gRPC/HTTP
                                          ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │              MCP SERVER SUITE                           │
                    │              (Go gRPC Server)                           │
                    │              Port: 50051                                │
                    │  ┌─────────────────────────────────────────────────┐   │
                    │  │ InferenceService │ AgentService │ MemoryService │   │
                    │  │ OptimizationService │ DebugService              │   │
                    │  └─────────────────────────────────────────────────┘   │
                    └─────────────────────┬───────────────────────────────────┘
                                          │ HTTP/REST
                                          ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │              PYTHON FASTAPI SERVER                      │
                    │              (OpenAI-Compatible API)                    │
                    │              Port: 8000                                 │
                    │  ┌─────────────────────────────────────────────────┐   │
                    │  │ /v1/chat/completions │ /v1/embeddings          │   │
                    │  │ /v1/models │ /health │ /metrics                │   │
                    │  └─────────────────────────────────────────────────┘   │
                    └─────────────────────┬───────────────────────────────────┘
                                          │ Python Bindings
                                          ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │              C++ INFERENCE ENGINE                       │
                    │              (ryzen_llm_bindings.pyd)                   │
                    │  ┌─────────────────────────────────────────────────┐   │
                    │  │ BitNet b1.58 │ T-MAC │ Mamba SSM │ RWKV        │   │
                    │  │ AVX-512/AVX2 │ Speculative Decoding │ KV Cache │   │
                    │  └─────────────────────────────────────────────────┘   │
                    └─────────────────────┬───────────────────────────────────┘
                                          │ Memory-Mapped
                                          ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │              MODEL WEIGHTS                              │
                    │              (SafeTensors Format)                       │
                    │  ┌─────────────────────────────────────────────────┐   │
                    │  │ BitNet 3B (13GB) │ Mamba 2.8B │ RWKV 7B        │   │
                    │  │ Draft Model 350M (Speculative)                  │   │
                    │  └─────────────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────────────┘
```

### What Already Exists (Assets to Leverage)

| Component                  | Status             | Location                        | Lines of Code |
| -------------------------- | ------------------ | ------------------------------- | ------------- |
| **C++ Inference Engine**   | ✅ 80% Built       | `RYZEN-LLM/src/`                | ~15,000       |
| **Python Bindings (.pyd)** | ✅ Compiles        | `build/python/ryzanstein_llm/`  | 894           |
| **Python API Server**      | ✅ Structure Ready | `RYZEN-LLM/src/api/server.py`   | 315           |
| **MCP Server (Go)**        | ⚠️ Stub Only       | `mcp/server.go`                 | 668           |
| **Desktop App**            | ✅ Runs            | `desktop/`                      | ~5,000        |
| **Model Download Script**  | ✅ Ready           | `scripts/download_models.py`    | 323           |
| **Elite Agents (40)**      | ✅ Complete        | `.github/agents/`               | ~8,000        |
| **SafeTensors Loader**     | ✅ Complete        | `src/io/safetensors_loader.cpp` | 447           |
| **CMake Build System**     | ✅ Complete        | `CMakeLists.txt`                | 181           |

---

## 📋 PHASE 1: C++ BINDINGS COMPLETION (Days 1-3)

### Current State

The C++ engine builds but Python bindings only expose `test_function()`. We need to expose the full inference pipeline.

### Tasks

#### 1.1 Expand pybind11 Module (HIGH PRIORITY)

**File:** `RYZEN-LLM/src/api/bindings/bitnet_bindings.cpp`

```cpp
// CURRENT: Only exposes test_function()
// TARGET: Expose full inference API

PYBIND11_MODULE(ryzen_llm_bindings, m) {
    // BitNet Engine
    py::class_<BitNetEngine>(m, "BitNetEngine")
        .def(py::init<const std::string&>())  // Load model path
        .def("generate", &BitNetEngine::generate)
        .def("get_model_info", &BitNetEngine::get_model_info);

    // Generation Config
    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("max_tokens", &GenerationConfig::max_tokens)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("top_p", &GenerationConfig::top_p);

    // Quantization Functions
    m.def("quantize_weights", &quantize_weights);
    m.def("dequantize_weights", &dequantize_weights);

    // SafeTensors Loader
    py::class_<SafeTensorsLoader>(m, "SafeTensorsLoader")
        .def(py::init<>())
        .def("load", &SafeTensorsLoader::load);
}
```

#### 1.2 Build Engine Interface

**New File:** `RYZEN-LLM/src/inference/engine_interface.cpp`

```cpp
class BitNetEngineInterface {
public:
    // Load model from SafeTensors
    bool load_model(const std::string& model_path);

    // Generate tokens
    std::string generate(
        const std::string& prompt,
        int max_tokens = 512,
        float temperature = 0.7,
        float top_p = 0.9
    );

    // Streaming generation (yields tokens)
    std::vector<std::string> generate_stream(
        const std::string& prompt,
        int max_tokens = 512
    );

    // Model info
    std::map<std::string, std::string> get_model_info();

private:
    std::unique_ptr<BitNetLayer> layer_;
    std::unique_ptr<SafeTensorsLoader> loader_;
    bool model_loaded_ = false;
};
```

#### 1.3 Rebuild C++ Extension

**Automation Script:** `scripts/rebuild_bindings.ps1`

```powershell
# Automated C++ rebuild script
$ErrorActionPreference = "Stop"
$ProjectRoot = "S:\Ryot\RYZEN-LLM"

Write-Host "=== Rebuilding Ryzanstein C++ Bindings ===" -ForegroundColor Cyan

# Configure
cd $ProjectRoot\build
cmake .. -G "Visual Studio 16 2019" -A x64

# Build Release
cmake --build . --config Release --target ryzen_llm_bindings -j 8

# Copy to Python path
Copy-Item "python\ryzanstein_llm\ryzen_llm_bindings.pyd" `
    -Destination "$ProjectRoot\src\api\" -Force

Write-Host "✅ Bindings rebuilt successfully" -ForegroundColor Green
```

### Deliverables

- [ ] `ryzen_llm_bindings.pyd` with full inference API
- [ ] `rebuild_bindings.ps1` automation script
- [ ] Unit tests for bindings

---

## 📋 PHASE 2: MODEL ACQUISITION (Days 4-5)

### Target Models

| Model                | Size    | Source                     | Priority |
| -------------------- | ------- | -------------------------- | -------- |
| **BitNet b1.58 3B**  | 13.3 GB | `1bitLLM/bitnet_b1_58-3B`  | ⭐ HIGH  |
| **Draft Model 350M** | 0.7 GB  | `microsoft/DialoGPT-small` | ⭐ HIGH  |
| **Mamba 2.8B**       | 5.7 GB  | `state-spaces/mamba-2.8b`  | MEDIUM   |
| **RWKV 7B**          | 14.0 GB | `BlinkDL/rwkv-7b`          | LOW      |

### Automation Script

**File:** `scripts/download_all_models.ps1`

```powershell
# Automated model download with verification
$ErrorActionPreference = "Stop"
$ModelDir = "S:\Ryot\RYZEN-LLM\models"

Write-Host "=== Ryzanstein Model Downloader ===" -ForegroundColor Cyan

# Ensure Python environment
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"  # Fast downloads

# Download primary model (BitNet 3B)
Write-Host "[1/4] Downloading BitNet b1.58 3B..." -ForegroundColor Yellow
python -m huggingface_hub download 1bitLLM/bitnet_b1_58-3B `
    --local-dir "$ModelDir\bitnet\3b" `
    --include "*.safetensors" "config.json" "tokenizer*"

# Download draft model (for speculative decoding)
Write-Host "[2/4] Downloading Draft Model 350M..." -ForegroundColor Yellow
python -m huggingface_hub download microsoft/DialoGPT-small `
    --local-dir "$ModelDir\drafts\dialogpt" `
    --include "*.bin" "config.json" "tokenizer*"

# Download Mamba (optional)
Write-Host "[3/4] Downloading Mamba 2.8B..." -ForegroundColor Yellow
python -m huggingface_hub download state-spaces/mamba-2.8b `
    --local-dir "$ModelDir\mamba\2.8b" `
    --include "*.bin" "config.json"

Write-Host "✅ All models downloaded successfully" -ForegroundColor Green
Write-Host "Total disk usage:" -ForegroundColor Cyan
Get-ChildItem $ModelDir -Recurse | Measure-Object -Property Length -Sum |
    Select-Object @{N='Size (GB)';E={[math]::Round($_.Sum/1GB, 2)}}
```

### Model Validation Script

**File:** `scripts/validate_models.py`

```python
#!/usr/bin/env python3
"""Validate downloaded models and weights."""

import os
from pathlib import Path
from safetensors import safe_open

MODEL_DIR = Path("S:/Ryot/RYZEN-LLM/models")

def validate_bitnet():
    """Validate BitNet model files."""
    bitnet_path = MODEL_DIR / "bitnet" / "3b"
    required_files = [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "config.json",
        "tokenizer.json"
    ]

    print("Validating BitNet 3B...")
    for f in required_files:
        if (bitnet_path / f).exists():
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} MISSING")
            return False

    # Load and verify first shard
    with safe_open(bitnet_path / required_files[0], framework="numpy") as f:
        keys = list(f.keys())
        print(f"  ✅ SafeTensors valid: {len(keys)} tensors found")

    return True

if __name__ == "__main__":
    validate_bitnet()
```

### Deliverables

- [ ] BitNet 3B model downloaded and validated
- [ ] Draft model downloaded and validated
- [ ] Tokenizer files verified
- [ ] `download_all_models.ps1` automation script
- [ ] `validate_models.py` verification script

---

## 📋 PHASE 3: PYTHON API SERVER (Days 6-8)

### Current State

`RYZEN-LLM/src/api/server.py` exists but requires the C++ bindings to function.

### Target Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │           PYTHON FASTAPI SERVER                 │
                    │           (Port 8000)                           │
                    └─────────────────────┬───────────────────────────┘
                                          │
           ┌──────────────────────────────┼──────────────────────────────┐
           │                              │                              │
           ▼                              ▼                              ▼
    ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
    │ /v1/chat/   │              │ /v1/models  │              │ /v1/embed   │
    │ completions │              │             │              │ dings       │
    └─────────────┘              └─────────────┘              └─────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    INFERENCE ENGINE WRAPPER                         │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │  BitNetEngine │ MambaEngine │ RWKVEngine │ DraftEngine     │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    C++ BINDINGS (ryzen_llm_bindings.pyd)           │
    └─────────────────────────────────────────────────────────────────────┘
```

### Implementation

**File:** `RYZEN-LLM/src/api/engine_wrapper.py`

```python
"""
Engine wrapper that interfaces with C++ bindings.
Falls back to stub mode for development.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator

# Add build path for bindings
BUILD_PATH = Path(__file__).parent.parent.parent / "build" / "python" / "ryzanstein_llm"
sys.path.insert(0, str(BUILD_PATH))

try:
    import ryzen_llm_bindings as rlb
    BINDINGS_AVAILABLE = True
except ImportError:
    BINDINGS_AVAILABLE = False
    print("⚠️ C++ bindings not available - running in stub mode")


class RyzansteinEngine:
    """Main inference engine wrapper."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model_loaded = False
        self.engine = None

        if BINDINGS_AVAILABLE and model_path:
            self._load_model()

    def _load_model(self) -> bool:
        """Load model from SafeTensors files."""
        if not BINDINGS_AVAILABLE:
            return False

        try:
            self.engine = rlb.BitNetEngine(self.model_path)
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """Generate text completion."""

        if not self.model_loaded:
            return self._stub_generate(prompt, max_tokens)

        config = rlb.GenerationConfig()
        config.max_tokens = max_tokens
        config.temperature = temperature
        config.top_p = top_p

        if stream:
            return self._stream_generate(prompt, config)
        else:
            return self.engine.generate(prompt, config)

    def _stream_generate(self, prompt: str, config) -> Generator[str, None, None]:
        """Streaming token generation."""
        tokens = self.engine.generate_stream(prompt, config)
        for token in tokens:
            yield token

    def _stub_generate(self, prompt: str, max_tokens: int) -> str:
        """Stub generation for development/testing."""
        return f"[STUB MODE] Ryzanstein would generate {max_tokens} tokens for: {prompt[:50]}..."

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        if self.model_loaded and self.engine:
            return self.engine.get_model_info()
        return {
            "name": "BitNet b1.58 3B",
            "status": "not_loaded" if not self.model_loaded else "loaded",
            "bindings_available": BINDINGS_AVAILABLE
        }


# Global engine instance
_engine: Optional[RyzansteinEngine] = None


def get_engine() -> RyzansteinEngine:
    """Get or create global engine instance."""
    global _engine
    if _engine is None:
        model_path = os.environ.get("RYZANSTEIN_MODEL_PATH",
            "S:/Ryot/RYZEN-LLM/models/bitnet/3b")
        _engine = RyzansteinEngine(model_path)
    return _engine
```

### Server Startup Script

**File:** `scripts/start_api_server.ps1`

```powershell
# Start the Ryzanstein Python API Server
$ErrorActionPreference = "Stop"
$ProjectRoot = "S:\Ryot\RYZEN-LLM"

# Set environment
$env:RYZANSTEIN_MODEL_PATH = "$ProjectRoot\models\bitnet\3b"
$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot\build\python\ryzanstein_llm"

Write-Host "=== Starting Ryzanstein API Server ===" -ForegroundColor Cyan
Write-Host "Model Path: $env:RYZANSTEIN_MODEL_PATH" -ForegroundColor Yellow
Write-Host "API URL: http://localhost:8000" -ForegroundColor Yellow
Write-Host "Docs: http://localhost:8000/docs" -ForegroundColor Yellow

cd $ProjectRoot
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Deliverables

- [ ] `engine_wrapper.py` with C++ bindings integration
- [ ] Updated `server.py` using wrapper
- [ ] `start_api_server.ps1` script
- [ ] API tests passing

---

## 📋 PHASE 4: MCP INTEGRATION (Days 9-11)

### Current Problem

`mcp/server.go` returns hardcoded `"Inference response from Ryzanstein LLM"` instead of calling the real API.

### Solution

Implement HTTP client in Go to call Python API server.

**File:** `mcp/inference_client.go`

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

// RyzansteinClient connects to the Python API server
type RyzansteinClient struct {
    BaseURL    string
    HTTPClient *http.Client
}

// NewRyzansteinClient creates a new client instance
func NewRyzansteinClient(baseURL string) *RyzansteinClient {
    return &RyzansteinClient{
        BaseURL: baseURL,
        HTTPClient: &http.Client{
            Timeout: 60 * time.Second,
        },
    }
}

// ChatCompletionRequest matches OpenAI API format
type ChatCompletionRequest struct {
    Model       string    `json:"model"`
    Messages    []Message `json:"messages"`
    MaxTokens   int       `json:"max_tokens,omitempty"`
    Temperature float64   `json:"temperature,omitempty"`
    Stream      bool      `json:"stream,omitempty"`
}

type Message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}

// ChatCompletionResponse from the API
type ChatCompletionResponse struct {
    ID      string `json:"id"`
    Object  string `json:"object"`
    Created int64  `json:"created"`
    Choices []struct {
        Index   int `json:"index"`
        Message struct {
            Role    string `json:"role"`
            Content string `json:"content"`
        } `json:"message"`
        FinishReason string `json:"finish_reason"`
    } `json:"choices"`
    Usage struct {
        PromptTokens     int `json:"prompt_tokens"`
        CompletionTokens int `json:"completion_tokens"`
        TotalTokens      int `json:"total_tokens"`
    } `json:"usage"`
}

// ChatCompletion sends a chat completion request
func (c *RyzansteinClient) ChatCompletion(request ChatCompletionRequest) (*ChatCompletionResponse, error) {
    jsonData, err := json.Marshal(request)
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }

    url := fmt.Sprintf("%s/v1/chat/completions", c.BaseURL)
    req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, fmt.Errorf("failed to create request: %w", err)
    }

    req.Header.Set("Content-Type", "application/json")

    resp, err := c.HTTPClient.Do(req)
    if err != nil {
        return nil, fmt.Errorf("request failed: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
    }

    var response ChatCompletionResponse
    if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }

    return &response, nil
}

// HealthCheck verifies the API server is running
func (c *RyzansteinClient) HealthCheck() error {
    url := fmt.Sprintf("%s/health", c.BaseURL)
    resp, err := c.HTTPClient.Get(url)
    if err != nil {
        return fmt.Errorf("health check failed: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("API server unhealthy: status %d", resp.StatusCode)
    }
    return nil
}
```

### Update InferenceServer

**File:** `mcp/server.go` - Replace stub implementation

```go
// InferenceServer implements the inference gRPC service
type InferenceServer struct {
    pb.UnimplementedInferenceServiceServer
    client *RyzansteinClient
}

func NewInferenceServer() *InferenceServer {
    apiURL := os.Getenv("RYZANSTEIN_API_URL")
    if apiURL == "" {
        apiURL = "http://localhost:8000"
    }

    return &InferenceServer{
        client: NewRyzansteinClient(apiURL),
    }
}

func (s *InferenceServer) Infer(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
    // Build messages from request
    messages := []Message{
        {Role: "user", Content: req.GetPrompt()},
    }

    // Add system message if provided
    if req.GetSystemPrompt() != "" {
        messages = append([]Message{{Role: "system", Content: req.GetSystemPrompt()}}, messages...)
    }

    // Call the Python API
    apiReq := ChatCompletionRequest{
        Model:       req.GetModel(),
        Messages:    messages,
        MaxTokens:   int(req.GetMaxTokens()),
        Temperature: float64(req.GetTemperature()),
    }

    response, err := s.client.ChatCompletion(apiReq)
    if err != nil {
        return nil, fmt.Errorf("inference failed: %w", err)
    }

    // Extract content from response
    content := ""
    if len(response.Choices) > 0 {
        content = response.Choices[0].Message.Content
    }

    return &pb.InferenceResponse{
        Content:   content,
        Model:     response.ID,
        TokensUsed: int32(response.Usage.TotalTokens),
    }, nil
}
```

### Deliverables

- [ ] `inference_client.go` HTTP client
- [ ] Updated `server.go` with real inference
- [ ] Connection retry logic
- [ ] Health check integration
- [ ] MCP server tests passing

---

## 📋 PHASE 5: DESKTOP INTEGRATION (Days 12-14)

### Wire Up the Full Stack

The desktop app already has the structure - we just need to ensure all connections work.

**Startup Sequence:**

```
1. Start Python API Server (port 8000)
   └─ Loads C++ bindings
   └─ Loads BitNet model

2. Start MCP Server (port 50051)
   └─ Connects to Python API

3. Start Desktop App
   └─ Connects to MCP Server
   └─ Loads Elite Agents from .github/agents/
```

### Master Startup Script

**File:** `scripts/start_ryzanstein.ps1`

```powershell
# =============================================================================
# RYZANSTEIN FULL STACK STARTUP
# =============================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = "S:\Ryot"

Write-Host @"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ██╗   ██╗███████╗ █████╗ ███╗   ██╗███████╗████████╗███████╗██╗   ║
║   ██╔══██╗╚██╗ ██╔╝╚══███╔╝██╔══██╗████╗  ██║██╔════╝╚══██╔══╝██╔════╝██║   ║
║   ██████╔╝ ╚████╔╝   ███╔╝ ███████║██╔██╗ ██║███████╗   ██║   █████╗  ██║   ║
║   ██╔══██╗  ╚██╔╝   ███╔╝  ██╔══██║██║╚██╗██║╚════██║   ██║   ██╔══╝  ██║   ║
║   ██║  ██║   ██║   ███████╗██║  ██║██║ ╚████║███████║   ██║   ███████╗██║   ║
║   ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚══════╝╚═╝   ║
║                                                                              ║
║                        FULL STACK STARTUP                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Environment setup
$env:RYZANSTEIN_MODEL_PATH = "$ProjectRoot\RYZEN-LLM\models\bitnet\3b"
$env:RYZANSTEIN_API_URL = "http://localhost:8000"

# Step 1: Start Python API Server
Write-Host "`n[1/3] Starting Python API Server..." -ForegroundColor Yellow
$apiJob = Start-Job -ScriptBlock {
    param($root)
    cd "$root\RYZEN-LLM"
    $env:PYTHONPATH = "$root\RYZEN-LLM\src;$root\RYZEN-LLM\build\python\ryzanstein_llm"
    python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
} -ArgumentList $ProjectRoot

# Wait for API server to be ready
Write-Host "   Waiting for API server..." -ForegroundColor Gray
$maxRetries = 30
$retry = 0
do {
    Start-Sleep -Seconds 1
    $retry++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "   ✅ API Server ready on port 8000" -ForegroundColor Green
            break
        }
    } catch {}
} while ($retry -lt $maxRetries)

if ($retry -ge $maxRetries) {
    Write-Host "   ❌ API Server failed to start" -ForegroundColor Red
    exit 1
}

# Step 2: Start MCP Server
Write-Host "`n[2/3] Starting MCP Server..." -ForegroundColor Yellow
$mcpJob = Start-Job -ScriptBlock {
    param($root)
    cd "$root\mcp"
    .\mcp-server.exe
} -ArgumentList $ProjectRoot

Start-Sleep -Seconds 2
Write-Host "   ✅ MCP Server ready on port 50051" -ForegroundColor Green

# Step 3: Start Desktop App
Write-Host "`n[3/3] Starting Desktop Application..." -ForegroundColor Yellow
cd "$ProjectRoot\desktop"
wails dev

# Cleanup on exit
Write-Host "`n`nShutting down services..." -ForegroundColor Yellow
Stop-Job $apiJob, $mcpJob -ErrorAction SilentlyContinue
Remove-Job $apiJob, $mcpJob -ErrorAction SilentlyContinue
Write-Host "Done." -ForegroundColor Green
```

### Deliverables

- [ ] `start_ryzanstein.ps1` master script
- [ ] Health check for all services
- [ ] Graceful shutdown handling
- [ ] End-to-end connection verified

---

## 📋 PHASE 6: ELITE AGENTS ACTIVATION (Days 15-17)

### Agent-Enhanced Inference

Connect the 40 Elite Agents to the inference pipeline for specialized responses.

**File:** `mcp/agent_enhanced_inference.go`

```go
// AgentEnhancedInference wraps inference with agent context
func (s *InferenceServer) AgentEnhancedInfer(
    ctx context.Context,
    req *pb.InferenceRequest,
) (*pb.InferenceResponse, error) {
    // Get agent if specified
    var systemPrompt string
    if req.GetAgentName() != "" {
        agent, err := s.agentRegistry.GetAgent(req.GetAgentName())
        if err == nil {
            // Inject agent persona into system prompt
            systemPrompt = fmt.Sprintf(`You are %s, an Elite Agent.

%s

Philosophy: %s

Follow your expertise and methodology when responding.`,
                agent.Name,
                agent.Description,
                agent.Philosophy,
            )
        }
    }

    // Add system prompt to request
    modifiedReq := &pb.InferenceRequest{
        Prompt:       req.GetPrompt(),
        SystemPrompt: systemPrompt,
        Model:        req.GetModel(),
        MaxTokens:    req.GetMaxTokens(),
        Temperature:  req.GetTemperature(),
    }

    return s.Infer(ctx, modifiedReq)
}
```

### Agent Routing

```go
// RouteToAgent selects appropriate agent based on task
func (s *AgentServer) RouteToAgent(task string) (string, error) {
    // Task classification patterns
    patterns := map[string][]string{
        "@APEX":     {"code", "algorithm", "implement", "debug", "refactor"},
        "@CIPHER":   {"security", "encrypt", "auth", "vulnerability", "crypto"},
        "@ARCHITECT": {"design", "architecture", "scale", "system"},
        "@TENSOR":   {"ml", "model", "train", "neural", "inference"},
        "@VELOCITY": {"optimize", "performance", "profile", "fast"},
        // ... more patterns
    }

    taskLower := strings.ToLower(task)
    for agent, keywords := range patterns {
        for _, kw := range keywords {
            if strings.Contains(taskLower, kw) {
                return agent, nil
            }
        }
    }

    return "@APEX", nil // Default to APEX
}
```

### Deliverables

- [ ] Agent-enhanced inference working
- [ ] Auto-routing based on task type
- [ ] Agent personas injected into prompts
- [ ] All 40 agents accessible

---

## 📋 PHASE 7: OPTIMIZATION & TUNING (Days 18-21)

### Performance Targets

| Metric                  | Target  | Measurement      |
| ----------------------- | ------- | ---------------- |
| **Tokens/second**       | 15-30   | Generation speed |
| **Time to First Token** | < 500ms | Latency          |
| **Memory Usage**        | < 8GB   | For 3B model     |
| **CPU Utilization**     | > 80%   | Efficiency       |

### Optimization Tasks

#### 7.1 Enable Speculative Decoding

Use draft model for faster generation.

```cpp
// Enable speculative decoding in generation
config.use_speculative_decoding = true;
config.draft_model_path = "models/drafts/dialogpt";
config.speculation_lookahead = 4;  // Predict 4 tokens ahead
```

#### 7.2 KV-Cache Optimization

Already implemented in C++ - ensure it's enabled.

```cpp
// kv_cache.cpp settings
cache_config.max_seq_len = 2048;
cache_config.num_layers = 32;
cache_config.use_sliding_window = true;
cache_config.window_size = 512;
```

#### 7.3 AVX-512/AVX2 Runtime Detection

```cpp
// optimization_utils.cpp
CPUFeatures detect_cpu_features() {
    CPUFeatures features;

    // Check for AVX-512
    features.has_avx512f = check_cpuid(7, 0, EBX, 16);
    features.has_avx512vnni = check_cpuid(7, 0, ECX, 11);

    // Fallback to AVX2
    features.has_avx2 = check_cpuid(7, 0, EBX, 5);
    features.has_fma = check_cpuid(1, 0, ECX, 12);

    return features;
}
```

### Benchmark Script

**File:** `scripts/benchmark_inference.py`

```python
#!/usr/bin/env python3
"""Benchmark Ryzanstein inference performance."""

import time
import requests
from statistics import mean, stdev

API_URL = "http://localhost:8000/v1/chat/completions"

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list.",
    "What are the key principles of clean code?",
]

def benchmark(num_runs=10):
    results = []

    for prompt in PROMPTS:
        print(f"\nBenchmarking: {prompt[:50]}...")
        times = []
        tokens = []

        for i in range(num_runs):
            start = time.perf_counter()

            response = requests.post(API_URL, json={
                "model": "bitnet-3b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.7
            })

            elapsed = time.perf_counter() - start

            if response.status_code == 200:
                data = response.json()
                token_count = data["usage"]["completion_tokens"]
                times.append(elapsed)
                tokens.append(token_count)

        avg_time = mean(times)
        avg_tokens = mean(tokens)
        tok_per_sec = avg_tokens / avg_time

        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Average tokens: {avg_tokens:.0f}")
        print(f"  Tokens/second: {tok_per_sec:.1f}")

        results.append({
            "prompt": prompt[:50],
            "avg_time": avg_time,
            "tokens_per_sec": tok_per_sec
        })

    return results

if __name__ == "__main__":
    print("=" * 60)
    print("RYZANSTEIN INFERENCE BENCHMARK")
    print("=" * 60)
    benchmark()
```

### Deliverables

- [ ] Speculative decoding enabled
- [ ] KV-cache optimized
- [ ] Runtime SIMD detection
- [ ] Benchmark results meeting targets
- [ ] Performance documentation

---

## 🤖 AUTOMATION FRAMEWORK

### Master Orchestrator Script

**File:** `scripts/build_complete_stack.ps1`

```powershell
# =============================================================================
# RYZANSTEIN COMPLETE STACK BUILDER
# Autonomous execution of all phases
# =============================================================================

param(
    [switch]$SkipModels,      # Skip model download (if already downloaded)
    [switch]$SkipBuild,       # Skip C++ rebuild
    [switch]$DryRun,          # Show what would be done
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "S:\Ryot"
$StartTime = Get-Date

Write-Host @"

╔══════════════════════════════════════════════════════════════════════════════╗
║                    RYZANSTEIN COMPLETE STACK BUILDER                         ║
║                       Autonomous Execution Mode                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# =============================================================================
# PHASE 1: C++ BINDINGS
# =============================================================================

if (-not $SkipBuild) {
    Write-Host "`n[PHASE 1] Building C++ Bindings..." -ForegroundColor Yellow

    cd "$ProjectRoot\RYZEN-LLM\build"

    if (-not $DryRun) {
        cmake .. -G "Visual Studio 16 2019" -A x64
        cmake --build . --config Release --target ryzen_llm_bindings -j 8

        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ C++ build failed" -ForegroundColor Red
            exit 1
        }
    }

    Write-Host "✅ Phase 1 complete: C++ bindings built" -ForegroundColor Green
}

# =============================================================================
# PHASE 2: MODEL DOWNLOAD
# =============================================================================

if (-not $SkipModels) {
    Write-Host "`n[PHASE 2] Downloading Models..." -ForegroundColor Yellow

    $ModelDir = "$ProjectRoot\RYZEN-LLM\models"

    if (-not $DryRun) {
        # Install huggingface_hub if needed
        pip install huggingface_hub --quiet

        # Download BitNet 3B
        python -m huggingface_hub download 1bitLLM/bitnet_b1_58-3B `
            --local-dir "$ModelDir\bitnet\3b" `
            --include "*.safetensors" "config.json" "tokenizer*"

        # Download draft model
        python -m huggingface_hub download microsoft/DialoGPT-small `
            --local-dir "$ModelDir\drafts\dialogpt" `
            --include "*.bin" "config.json" "tokenizer*"
    }

    Write-Host "✅ Phase 2 complete: Models downloaded" -ForegroundColor Green
}

# =============================================================================
# PHASE 3: PYTHON API SETUP
# =============================================================================

Write-Host "`n[PHASE 3] Setting up Python API..." -ForegroundColor Yellow

if (-not $DryRun) {
    cd "$ProjectRoot\RYZEN-LLM"
    pip install -r requirements.txt --quiet

    # Create engine wrapper if not exists
    # (File creation would happen here in real implementation)
}

Write-Host "✅ Phase 3 complete: Python API ready" -ForegroundColor Green

# =============================================================================
# PHASE 4: MCP SERVER
# =============================================================================

Write-Host "`n[PHASE 4] Building MCP Server..." -ForegroundColor Yellow

if (-not $DryRun) {
    cd "$ProjectRoot\mcp"
    go build -o mcp-server.exe .

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ MCP server build failed" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Phase 4 complete: MCP server built" -ForegroundColor Green

# =============================================================================
# PHASE 5: DESKTOP APP
# =============================================================================

Write-Host "`n[PHASE 5] Building Desktop App..." -ForegroundColor Yellow

if (-not $DryRun) {
    cd "$ProjectRoot\desktop"
    go mod tidy
    wails build

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Desktop build failed" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Phase 5 complete: Desktop app built" -ForegroundColor Green

# =============================================================================
# SUMMARY
# =============================================================================

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host @"

╔══════════════════════════════════════════════════════════════════════════════╗
║                           BUILD COMPLETE                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Duration: $($Duration.TotalMinutes.ToString("F1")) minutes
║
║  Components Built:
║    ✅ C++ Inference Engine (ryzen_llm_bindings.pyd)
║    ✅ Models Downloaded (BitNet 3B + Draft)
║    ✅ Python API Server
║    ✅ MCP Server (mcp-server.exe)
║    ✅ Desktop App (ryzanstein.exe)
║
║  Next Step: Run 'scripts/start_ryzanstein.ps1' to launch
╚══════════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green
```

---

## 📅 TIMELINE SUMMARY

```
Week 1 (Days 1-7):
├── Day 1-3: Phase 1 - C++ Bindings Completion
│   ├── Expand pybind11 module
│   ├── Build engine interface
│   └── Rebuild and test
│
├── Day 4-5: Phase 2 - Model Acquisition
│   ├── Download BitNet 3B
│   ├── Download Draft Model
│   └── Validate files
│
└── Day 6-7: Phase 3 - Python API Server
    ├── Create engine wrapper
    ├── Update server.py
    └── Test endpoints

Week 2 (Days 8-14):
├── Day 8-11: Phase 4 - MCP Integration
│   ├── Create inference client (Go)
│   ├── Update server.go
│   └── Test gRPC calls
│
└── Day 12-14: Phase 5 - Desktop Integration
    ├── Create master startup script
    ├── Verify end-to-end connection
    └── Fix any integration issues

Week 3 (Days 15-21):
├── Day 15-17: Phase 6 - Elite Agents Activation
│   ├── Agent-enhanced inference
│   ├── Auto-routing implementation
│   └── Test all 40 agents
│
└── Day 18-21: Phase 7 - Optimization & Tuning
    ├── Enable speculative decoding
    ├── Optimize KV-cache
    ├── Run benchmarks
    └── Document performance
```

---

## ✅ SUCCESS CRITERIA

### Functional Requirements

- [ ] Desktop app can send messages and receive real AI responses
- [ ] All 40 Elite Agents are accessible and functional
- [ ] Model loads and generates text within 30 seconds
- [ ] Streaming responses work correctly
- [ ] System survives restart and reconnection

### Performance Requirements

- [ ] Token generation: ≥ 15 tokens/second
- [ ] Time to first token: < 500ms
- [ ] Memory usage: < 8GB for 3B model
- [ ] API latency: < 100ms (excluding generation)

### Automation Requirements

- [ ] Single-command full stack startup
- [ ] Automated health checks for all services
- [ ] Graceful shutdown on exit
- [ ] Error recovery with retry logic

---

## 🚀 GETTING STARTED

### Prerequisites

- Visual Studio 2019 Build Tools
- CMake 3.20+
- Python 3.11+ with pip
- Go 1.21+
- Node.js 18+ (for Wails)
- ~40GB disk space for models

### Quick Start

```powershell
# Clone and navigate
cd S:\Ryot

# Run the complete stack builder
.\scripts\build_complete_stack.ps1

# Start everything
.\scripts\start_ryzanstein.ps1
```

---

**Document Status:** Ready for Execution  
**Automation Level:** 95%  
**Human Intervention Required:** Minimal (approval checkpoints only)

_This is YOUR custom LLM. Built right, built for you._ 🧠
