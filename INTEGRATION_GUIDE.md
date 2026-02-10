# Ryzanstein LLM Integration Guide

**Using BitNet in Your Project**

> **Audience:** Developers integrating Ryzanstein LLM into applications  
> **Status:** ✅ Production Ready | **Tested:** Ryzanstein 7 7730U, Windows 11

---

## 📦 Installation

### Option A: Pre-built Python Package

```bash
# Install the Ryzanstein LLM package
pip install ryzanstein-llm

# Verify installation
python -c "import ryzanstein_llm; print(ryzanstein_llm.__version__)"
```

### Option B: Build from Source

Follow [QUICKSTART.md](./QUICKSTART.md) to build the C++ extension, then:

```powershell
cd Ryzanstein LLM
pip install -e .
```

---

## 🚀 Basic Usage

### 1. Initialize the Engine

```python
from ryzanstein_llm import BitNetEngine

# Create engine with default configuration
engine = BitNetEngine(
    model_type="bitnet1.58b",      # Quantization type
    max_tokens=2048,                 # Maximum sequence length
    device="cpu"                     # or "cuda" if available
)

print(f"Engine ready: {engine.status}")
```

### 2. Load Model Weights

```python
# Load pre-trained BitNet weights
engine.load_weights(
    path="./models/bitnet-1.58b-weights.safetensors",
    quantization_aware=True          # Enable QAT optimizations
)

print(f"Model loaded | Memory: {engine.memory_usage_mb} MB")
```

### 3. Run Inference

```python
# Simple text generation
prompt = "The future of AI is"
response = engine.generate(
    prompt=prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)

print(f"Prompt: {prompt}")
print(f"Response: {response}")
print(f"Throughput: {engine.throughput_tokens_per_sec} tok/s")
```

---

## ⚙️ Configuration

### Engine Parameters

```python
from ryzanstein_llm import BitNetEngine, EngineConfig

config = EngineConfig(
    # Model architecture
    vocab_size=32000,
    hidden_size=768,
    num_layers=12,
    num_heads=12,

    # Quantization (BitNet 1.58b)
    bit_width=1.58,
    activation_quantization=True,
    weight_quantization=True,

    # KV Cache optimization
    kv_cache_enabled=True,
    cache_dtype="float16",          # or "bfloat16"
    cache_max_batch_size=32,

    # Performance
    enable_flash_attention=True,
    enable_tensor_parallelism=False,

    # Memory
    max_memory_mb=500,              # Hard limit
    memory_efficient_mode=True
)

engine = BitNetEngine(config)
```

### T-MAC Configuration

```python
# Token-Aligned Memory Configuration
tmac_config = {
    "alignment_bytes": 64,          # Cache line alignment
    "prefetch_distance": 4,         # Prefetch lookahead
    "numa_aware": True,             # NUMA optimization
    "adaptive": True                # Adaptive prefetching
}

engine.configure_tmac(tmac_config)
```

---

## 🔌 Batch Processing

### Process Multiple Inputs

```python
prompts = [
    "What is machine learning?",
    "Explain neural networks",
    "Define AI safety"
]

results = engine.batch_generate(
    prompts=prompts,
    batch_size=3,
    max_new_tokens=50
)

for prompt, response in zip(prompts, results):
    print(f"\nQ: {prompt}")
    print(f"A: {response}")
    print(f"  Time: {engine.last_inference_time_ms}ms")
```

---

## 💾 Advanced Features

### KV Cache Management

```python
# Enable intelligent cache reuse
engine.enable_kv_cache_reuse()

# Pre-allocate cache for batch
engine.preallocate_cache(batch_size=32)

# Clear cache between generations
engine.clear_cache()

print(f"Cache utilization: {engine.cache_stats()}")
```

### Quantization Control

```python
# Use mixed precision (dynamic selection)
engine.set_quantization_mode("mixed")

# Or fixed-point quantization
engine.set_quantization_mode("int8")

# Query current mode
current = engine.get_quantization_mode()
print(f"Quantization: {current}")
```

### Performance Profiling

```python
# Enable profiling
engine.start_profiling()

# Run inference
response = engine.generate(prompt, max_new_tokens=100)

# Get profile
profile = engine.get_profile()
print(f"Token latency: {profile['token_latency_ms']} ms")
print(f"Compute time: {profile['compute_time_ms']} ms")
print(f"Memory peak: {profile['memory_peak_mb']} MB")
```

---

## 📊 Integration Patterns

### Pattern 1: Web API Server

```python
from fastapi import FastAPI
from ryzanstein_llm import BitNetEngine

app = FastAPI()
engine = BitNetEngine()
engine.load_weights("./models/bitnet.safetensors")

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 100):
    response = engine.generate(prompt, max_new_tokens=max_tokens)
    return {
        "prompt": prompt,
        "response": response,
        "throughput_tok_s": engine.throughput_tokens_per_sec
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Usage:**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

### Pattern 2: Batch Processing Pipeline

```python
import queue
from threading import Thread
from ryzanstein_llm import BitNetEngine

class InferencePipeline:
    def __init__(self, model_path: str):
        self.engine = BitNetEngine()
        self.engine.load_weights(model_path)
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Start worker thread
        worker = Thread(target=self._worker, daemon=True)
        worker.start()

    def _worker(self):
        while True:
            task_id, prompt = self.task_queue.get()
            response = self.engine.generate(prompt)
            self.result_queue.put((task_id, response))

    def submit(self, task_id: str, prompt: str):
        self.task_queue.put((task_id, prompt))

    def get_result(self, task_id: str, timeout=30):
        # Wait for result with timeout
        return self.result_queue.get(timeout=timeout)

# Usage
pipeline = InferencePipeline("./models/bitnet.safetensors")

# Submit tasks
for i, prompt in enumerate(["Task 1", "Task 2", "Task 3"]):
    pipeline.submit(f"task_{i}", prompt)

# Retrieve results
for i in range(3):
    task_id, result = pipeline.get_result(f"task_{i}")
    print(f"{task_id}: {result}")
```

### Pattern 3: Real-time Streaming

```python
from ryzanstein_llm import BitNetEngine

engine = BitNetEngine()
engine.load_weights("./models/bitnet.safetensors")

# Stream tokens as they're generated
prompt = "The revolution in AI started when"

print(f"Generating: ", end="", flush=True)
for token in engine.generate_streaming(prompt):
    print(token, end="", flush=True)
print()
```

---

## 🛡️ Error Handling

```python
from ryzanstein_llm import (
    BitNetEngine,
    OutOfMemoryError,
    QuantizationError,
    InvalidConfigError
)

try:
    engine = BitNetEngine()
    engine.load_weights("./models/bitnet.safetensors")
    response = engine.generate("Hello", max_new_tokens=100)

except OutOfMemoryError as e:
    print(f"Memory limit exceeded: {e}")
    # Reduce batch size or max_tokens

except QuantizationError as e:
    print(f"Quantization failed: {e}")
    # Check weight format

except InvalidConfigError as e:
    print(f"Invalid configuration: {e}")
    # Review config parameters
```

---

## 📈 Performance Tuning

| Setting                  | Impact               | Recommendation            |
| ------------------------ | -------------------- | ------------------------- |
| `enable_flash_attention` | ⚡ Throughput +15%   | Always enable             |
| `kv_cache_enabled`       | ⚡ Latency -30%      | Enable for long sequences |
| `cache_dtype="float16"`  | 💾 Memory -50%       | Use for limited VRAM      |
| `tensor_parallelism`     | ⚡ Multi-GPU scaling | Disable on single GPU     |
| `batch_size`             | 📊 Throughput        | Tune to available memory  |

See **[PERFORMANCE_REPORT.md](./PERFORMANCE_REPORT.md)** for benchmark results.

---

## 🐛 Debugging

### Enable Verbose Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ryzanstein_llm")
logger.setLevel(logging.DEBUG)

engine = BitNetEngine()
# Now all operations are logged
```

### Profile Memory Usage

```python
import tracemalloc

tracemalloc.start()

engine = BitNetEngine()
engine.load_weights("./models/bitnet.safetensors")
response = engine.generate("Hello", max_new_tokens=100)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
```

---

## 📚 API Reference

### Core Methods

| Method                                | Purpose                 | Returns         |
| ------------------------------------- | ----------------------- | --------------- |
| `generate(prompt, max_new_tokens)`    | Single inference        | `str`           |
| `batch_generate(prompts, batch_size)` | Batch inference         | `List[str]`     |
| `generate_streaming(prompt)`          | Streaming tokens        | `Iterator[str]` |
| `load_weights(path)`                  | Load model weights      | `None`          |
| `clear_cache()`                       | Reset KV cache          | `None`          |
| `get_profile()`                       | Get timing/memory stats | `Dict`          |

### Properties

| Property                    | Type    | Description                 |
| --------------------------- | ------- | --------------------------- |
| `throughput_tokens_per_sec` | `float` | Current throughput          |
| `memory_usage_mb`           | `float` | Current memory usage        |
| `cache_stats()`             | `Dict`  | Cache hit rate, utilization |
| `status`                    | `str`   | Engine operational status   |

---

## 🔗 Related Documentation

- **[QUICKSTART.md](./QUICKSTART.md)** – Build & setup guide
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** – Component architecture
- **[PERFORMANCE_REPORT.md](./PERFORMANCE_REPORT.md)** – Benchmark results
- **[DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)** – Production readiness

---

## ✨ Support & Examples

**More examples available at:**

- `examples/basic_inference.py` – Simple usage
- `examples/web_api.py` – FastAPI integration
- `examples/batch_processing.py` – Bulk processing
- `examples/streaming.py` – Token streaming

For issues: Open an issue on GitHub with:

1. Reproduction code
2. Engine configuration
3. Error messages & stack trace

---

**Status:** ✅ Production Ready  
**Last Updated:** December 2025  
**Tested On:** Ryzanstein 7 7730U, Windows 11
