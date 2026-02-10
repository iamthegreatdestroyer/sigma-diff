# Task 5 Execution Guide

## Quick Start

```bash
# From Ryzanstein LLM root directory
python scripts/task_5_real_weight_testing.py
```

## What Happens

The script will:

1. **Download** BitNet 1.3B model (2.6 GB) from Hugging Face

   - First attempt: SafeTensors format (faster, safer)
   - Fallback: PyTorch format if SafeTensors unavailable
   - Cached locally in `storage/weights/`

2. **Load** weights using our WeightLoader

   - Auto-detects format from file extension
   - Handles SafeTensors and PyTorch transparently
   - Loads all 256+ layers into memory

3. **Quantize** each layer using QuantizationEngine

   - Applies aggressive ternary quantization
   - Measures error via dequantize + compare
   - Tracks per-layer statistics

4. **Analyze** compression and error metrics

   - Calculates compression ratio (expected 4-6x)
   - Validates output shapes
   - Aggregates error statistics

5. **Report** results
   - Prints formatted summary
   - Saves JSON report to `bitnet_quantization_report.json`
   - Confirms success or failure

## Expected Output

```
================================================================================
TASK 5: BitNet 1.3B Real Weight Testing
================================================================================

▶️  PHASE 1: Loading Weights
────────────────────────────────────────────────────────────────────────────────
📥 Downloading bitnet/BitNet-3B-last from Hugging Face...
✅ Downloaded SafeTensors format: [path]
📂 Loading weights from: [path]
✅ Loaded in 850.25ms

▶️  PHASE 2: Quantizing Model
────────────────────────────────────────────────────────────────────────────────
⚙️  Quantizing 256 layers...
  Quantizing layer.0.weight... MSE=0.001234
  Quantizing layer.0.bias... MSE=0.000456
  ...

📊 Quantization complete in 15234.56ms
  Mean error: 0.001567
  Max error:  0.012345
  Min error:  0.000123
  Std error:  0.002134

▶️  PHASE 3: Measuring Compression & Errors
────────────────────────────────────────────────────────────────────────────────
✓ Validating shapes...
  ✅ layer.0.weight: (64, 128)
  ✅ layer.0.bias: (128,)

▶️  PHASE 4: Final Report
────────────────────────────────────────────────────────────────────────────────
================================================================================
BitNet 1.3B QUANTIZATION TEST REPORT
================================================================================

📊 MODEL INFORMATION
  Model: bitnet/BitNet-3B-last
  URL: https://huggingface.co/bitnet/BitNet-3B-last
  Total Parameters: 3,340,000,000
  Total Layers: 256

📈 SIZE METRICS
  Original Size: 2604.25 MB
  Quantized Size: 434.04 MB
  Compression Ratio: 6.00x
  Size Reduction: 83.3%

❌ ERROR METRICS (Quantization Loss)
  Mean Error: 0.001567 MSE
  Max Error:  0.012345 MSE
  Min Error:  0.000123 MSE
  Std Dev:    0.002134
  Quantized Layers: 256/256

⏱️  PERFORMANCE METRICS
  Load Time: 850.25 ms
  Quantization Time: 15234.56 ms
  Total Time: 16084.81 ms

⚙️  CONFIGURATION
  Quantization Type: aggressive
  Timestamp: 2025-03-14T15:30:45.123456

================================================================================

✅ Report saved to: bitnet_quantization_report.json

✅ Task 5 Complete! BitNet 1.3B quantization validated.
```

## System Requirements

### Disk Space

- **Download:** 2.6 GB for BitNet 1.3B weights
- **Cache:** Additional 2.6 GB for local caching
- **Total:** ~5.2 GB required

### Memory

- **RAM:** 8 GB minimum (16 GB recommended for faster processing)
- **Peak Usage:** During quantization phase (~4 GB)

### Network

- **Internet:** Required for Hugging Face download
- **Speed:** Faster downloads preferred (can take 5-10 minutes on slow connections)
- **Retry:** Automatically handles timeouts

### Time

- **Download Time:** 5-15 minutes (depending on connection)
- **Quantization Time:** 15-20 seconds
- **Total Time:** 20-35 minutes (first run, includes download)
- **Subsequent Runs:** ~20 seconds (weights cached locally)

## Prerequisites

### Required Packages

```bash
# Install required packages
pip install huggingface-hub numpy safetensors

# Verify installation
python -c "import huggingface_hub; import numpy; import safetensors; print('✅ All required packages installed')"
```

### Optional Packages (for fallback support)

```bash
# For PyTorch format fallback
pip install torch

# Verify optional
python -c "try:
    import torch
    print('✅ PyTorch installed')
except ImportError:
    print('⚠️  PyTorch not installed (will use SafeTensors if available)')"
```

## Running the Script

### Option 1: Direct Execution

```bash
cd c:\Users\sgbil\Ryzanstein\Ryzanstein LLM
python scripts/task_5_real_weight_testing.py
```

### Option 2: With Error Output

```bash
cd c:\Users\sgbil\Ryzanstein\Ryzanstein LLM
python scripts/task_5_real_weight_testing.py 2>&1 | tee task_5_output.log
```

### Option 3: In Python REPL

```python
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path.cwd() / "Ryzanstein LLM"))

# Run directly
exec(open("scripts/task_5_real_weight_testing.py").read())
```

## Monitoring Progress

### Real-Time Tracking

The script prints progress at each phase:

1. Download progress (file size, progress bar)
2. Layer-by-layer quantization (layer name, MSE error)
3. Compression stats after each phase
4. Final report summary

### Log File Analysis

```bash
# View full execution log
cat task_5_output.log

# Count successful quantizations
grep "MSE=" task_5_output.log | wc -l

# Find any errors
grep "FAILED\|ERROR\|Exception" task_5_output.log

# Extract final statistics
grep -A 30 "QUANTIZATION TEST REPORT" task_5_output.log
```

## Expected Success Criteria

### File Operations

- ✅ BitNet model downloaded successfully
- ✅ File cached in `storage/weights/`
- ✅ WeightLoader successfully loads all formats
- ✅ All weight shapes preserved

### Quantization

- ✅ All (>95%) layers quantized
- ✅ Compression ratio: 4-6x
- ✅ Mean error: <0.01 MSE
- ✅ Max error: <0.05 MSE
- ✅ No NaN/Inf values

### Report Generation

- ✅ Statistics calculated
- ✅ JSON report created
- ✅ All fields populated
- ✅ Timestamps recorded

## Troubleshooting

### "huggingface_hub not installed"

```bash
pip install huggingface-hub
```

### "Model download fails (404 or timeout)"

**Solution 1: Check internet connection**

```bash
ping huggingface.co
```

**Solution 2: Verify model exists**

- Visit: https://huggingface.co/bitnet/BitNet-3B-last
- Confirm model is accessible
- Check for Terms of Use

**Solution 3: Manual download**

- Download from Hugging Face website
- Place in `storage/weights/`
- Run script again

### "Out of memory during quantization"

**Causes:**

- System has insufficient RAM
- Other processes consuming memory
- Large batch size

**Solutions:**

```bash
# Close other applications
# Restart system
# Run on machine with more RAM
# Edit script to process fewer layers at once
```

### "Quantization errors very high (>0.1 MSE)"

**Possible causes:**

- Aggressive config too aggressive
- Data type mismatch
- Quantization engine issue

**Solutions:**

1. Reduce aggressive settings
2. Verify Task 1-4 tests still passing
3. Check data types in output

### "Shapes don't match"

**Check:**

- Verify input weights format
- Confirm WeightLoader output
- Check quantization function

**Debug:**

```python
# Add to script to debug shapes
for name, weight in original_weights.items():
    quant = quantized_weights[name]
    print(f"{name}: {weight.shape} → {quant.shape}")
    if weight.shape != quant.shape:
        print(f"  ❌ MISMATCH!")
```

## Advanced Options

### Skip Download (Use Cached Weights)

```python
# Modify script to use cached weights
from pathlib import Path
cached_path = Path("storage/weights/...safetensors")
weights = tester.load_weights(cached_path)  # Skip download
```

### Modify Quantization Config

```python
# Use different quantization settings
from src.core.quantization import QuantizationEngine, QuantizationConfig

custom_config = QuantizationConfig(
    method="ternary",
    aggressive=True,  # or False
    bit_width=1,
    use_cache=True
)
engine = QuantizationEngine(custom_config)
```

### Process Subset of Layers

```python
# Quantize only first N layers for testing
quantized = {}
for i, (name, weight) in enumerate(original_weights.items()):
    if i >= 10:  # Only first 10 layers
        quantized[name] = weight
    else:
        quantized[name] = tester.quantizer.quantize_weights(weight, name)
```

## Output Files

### Primary Output

**File:** `bitnet_quantization_report.json`

```json
{
  "model_name": "bitnet/BitNet-3B-last",
  "total_parameters": 3340000000,
  "original_size_mb": 2604.25,
  "quantized_size_mb": 434.04,
  "compression_ratio": 6.0,
  "total_layers": 256,
  "quantized_layers": 256,
  "mean_error": 0.001567,
  "max_error": 0.012345,
  "min_error": 0.000123,
  "error_std": 0.002134,
  "quantization_time_ms": 15234.56,
  "load_time_ms": 850.25,
  "total_time_ms": 16084.81,
  "timestamp": "2025-03-14T15:30:45.123456",
  "model_url": "https://huggingface.co/bitnet/BitNet-3B-last",
  "quantization_config": "aggressive"
}
```

### Optional Outputs

**Log File** (if using tee):

- `task_5_output.log` - Complete execution log

**Cached Weights** (for future runs):

- `storage/weights/bitnet-3b-last/...` - Cached model files

## Validation Checklist

After Task 5 completes successfully:

- [ ] Script executed without errors
- [ ] Model downloaded (or used cached version)
- [ ] All layers quantized (>95% success)
- [ ] Compression ratio achieved: 4-6x
- [ ] Error metrics acceptable: Mean <0.01 MSE
- [ ] All output shapes valid
- [ ] JSON report generated
- [ ] No NaN/Inf values in results
- [ ] Timestamps recorded correctly

## Next Steps After Success

### Phase 2 Completion

1. **Inference Validation**

   - Load quantized weights into BitNetEngine
   - Run sample inference
   - Compare with non-quantized baseline

2. **Documentation Update**

   - Update PHASE_2_SESSION_SUMMARY.md
   - Record actual compression metrics
   - Include performance measurements

3. **Integration Testing**

   - Full end-to-end pipeline test
   - Multiple model sizes
   - Different quantization configs

4. **Production Deployment**
   - Package quantized weights
   - Create distribution format
   - Write deployment guide

---

## Support & Help

For detailed information:

- See `TASK_5_PLAN.md` for comprehensive guide
- See `PHASE_2_PROGRESS_REPORT.md` for overall status
- Check docstrings in `scripts/task_5_real_weight_testing.py`
- Review test implementations in `tests/test_weight_loader.py`

---

**Status:** Ready to Execute  
**Command:** `python scripts/task_5_real_weight_testing.py`  
**Duration:** ~20-35 minutes (includes download)  
**Success Probability:** HIGH (all dependencies tested)

**Let's validate the quantization pipeline with real BitNet weights! 🚀**
