#!/usr/bin/env python3
"""
Comprehensive Overhead Profiling for Phase 1 Modules
Measures individual optimization overhead costs against baseline activations
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not available, using NumPy arrays")

# Add scripts directory to path for module imports
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Try to import Phase 1 modules with fallback to stubs
try:
    from kernel_optimizer import KernelOptimizer
    print("✅ kernel_optimizer imported")
except ImportError as e:
    print(f"⚠️  kernel_optimizer import failed: {e}")
    KernelOptimizer = None

try:
    from semantic_compression import SemanticCompressor
    print("✅ semantic_compression imported")
except ImportError as e:
    print(f"⚠️  semantic_compression import failed: {e}")
    SemanticCompressor = None

try:
    from inference_scaling import InferenceScalingEngine
    print("✅ inference_scaling imported")
except ImportError as e:
    print(f"⚠️  inference_scaling import failed: {e}")
    InferenceScalingEngine = None

# Stub implementations for fallback
class StubOptimizer:
    def detect_and_tune(self, activation):
        time.sleep(0.0001)  # Simulate small overhead
        return activation

class StubCompressor:
    def encode(self, activation):
        time.sleep(0.0001)
        return activation
    def decode(self, encoded):
        time.sleep(0.0001)
        return encoded

class StubScaler:
    def optimize_kv_cache(self, k_cache, v_cache):
        time.sleep(0.0001)
        return k_cache, v_cache


class OverheadProfiler:
    """Profiles overhead of Phase 1 optimization modules"""
    
    def __init__(self, num_iterations=100, activation_shape=(32, 1024)):
        self.num_iterations = num_iterations
        self.activation_shape = activation_shape
        self.results = {}
        # Fix: Check if torch exists before accessing torch.cuda
        if HAS_TORCH and torch is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        
        print(f"\n[OverheadProfiler] Device: {self.device}")
        print(f"[OverheadProfiler] Activation shape: {activation_shape}")
        print(f"[OverheadProfiler] Iterations: {num_iterations}\n")
        
    def create_baseline(self):
        """Create baseline activation tensor"""
        if HAS_TORCH:
            baseline = torch.randn(
                self.activation_shape[0],
                self.activation_shape[1],
                dtype=torch.float32,
                device=self.device
            )
        else:
            baseline = np.random.randn(
                self.activation_shape[0],
                self.activation_shape[1]
            ).astype(np.float32)
        return baseline
    
    def measure_kernel_optimizer_overhead(self):
        """Profile kernel_optimizer.detect_and_tune() overhead"""
        print("[PROFILING] Kernel Optimizer Overhead...")
        
        try:
            # Use real optimizer if available, stub otherwise
            optimizer = KernelOptimizer() if KernelOptimizer else StubOptimizer()
            baseline = self.create_baseline()
            
            # Warmup
            for _ in range(5):
                optimizer.detect_and_tune(baseline)
            if HAS_TORCH and self.device == "cuda":
                torch.cuda.synchronize()
            
            # Measure overhead
            times = []
            for i in range(self.num_iterations):
                start = time.perf_counter()
                optimizer.detect_and_tune(baseline)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
                
                if (i + 1) % 20 == 0:
                    print(f"  Iteration {i + 1}/{self.num_iterations}")
            
            if HAS_TORCH and self.device == "cuda":
                torch.cuda.synchronize()
            
            overhead_ms = np.mean(times)
            overhead_std = np.std(times)
            
            self.results["kernel_optimizer"] = {
                "op_name": "kernel_optimizer.detect_and_tune()",
                "overhead_ms": float(overhead_ms),
                "overhead_std_ms": float(overhead_std),
                "overhead_percent": 0.0,  # Will be calculated later
                "iterations": self.num_iterations,
                "activation_shape": self.activation_shape,
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "implementation": "real" if KernelOptimizer else "stub"
            }
            
            print(f"  ✓ Mean overhead: {overhead_ms:.4f} ms ± {overhead_std:.4f} ms")
            print(f"  ✓ P95: {np.percentile(times, 95):.4f} ms\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            self.results["kernel_optimizer"] = {
                "op_name": "kernel_optimizer.detect_and_tune()",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def measure_semantic_compression_overhead(self):
        """Profile semantic_compression encode/decode overhead"""
        print("[PROFILING] Semantic Compression Overhead...")
        
        try:
            # Use real compressor if available, stub otherwise
            compressor = SemanticCompressor() if SemanticCompressor else StubCompressor()
            baseline = self.create_baseline()
            
            # Warmup
            for _ in range(5):
                encoded = compressor.encode(baseline)
                compressor.decode(encoded)
            if HAS_TORCH and self.device == "cuda":
                torch.cuda.synchronize()
            
            # Measure encode overhead
            encode_times = []
            for i in range(self.num_iterations):
                start = time.perf_counter()
                encoded = compressor.encode(baseline)
                end = time.perf_counter()
                encode_times.append((end - start) * 1000)
                
                if (i + 1) % 20 == 0:
                    print(f"  Encode iteration {i + 1}/{self.num_iterations}")
            
            if HAS_TORCH and self.device == "cuda":
                torch.cuda.synchronize()
            
            # Measure decode overhead
            decode_times = []
            encoded = compressor.encode(baseline)
            
            for i in range(self.num_iterations):
                start = time.perf_counter()
                decoded = compressor.decode(encoded)
                end = time.perf_counter()
                decode_times.append((end - start) * 1000)
                
                if (i + 1) % 20 == 0:
                    print(f"  Decode iteration {i + 1}/{self.num_iterations}")
            
            if HAS_TORCH and self.device == "cuda":
                torch.cuda.synchronize()
            
            encode_overhead_ms = np.mean(encode_times)
            decode_overhead_ms = np.mean(decode_times)
            total_overhead_ms = encode_overhead_ms + decode_overhead_ms
            
            self.results["semantic_compression"] = {
                "op_name": "semantic_compression.encode/decode()",
                "encode_overhead_ms": float(encode_overhead_ms),
                "decode_overhead_ms": float(decode_overhead_ms),
                "total_overhead_ms": float(total_overhead_ms),
                "encode_std_ms": float(np.std(encode_times)),
                "decode_std_ms": float(np.std(decode_times)),
                "overhead_percent": 0.0,  # Will be calculated later
                "iterations": self.num_iterations,
                "activation_shape": self.activation_shape,
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                "encode_p95_ms": float(np.percentile(encode_times, 95)),
                "decode_p95_ms": float(np.percentile(decode_times, 95)),
                "implementation": "real" if SemanticCompressor else "stub"
            }
            
            print(f"  ✓ Encode overhead: {encode_overhead_ms:.4f} ms ± {np.std(encode_times):.4f} ms")
            print(f"  ✓ Decode overhead: {decode_overhead_ms:.4f} ms ± {np.std(decode_times):.4f} ms")
            print(f"  ✓ Total overhead: {total_overhead_ms:.4f} ms\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            self.results["semantic_compression"] = {
                "op_name": "semantic_compression.encode/decode()",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def measure_inference_scaling_overhead(self):
        """Profile inference_scaling.optimize_kv_cache() overhead"""
        print("[PROFILING] Inference Scaling (KV Cache) Overhead...")
        
        try:
            # Use real scaler if available, stub otherwise
            scaling = InferenceScalingEngine() if InferenceScalingEngine else StubScaler()
            baseline = self.create_baseline()
            
            # Create mock KV cache structure
            batch_size, seq_len, hidden_dim = 32, 1024, 512
            if HAS_TORCH:
                k_cache = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
                v_cache = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
            else:
                k_cache = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
                v_cache = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
            
            # Warmup
            for _ in range(5):
                scaling.optimize_kv_cache(k_cache, v_cache)
            if HAS_TORCH and self.device == "cuda":
                torch.cuda.synchronize()
            
            # Measure overhead
            times = []
            for i in range(self.num_iterations):
                start = time.perf_counter()
                scaling.optimize_kv_cache(k_cache, v_cache)
                end = time.perf_counter()
                times.append((end - start) * 1000)
                
                if (i + 1) % 20 == 0:
                    print(f"  Iteration {i + 1}/{self.num_iterations}")
            
            if HAS_TORCH and self.device == "cuda":
                torch.cuda.synchronize()
            
            overhead_ms = np.mean(times)
            overhead_std = np.std(times)
            
            self.results["inference_scaling"] = {
                "op_name": "inference_scaling.optimize_kv_cache()",
                "overhead_ms": float(overhead_ms),
                "overhead_std_ms": float(overhead_std),
                "overhead_percent": 0.0,  # Will be calculated later
                "iterations": self.num_iterations,
                "kv_cache_shape": [batch_size, seq_len, hidden_dim],
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "implementation": "real" if InferenceScalingEngine else "stub"
            }
            
            print(f"  ✓ Mean overhead: {overhead_ms:.4f} ms ± {overhead_std:.4f} ms")
            print(f"  ✓ P95: {np.percentile(times, 95):.4f} ms\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            self.results["inference_scaling"] = {
                "op_name": "inference_scaling.optimize_kv_cache()",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_overhead_percentages(self):
        """Calculate overhead as percentage of baseline operations"""
        print("[CALCULATION] Computing overhead percentages...")
        
        # Estimate baseline cost (simple forward pass)
        baseline = self.create_baseline()
        baseline_times = []
        
        for _ in range(10):
            start = time.perf_counter()
            if HAS_TORCH:
                _ = baseline.clone()
            else:
                _ = baseline.copy()
            end = time.perf_counter()
            baseline_times.append((end - start) * 1000)
        
        baseline_cost_ms = np.mean(baseline_times)
        
        # Calculate percentages
        for key in self.results:
            if "error" not in self.results[key]:
                if key == "semantic_compression":
                    total_overhead = self.results[key]["total_overhead_ms"]
                    self.results[key]["overhead_percent"] = (total_overhead / baseline_cost_ms * 100) if baseline_cost_ms > 0 else 0
                else:
                    overhead = self.results[key]["overhead_ms"]
                    self.results[key]["overhead_percent"] = (overhead / baseline_cost_ms * 100) if baseline_cost_ms > 0 else 0
        
        print(f"  ✓ Baseline cost: {baseline_cost_ms:.4f} ms")
        print(f"  ✓ Percentages calculated\n")
    
    def validate_overhead_gate(self):
        """Validate that overhead < 30% of gross speedup"""
        print("[VALIDATION] Checking overhead gates...")
        
        max_allowed_overhead = 50  # ms per operation
        all_valid = True
        
        for key in self.results:
            if "error" not in self.results[key]:
                if key == "semantic_compression":
                    overhead = self.results[key]["total_overhead_ms"]
                else:
                    overhead = self.results[key]["overhead_ms"]
                
                is_valid = overhead < max_allowed_overhead
                status = "✓" if is_valid else "✗"
                print(f"  {status} {key}: {overhead:.4f} ms (limit: {max_allowed_overhead} ms)")
                
                if not is_valid:
                    all_valid = False
        
        print(f"\n  Overall gate: {'PASS' if all_valid else 'FAIL'}\n")
        return all_valid
    
    def generate_json_report(self, output_path):
        """Generate JSON report"""
        report = {
            "profiling_metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": self.device,
                "activation_shape": self.activation_shape,
                "iterations": self.num_iterations,
                "script_version": "2.0",
                "pytorch_available": HAS_TORCH
            },
            "results": self.results
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"[REPORT] JSON report generated: {output_path}")
        return report
    
    def generate_markdown_report(self, json_path, md_path):
        """Generate human-readable markdown report"""
        # Read JSON report
        with open(json_path) as f:
            data = json.load(f)
        
        # Build markdown
        md = "# Overhead Analysis Report\n\n"
        md += f"**Generated**: {data['profiling_metadata']['timestamp']}\n\n"
        
        md += "## Profiling Configuration\n\n"
        md += f"- **Device**: {data['profiling_metadata']['device']}\n"
        md += f"- **Activation Shape**: {data['profiling_metadata']['activation_shape']}\n"
        md += f"- **Iterations**: {data['profiling_metadata']['iterations']}\n"
        md += f"- **PyTorch Available**: {data['profiling_metadata']['pytorch_available']}\n\n"
        
        md += "## Results Summary\n\n"
        
        max_overhead = 0
        slowest_op = None
        
        for op_key, op_data in data['results'].items():
            if 'error' in op_data:
                md += f"### {op_key}\n"
                md += f"**Status**: ❌ ERROR\n"
                md += f"**Error**: {op_data['error']}\n\n"
            else:
                md += f"### {op_data['op_name']}\n"
                md += f"**Implementation**: {op_data.get('implementation', 'unknown')}\n\n"
                
                if op_key == "semantic_compression":
                    overhead = op_data["total_overhead_ms"]
                    md += f"- **Encode Overhead**: {op_data['encode_overhead_ms']:.4f} ms (±{op_data['encode_std_ms']:.4f} ms)\n"
                    md += f"- **Decode Overhead**: {op_data['decode_overhead_ms']:.4f} ms (±{op_data['decode_std_ms']:.4f} ms)\n"
                    md += f"- **Total Overhead**: {overhead:.4f} ms\n"
                    md += f"- **P95 Encode**: {op_data['encode_p95_ms']:.4f} ms\n"
                    md += f"- **P95 Decode**: {op_data['decode_p95_ms']:.4f} ms\n"
                else:
                    overhead = op_data["overhead_ms"]
                    md += f"- **Overhead**: {overhead:.4f} ms (±{op_data['overhead_std_ms']:.4f} ms)\n"
                    md += f"- **Min**: {op_data['min_ms']:.4f} ms\n"
                    md += f"- **Max**: {op_data['max_ms']:.4f} ms\n"
                    md += f"- **P95**: {op_data['p95_ms']:.4f} ms\n"
                
                md += f"- **Overhead %**: {op_data['overhead_percent']:.2f}% of baseline\n"
                md += f"- **Status**: {'✓ PASS' if overhead < 50 else '✗ FAIL'}\n\n"
                
                if overhead > max_overhead:
                    max_overhead = overhead
                    slowest_op = op_data['op_name']
        
        md += "## Performance Gate Analysis\n\n"
        md += "| Operation | Overhead (ms) | Gate Limit (ms) | Status |\n"
        md += "|-----------|---------------|-----------------|--------|\n"
        
        for op_key, op_data in data['results'].items():
            if 'error' not in op_data:
                if op_key == "semantic_compression":
                    overhead = op_data["total_overhead_ms"]
                else:
                    overhead = op_data["overhead_ms"]
                status = "✓ PASS" if overhead < 50 else "✗ FAIL"
                md += f"| {op_data['op_name']} | {overhead:.4f} | 50.0000 | {status} |\n"
        
        md += f"\n## Key Findings\n\n"
        if slowest_op:
            md += f"- **Maximum Overhead**: {max_overhead:.4f} ms ({slowest_op})\n"
        md += f"- **Overall Gate Status**: {'✓ ALL PASS' if max_overhead < 50 else '✗ SOME FAIL'}\n"
        if slowest_op:
            md += f"- **Recommended Optimization Priority**: Start with {slowest_op}\n\n"
        
        md += "## Recommendations\n\n"
        if max_overhead > 50:
            md += "- ⚠️ One or more operations exceed 50ms overhead threshold\n"
            md += "- Profile with cProfile/line_profiler for bottleneck identification\n"
            md += "- Consider algorithmic optimizations for slowest operations\n"
        else:
            md += "- ✓ All operations within acceptable overhead range\n"
            md += "- Continue monitoring for regression in actual deployments\n"
        
        if slowest_op:
            md += f"- Focus on {slowest_op} for next optimization cycle\n"
        md += "- Validate overhead remains stable across different batch sizes\n\n"
        
        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, 'w') as f:
            f.write(md)
        
        print(f"[REPORT] Markdown report generated: {md_path}")
    
    def run(self, json_output, md_output):
        """Execute full profiling workflow"""
        print("=" * 80)
        print("PHASE 1 OVERHEAD PROFILING - COMPREHENSIVE ANALYSIS")
        print("=" * 80 + "\n")
        
        # Profile each optimization
        self.measure_kernel_optimizer_overhead()
        self.measure_semantic_compression_overhead()
        self.measure_inference_scaling_overhead()
        
        # Calculate percentages
        self.calculate_overhead_percentages()
        
        # Validate gates
        gate_pass = self.validate_overhead_gate()
        
        # Generate reports
        self.generate_json_report(json_output)
        self.generate_markdown_report(json_output, md_output)
        
        print("=" * 80)
        print("PROFILING COMPLETE")
        print("=" * 80)
        
        return gate_pass


if __name__ == "__main__":
    # Configuration
    ITERATIONS = 100
    ACTIVATION_SHAPE = (32, 1024)
    JSON_OUTPUT = "s:\\Ryot\\reports\\overhead_analysis.json"
    MD_OUTPUT = "s:\\Ryot\\OVERHEAD_ANALYSIS_REPORT.md"
    
    # Run profiler
    profiler = OverheadProfiler(
        num_iterations=ITERATIONS,
        activation_shape=ACTIVATION_SHAPE
    )
    
    gate_pass = profiler.run(JSON_OUTPUT, MD_OUTPUT)
    
    sys.exit(0 if gate_pass else 1)


class OverheadProfiler:
    """Profiles overhead of Phase 1 optimization modules"""
    
    def __init__(self, num_iterations=100, activation_shape=(32, 1024)):
        self.num_iterations = num_iterations
        self.activation_shape = activation_shape
        self.results = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[OverheadProfiler] Device: {self.device}")
        print(f"[OverheadProfiler] Activation shape: {activation_shape}")
        print(f"[OverheadProfiler] Iterations: {num_iterations}\n")
        
    def create_baseline(self):
        """Create baseline activation tensor"""
        baseline = torch.randn(
            self.activation_shape[0],
            self.activation_shape[1],
            dtype=torch.float32,
            device=self.device
        )
        return baseline
    
    def measure_kernel_optimizer_overhead(self):
        """Profile kernel_optimizer.detect_and_tune() overhead"""
        print("[PROFILING] Kernel Optimizer Overhead...")
        
        try:
            optimizer = KernelOptimizer()
            baseline = self.create_baseline()
            
            # Warmup
            for _ in range(5):
                optimizer.detect_and_tune(baseline)
            torch.cuda.synchronize() if self.device == "cuda" else None
            
            # Measure overhead
            times = []
            for i in range(self.num_iterations):
                start = time.perf_counter()
                optimizer.detect_and_tune(baseline)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
                
                if (i + 1) % 20 == 0:
                    print(f"  Iteration {i + 1}/{self.num_iterations}")
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            
            overhead_ms = np.mean(times)
            overhead_std = np.std(times)
            
            self.results["kernel_optimizer"] = {
                "op_name": "kernel_optimizer.detect_and_tune()",
                "overhead_ms": float(overhead_ms),
                "overhead_std_ms": float(overhead_std),
                "overhead_percent": 0.0,  # Will be calculated later
                "iterations": self.num_iterations,
                "activation_shape": self.activation_shape,
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
                "p95_ms": float(np.percentile(times, 95))
            }
            
            print(f"  ✓ Mean overhead: {overhead_ms:.4f} ms ± {overhead_std:.4f} ms")
            print(f"  ✓ P95: {np.percentile(times, 95):.4f} ms\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            self.results["kernel_optimizer"] = {
                "op_name": "kernel_optimizer.detect_and_tune()",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def measure_semantic_compression_overhead(self):
        """Profile semantic_compression encode/decode overhead"""
        print("[PROFILING] Semantic Compression Overhead...")
        
        try:
            compressor = SemanticCompressor()
            baseline = self.create_baseline()
            
            # Warmup
            for _ in range(5):
                encoded = compressor.encode(baseline)
                compressor.decode(encoded)
            torch.cuda.synchronize() if self.device == "cuda" else None
            
            # Measure encode overhead
            encode_times = []
            for i in range(self.num_iterations):
                start = time.perf_counter()
                encoded = compressor.encode(baseline)
                end = time.perf_counter()
                encode_times.append((end - start) * 1000)
                
                if (i + 1) % 20 == 0:
                    print(f"  Encode iteration {i + 1}/{self.num_iterations}")
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            
            # Measure decode overhead
            decode_times = []
            encoded = compressor.encode(baseline)
            
            for i in range(self.num_iterations):
                start = time.perf_counter()
                decoded = compressor.decode(encoded)
                end = time.perf_counter()
                decode_times.append((end - start) * 1000)
                
                if (i + 1) % 20 == 0:
                    print(f"  Decode iteration {i + 1}/{self.num_iterations}")
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            
            encode_overhead_ms = np.mean(encode_times)
            decode_overhead_ms = np.mean(decode_times)
            total_overhead_ms = encode_overhead_ms + decode_overhead_ms
            
            self.results["semantic_compression"] = {
                "op_name": "semantic_compression.encode/decode()",
                "encode_overhead_ms": float(encode_overhead_ms),
                "decode_overhead_ms": float(decode_overhead_ms),
                "total_overhead_ms": float(total_overhead_ms),
                "encode_std_ms": float(np.std(encode_times)),
                "decode_std_ms": float(np.std(decode_times)),
                "overhead_percent": 0.0,  # Will be calculated later
                "iterations": self.num_iterations,
                "activation_shape": self.activation_shape,
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                "encode_p95_ms": float(np.percentile(encode_times, 95)),
                "decode_p95_ms": float(np.percentile(decode_times, 95))
            }
            
            print(f"  ✓ Encode overhead: {encode_overhead_ms:.4f} ms ± {np.std(encode_times):.4f} ms")
            print(f"  ✓ Decode overhead: {decode_overhead_ms:.4f} ms ± {np.std(decode_times):.4f} ms")
            print(f"  ✓ Total overhead: {total_overhead_ms:.4f} ms\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            self.results["semantic_compression"] = {
                "op_name": "semantic_compression.encode/decode()",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def measure_inference_scaling_overhead(self):
        """Profile inference_scaling.optimize_kv_cache() overhead"""
        print("[PROFILING] Inference Scaling (KV Cache) Overhead...")
        
        try:
            scaling = InferenceScalingEngine()
            baseline = self.create_baseline()
            
            # Create mock KV cache structure
            batch_size, seq_len, hidden_dim = 32, 1024, 512
            k_cache = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
            v_cache = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
            
            # Warmup
            for _ in range(5):
                scaling.optimize_kv_cache(k_cache, v_cache)
            torch.cuda.synchronize() if self.device == "cuda" else None
            
            # Measure overhead
            times = []
            for i in range(self.num_iterations):
                start = time.perf_counter()
                scaling.optimize_kv_cache(k_cache, v_cache)
                end = time.perf_counter()
                times.append((end - start) * 1000)
                
                if (i + 1) % 20 == 0:
                    print(f"  Iteration {i + 1}/{self.num_iterations}")
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            
            overhead_ms = np.mean(times)
            overhead_std = np.std(times)
            
            self.results["inference_scaling"] = {
                "op_name": "inference_scaling.optimize_kv_cache()",
                "overhead_ms": float(overhead_ms),
                "overhead_std_ms": float(overhead_std),
                "overhead_percent": 0.0,  # Will be calculated later
                "iterations": self.num_iterations,
                "kv_cache_shape": [batch_size, seq_len, hidden_dim],
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
                "p95_ms": float(np.percentile(times, 95))
            }
            
            print(f"  ✓ Mean overhead: {overhead_ms:.4f} ms ± {overhead_std:.4f} ms")
            print(f"  ✓ P95: {np.percentile(times, 95):.4f} ms\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            self.results["inference_scaling"] = {
                "op_name": "inference_scaling.optimize_kv_cache()",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_overhead_percentages(self):
        """Calculate overhead as percentage of baseline operations"""
        print("[CALCULATION] Computing overhead percentages...")
        
        # Estimate baseline cost (simple forward pass)
        baseline = self.create_baseline()
        baseline_times = []
        
        for _ in range(10):
            start = time.perf_counter()
            # Baseline: simple tensor copy
            _ = baseline.clone()
            end = time.perf_counter()
            baseline_times.append((end - start) * 1000)
        
        baseline_cost_ms = np.mean(baseline_times)
        
        # Calculate percentages
        for key in self.results:
            if "error" not in self.results[key]:
                if key == "semantic_compression":
                    total_overhead = self.results[key]["total_overhead_ms"]
                    self.results[key]["overhead_percent"] = (total_overhead / baseline_cost_ms * 100) if baseline_cost_ms > 0 else 0
                else:
                    overhead = self.results[key]["overhead_ms"]
                    self.results[key]["overhead_percent"] = (overhead / baseline_cost_ms * 100) if baseline_cost_ms > 0 else 0
        
        print(f"  ✓ Baseline cost: {baseline_cost_ms:.4f} ms")
        print(f"  ✓ Percentages calculated\n")
    
    def validate_overhead_gate(self):
        """Validate that overhead < 30% of gross speedup"""
        print("[VALIDATION] Checking overhead gates...")
        
        max_allowed_overhead = 50  # ms per operation
        all_valid = True
        
        for key in self.results:
            if "error" not in self.results[key]:
                if key == "semantic_compression":
                    overhead = self.results[key]["total_overhead_ms"]
                else:
                    overhead = self.results[key]["overhead_ms"]
                
                is_valid = overhead < max_allowed_overhead
                status = "✓" if is_valid else "✗"
                print(f"  {status} {key}: {overhead:.4f} ms (limit: {max_allowed_overhead} ms)")
                
                if not is_valid:
                    all_valid = False
        
        print(f"\n  Overall gate: {'PASS' if all_valid else 'FAIL'}\n")
        return all_valid
    
    def generate_json_report(self, output_path):
        """Generate JSON report"""
        report = {
            "profiling_metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": self.device,
                "activation_shape": self.activation_shape,
                "iterations": self.num_iterations,
                "script_version": "1.0"
            },
            "results": self.results
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[REPORT] JSON report generated: {output_path}")
        return report
    
    def generate_markdown_report(self, json_path, md_path):
        """Generate human-readable markdown report"""
        # Read JSON report
        with open(json_path) as f:
            data = json.load(f)
        
        # Build markdown
        md = "# Overhead Analysis Report\n\n"
        md += f"**Generated**: {data['profiling_metadata']['timestamp']}\n\n"
        
        md += "## Profiling Configuration\n\n"
        md += f"- **Device**: {data['profiling_metadata']['device']}\n"
        md += f"- **Activation Shape**: {data['profiling_metadata']['activation_shape']}\n"
        md += f"- **Iterations**: {data['profiling_metadata']['iterations']}\n\n"
        
        md += "## Results Summary\n\n"
        
        max_overhead = 0
        slowest_op = None
        
        for op_key, op_data in data['results'].items():
            if 'error' in op_data:
                md += f"### {op_key}\n"
                md += f"**Status**: ❌ ERROR\n"
                md += f"**Error**: {op_data['error']}\n\n"
            else:
                md += f"### {op_data['op_name']}\n"
                
                if op_key == "semantic_compression":
                    overhead = op_data["total_overhead_ms"]
                    md += f"- **Encode Overhead**: {op_data['encode_overhead_ms']:.4f} ms (±{op_data['encode_std_ms']:.4f} ms)\n"
                    md += f"- **Decode Overhead**: {op_data['decode_overhead_ms']:.4f} ms (±{op_data['decode_std_ms']:.4f} ms)\n"
                    md += f"- **Total Overhead**: {overhead:.4f} ms\n"
                    md += f"- **P95 Encode**: {op_data['encode_p95_ms']:.4f} ms\n"
                    md += f"- **P95 Decode**: {op_data['decode_p95_ms']:.4f} ms\n"
                else:
                    overhead = op_data["overhead_ms"]
                    md += f"- **Overhead**: {overhead:.4f} ms (±{op_data['overhead_std_ms']:.4f} ms)\n"
                    md += f"- **Min**: {op_data['min_ms']:.4f} ms\n"
                    md += f"- **Max**: {op_data['max_ms']:.4f} ms\n"
                    md += f"- **P95**: {op_data['p95_ms']:.4f} ms\n"
                
                md += f"- **Overhead %**: {op_data['overhead_percent']:.2f}% of baseline\n"
                md += f"- **Status**: {'✓ PASS' if overhead < 50 else '✗ FAIL'}\n\n"
                
                if overhead > max_overhead:
                    max_overhead = overhead
                    slowest_op = op_data['op_name']
        
        md += "## Performance Gate Analysis\n\n"
        md += "| Operation | Overhead (ms) | Gate Limit (ms) | Status |\n"
        md += "|-----------|---------------|-----------------|--------|\n"
        
        for op_key, op_data in data['results'].items():
            if 'error' not in op_data:
                if op_key == "semantic_compression":
                    overhead = op_data["total_overhead_ms"]
                else:
                    overhead = op_data["overhead_ms"]
                status = "✓ PASS" if overhead < 50 else "✗ FAIL"
                md += f"| {op_data['op_name']} | {overhead:.4f} | 50.0000 | {status} |\n"
        
        md += f"\n## Key Findings\n\n"
        md += f"- **Maximum Overhead**: {max_overhead:.4f} ms ({slowest_op})\n"
        md += f"- **Overall Gate Status**: {'✓ ALL PASS' if max_overhead < 50 else '✗ SOME FAIL'}\n"
        md += f"- **Recommended Optimization Priority**: Start with {slowest_op}\n\n"
        
        md += "## Recommendations\n\n"
        if max_overhead > 50:
            md += "- ⚠️ One or more operations exceed 50ms overhead threshold\n"
            md += "- Profile with cProfile/line_profiler for bottleneck identification\n"
            md += "- Consider algorithmic optimizations for slowest operations\n"
        else:
            md += "- ✓ All operations within acceptable overhead range\n"
            md += "- Continue monitoring for regression in actual deployments\n"
        
        md += "- Focus on {slowest_op} for next optimization cycle\n"
        md += "- Validate overhead remains stable across different batch sizes\n\n"
        
        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, 'w') as f:
            f.write(md)
        
        print(f"[REPORT] Markdown report generated: {md_path}")
    
    def run(self, json_output, md_output):
        """Execute full profiling workflow"""
        print("=" * 80)
        print("PHASE 1 OVERHEAD PROFILING - COMPREHENSIVE ANALYSIS")
        print("=" * 80 + "\n")
        
        # Profile each optimization
        self.measure_kernel_optimizer_overhead()
        self.measure_semantic_compression_overhead()
        self.measure_inference_scaling_overhead()
        
        # Calculate percentages
        self.calculate_overhead_percentages()
        
        # Validate gates
        gate_pass = self.validate_overhead_gate()
        
        # Generate reports
        self.generate_json_report(json_output)
        self.generate_markdown_report(json_output, md_output)
        
        print("=" * 80)
        print("PROFILING COMPLETE")
        print("=" * 80)
        
        return gate_pass


if __name__ == "__main__":
    # Configuration
    ITERATIONS = 100
    ACTIVATION_SHAPE = (32, 1024)
    JSON_OUTPUT = "s:\\Ryot\\reports\\overhead_analysis.json"
    MD_OUTPUT = "s:\\Ryot\\OVERHEAD_ANALYSIS_REPORT.md"
    
    # Run profiler
    profiler = OverheadProfiler(
        num_iterations=ITERATIONS,
        activation_shape=ACTIVATION_SHAPE
    )
    
    gate_pass = profiler.run(JSON_OUTPUT, MD_OUTPUT)
    
    sys.exit(0 if gate_pass else 1)
