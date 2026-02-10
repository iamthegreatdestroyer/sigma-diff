"""
Distributed Inference Communication Utilities

NCCL optimization helpers and benchmarking tools.
"""

import torch
import torch.distributed as dist
import logging
import time

logger = logging.getLogger(__name__)


class NCCLCommunicationBenchmark:
    """Benchmark NCCL communication operations.
    
    Measures latency and bandwidth for:
    - all_reduce
    - all_gather
    - broadcast
    """
    
    def __init__(self, rank: int, world_size: int, device: torch.device):
        """Initialize benchmark.
        
        Args:
            rank: Current rank
            world_size: Total number of ranks
            device: CUDA device
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device
    
    def benchmark_all_reduce(self, tensor_size: int, num_iterations: int = 100) -> float:
        """Benchmark all_reduce operation.
        
        Args:
            tensor_size: Size of tensor to reduce (in elements)
            num_iterations: Number of iterations
            
        Returns:
            Average latency in milliseconds
        """
        tensor = torch.randn(tensor_size, device=self.device)
        
        # Warmup
        for _ in range(10):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        
        # Timed
        start = time.time()
        for _ in range(num_iterations):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        latency_ms = (elapsed / num_iterations) * 1000
        return latency_ms
    
    def benchmark_all_gather(self, tensor_size: int, num_iterations: int = 100) -> float:
        """Benchmark all_gather operation.
        
        Args:
            tensor_size: Size of tensor per rank (in elements)
            num_iterations: Number of iterations
            
        Returns:
            Average latency in milliseconds
        """
        input_tensor = torch.randn(tensor_size, device=self.device)
        output_tensors = [torch.zeros_like(input_tensor) for _ in range(self.world_size)]
        
        # Warmup
        for _ in range(10):
            dist.all_gather(output_tensors, input_tensor)
        torch.cuda.synchronize()
        
        # Timed
        start = time.time()
        for _ in range(num_iterations):
            dist.all_gather(output_tensors, input_tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        latency_ms = (elapsed / num_iterations) * 1000
        return latency_ms
    
    def benchmark_broadcast(self, tensor_size: int, num_iterations: int = 100) -> float:
        """Benchmark broadcast operation.
        
        Args:
            tensor_size: Size of tensor to broadcast (in elements)
            num_iterations: Number of iterations
            
        Returns:
            Average latency in milliseconds
        """
        tensor = torch.randn(tensor_size, device=self.device)
        
        # Warmup
        for _ in range(10):
            dist.broadcast(tensor, src=0)
        torch.cuda.synchronize()
        
        # Timed
        start = time.time()
        for _ in range(num_iterations):
            dist.broadcast(tensor, src=0)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        latency_ms = (elapsed / num_iterations) * 1000
        return latency_ms


class CommunicationProfiler:
    """Profile communication operations for bottleneck identification."""
    
    def __init__(self):
        """Initialize profiler."""
        self.measurements = {}
    
    def record(self, op_name: str, latency_ms: float) -> None:
        """Record a communication operation latency.
        
        Args:
            op_name: Operation name (e.g., "all_reduce", "broadcast")
            latency_ms: Latency in milliseconds
        """
        if op_name not in self.measurements:
            self.measurements[op_name] = []
        self.measurements[op_name].append(latency_ms)
    
    def get_stats(self, op_name: str) -> dict:
        """Get statistics for an operation.
        
        Args:
            op_name: Operation name
            
        Returns:
            Dictionary with min, max, mean, median, p95
        """
        if op_name not in self.measurements:
            return {}
        
        measurements = sorted(self.measurements[op_name])
        n = len(measurements)
        
        return {
            "count": n,
            "min_ms": min(measurements),
            "max_ms": max(measurements),
            "mean_ms": sum(measurements) / n,
            "median_ms": measurements[n // 2],
            "p95_ms": measurements[int(n * 0.95)] if n > 0 else 0,
        }
    
    def print_report(self) -> None:
        """Print communication profiling report."""
        logger.info("=" * 60)
        logger.info("Communication Profiling Report")
        logger.info("=" * 60)
        
        for op_name in sorted(self.measurements.keys()):
            stats = self.get_stats(op_name)
            logger.info(f"\n{op_name}:")
            logger.info(f"  Count: {stats['count']}")
            logger.info(f"  Min: {stats['min_ms']:.3f} ms")
            logger.info(f"  Max: {stats['max_ms']:.3f} ms")
            logger.info(f"  Mean: {stats['mean_ms']:.3f} ms")
            logger.info(f"  Median: {stats['median_ms']:.3f} ms")
            logger.info(f"  P95: {stats['p95_ms']:.3f} ms")


class NCCLCommunicator:
    """NCCL-based communication handler for tensor parallelism.
    
    Provides high-level interface for collective operations used in
    distributed inference.
    """
    
    def __init__(self):
        """Initialize NCCL communicator."""
        self.initialized = torch.distributed.is_initialized()
        if not self.initialized:
            logger.warning("torch.distributed not initialized - operations will be no-ops")
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """Perform all-reduce operation.
        
        Args:
            tensor: Input tensor
            op: Reduction operation ("sum", "mean", "max", "min")
            
        Returns:
            Reduced tensor
        """
        if not self.initialized:
            # No-op for CPU-only testing
            return tensor
            
        if op == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(dist.get_world_size())
        elif op == "max":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        elif op == "min":
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Unsupported reduction operation: {op}")
            
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform all-gather operation.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Gathered tensor from all ranks
        """
        if not self.initialized:
            # No-op for CPU-only testing
            return tensor
            
        world_size = dist.get_world_size()
        output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output_tensors, tensor)
        
        return torch.stack(output_tensors, dim=0)
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank to all ranks.
        
        Args:
            tensor: Tensor to broadcast
            src: Source rank
            
        Returns:
            Broadcast tensor
        """
        if not self.initialized:
            # No-op for CPU-only testing
            return tensor
            
        dist.broadcast(tensor, src=src)
        return tensor
    
    def barrier(self):
        """Synchronization barrier across all ranks."""
        if self.initialized:
            dist.barrier()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Communication utilities module loaded")
