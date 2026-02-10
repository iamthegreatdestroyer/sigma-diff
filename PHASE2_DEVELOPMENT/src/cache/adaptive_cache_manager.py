"""
Adaptive Cache Sizing Manager
=============================

Dynamically adjusts cache size based on memory pressure, workload patterns,
and system resources. Enables optimal cache utilization without OOM.

Key Features:
- Memory pressure monitoring (PSI, /proc/meminfo)
- Workload pattern analysis (hit rate, access frequency)
- Dynamic threshold adjustment
- Graceful degradation under pressure
- Integration with eviction policies
- Multi-tier cache management

Sprint 2.2: Days 5-6 Delivery
"""

import torch
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import psutil
import os

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure levels for adaptive sizing."""
    LOW = "low"           # < 50% memory used
    MODERATE = "moderate" # 50-70% memory used
    HIGH = "high"         # 70-85% memory used
    CRITICAL = "critical" # > 85% memory used


class CacheTier(Enum):
    """Cache tiers for multi-level management."""
    HOT = "hot"       # Frequently accessed, keep in memory
    WARM = "warm"     # Moderately accessed, may compress
    COLD = "cold"     # Rarely accessed, compress or evict


@dataclass
class MemoryStats:
    """Current memory statistics."""
    total_mb: float
    available_mb: float
    used_mb: float
    cache_mb: float
    swap_used_mb: float
    pressure_level: MemoryPressureLevel
    timestamp: float = field(default_factory=time.time)
    
    @property
    def usage_percent(self) -> float:
        return (self.used_mb / self.total_mb * 100) if self.total_mb > 0 else 0
    
    @property
    def available_percent(self) -> float:
        return (self.available_mb / self.total_mb * 100) if self.total_mb > 0 else 0


@dataclass
class WorkloadStats:
    """Workload statistics for sizing decisions."""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    requests_per_sec: float = 0.0
    cache_size_mb: float = 0.0
    evictions_per_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SizingDecision:
    """Cache sizing decision."""
    action: str  # "expand", "shrink", "maintain", "emergency_shrink"
    target_size_mb: float
    reason: str
    priority: int = 0  # Higher = more urgent
    confidence: float = 1.0


@dataclass
class AdaptiveSizingConfig:
    """Configuration for adaptive cache sizing."""
    # Memory thresholds
    low_memory_threshold: float = 0.50      # Start monitoring
    moderate_memory_threshold: float = 0.70  # Begin conservative mode
    high_memory_threshold: float = 0.85      # Aggressive eviction
    critical_memory_threshold: float = 0.95  # Emergency mode
    
    # Cache size bounds
    min_cache_size_mb: float = 64.0
    max_cache_size_mb: float = 8192.0
    initial_cache_size_mb: float = 512.0
    
    # Sizing parameters
    growth_factor: float = 1.5          # How much to grow when expanding
    shrink_factor: float = 0.7          # How much to shrink when contracting
    emergency_shrink_factor: float = 0.5 # Emergency shrink amount
    
    # Timing
    check_interval_sec: float = 1.0     # How often to check memory
    decision_interval_sec: float = 5.0   # How often to make sizing decisions
    cooldown_sec: float = 10.0           # Cooldown after size change
    
    # Workload thresholds
    min_hit_rate_for_growth: float = 0.6   # Need good hit rate to grow
    max_eviction_rate: float = 100.0       # Evictions/sec before shrinking
    
    # Target utilization
    target_memory_utilization: float = 0.75  # Target memory usage


class MemoryMonitor:
    """Monitors system memory and provides pressure signals."""
    
    def __init__(self, config: AdaptiveSizingConfig):
        self.config = config
        self.history: deque = deque(maxlen=60)  # 1 minute of history
        self._lock = threading.Lock()
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        total_mb = mem.total / (1024 * 1024)
        available_mb = mem.available / (1024 * 1024)
        used_mb = mem.used / (1024 * 1024)
        cached_mb = getattr(mem, 'cached', 0) / (1024 * 1024)
        swap_used_mb = swap.used / (1024 * 1024)
        
        # Determine pressure level
        usage_ratio = mem.percent / 100
        
        if usage_ratio < self.config.low_memory_threshold:
            pressure = MemoryPressureLevel.LOW
        elif usage_ratio < self.config.moderate_memory_threshold:
            pressure = MemoryPressureLevel.MODERATE
        elif usage_ratio < self.config.high_memory_threshold:
            pressure = MemoryPressureLevel.HIGH
        else:
            pressure = MemoryPressureLevel.CRITICAL
        
        stats = MemoryStats(
            total_mb=total_mb,
            available_mb=available_mb,
            used_mb=used_mb,
            cache_mb=cached_mb,
            swap_used_mb=swap_used_mb,
            pressure_level=pressure
        )
        
        with self._lock:
            self.history.append(stats)
        
        return stats
    
    def get_pressure_trend(self) -> str:
        """Analyze memory pressure trend: increasing, stable, decreasing."""
        with self._lock:
            if len(self.history) < 10:
                return "stable"
            
            recent = list(self.history)[-10:]
            older = list(self.history)[-20:-10] if len(self.history) >= 20 else recent
            
        recent_avg = sum(s.usage_percent for s in recent) / len(recent)
        older_avg = sum(s.usage_percent for s in older) / len(older)
        
        diff = recent_avg - older_avg
        
        if diff > 5:
            return "increasing"
        elif diff < -5:
            return "decreasing"
        else:
            return "stable"
    
    def predict_oom_seconds(self) -> Optional[float]:
        """Predict seconds until OOM based on trend."""
        with self._lock:
            if len(self.history) < 30:
                return None
            
            recent = list(self.history)[-30:]
        
        # Calculate memory consumption rate
        time_span = recent[-1].timestamp - recent[0].timestamp
        if time_span < 1:
            return None
        
        memory_change = recent[-1].used_mb - recent[0].used_mb
        rate_mb_per_sec = memory_change / time_span
        
        if rate_mb_per_sec <= 0:
            return None  # Not increasing
        
        available = recent[-1].available_mb
        seconds_to_oom = available / rate_mb_per_sec
        
        return max(0, seconds_to_oom)


class WorkloadAnalyzer:
    """Analyzes cache workload for sizing decisions."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.hit_times: deque = deque(maxlen=window_size)
        self.miss_times: deque = deque(maxlen=window_size)
        self.latencies: deque = deque(maxlen=window_size)
        self.evictions: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
        
    def record_hit(self, latency_ms: float) -> None:
        """Record a cache hit."""
        with self._lock:
            now = time.time()
            self.hit_times.append(now)
            self.latencies.append(latency_ms)
    
    def record_miss(self, latency_ms: float) -> None:
        """Record a cache miss."""
        with self._lock:
            now = time.time()
            self.miss_times.append(now)
            self.latencies.append(latency_ms)
    
    def record_eviction(self) -> None:
        """Record an eviction."""
        with self._lock:
            self.evictions.append(time.time())
    
    def get_stats(self, cache_size_mb: float = 0.0) -> WorkloadStats:
        """Get current workload statistics."""
        with self._lock:
            now = time.time()
            window_start = now - 60  # Last 60 seconds
            
            recent_hits = [t for t in self.hit_times if t > window_start]
            recent_misses = [t for t in self.miss_times if t > window_start]
            recent_evictions = [t for t in self.evictions if t > window_start]
            recent_latencies = list(self.latencies)
        
        total_requests = len(recent_hits) + len(recent_misses)
        hit_rate = len(recent_hits) / max(1, total_requests)
        miss_rate = 1 - hit_rate
        
        avg_latency = sum(recent_latencies) / max(1, len(recent_latencies))
        sorted_latencies = sorted(recent_latencies)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p99_latency = sorted_latencies[p99_idx] if sorted_latencies else 0
        
        time_window = min(60, now - (recent_hits[0] if recent_hits else now))
        rps = total_requests / max(1, time_window)
        evictions_per_sec = len(recent_evictions) / max(1, time_window)
        
        return WorkloadStats(
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            avg_latency_ms=avg_latency,
            p99_latency_ms=p99_latency,
            requests_per_sec=rps,
            cache_size_mb=cache_size_mb,
            evictions_per_sec=evictions_per_sec
        )


class AdaptiveCacheSizer:
    """
    Main controller for adaptive cache sizing.
    
    Monitors memory pressure and workload, making sizing decisions
    to optimize cache performance while avoiding OOM.
    """
    
    def __init__(
        self,
        config: AdaptiveSizingConfig,
        resize_callback: Optional[Callable[[float], None]] = None
    ):
        self.config = config
        self.resize_callback = resize_callback
        
        self.memory_monitor = MemoryMonitor(config)
        self.workload_analyzer = WorkloadAnalyzer()
        
        self.current_size_mb = config.initial_cache_size_mb
        self.last_resize_time = 0.0
        self.decision_history: deque = deque(maxlen=100)
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def start(self) -> None:
        """Start the adaptive sizing monitor."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Adaptive cache sizer started")
        
    def stop(self) -> None:
        """Stop the adaptive sizing monitor."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Adaptive cache sizer stopped")
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        last_decision_time = 0.0
        
        while self._running:
            try:
                # Get current stats
                mem_stats = self.memory_monitor.get_memory_stats()
                workload_stats = self.workload_analyzer.get_stats(self.current_size_mb)
                
                # Make sizing decision periodically
                now = time.time()
                if now - last_decision_time >= self.config.decision_interval_sec:
                    decision = self._make_sizing_decision(mem_stats, workload_stats)
                    
                    if decision.action != "maintain":
                        self._execute_decision(decision)
                    
                    self.decision_history.append(decision)
                    last_decision_time = now
                
                # Check for emergency conditions
                if mem_stats.pressure_level == MemoryPressureLevel.CRITICAL:
                    self._handle_emergency(mem_stats)
                
                time.sleep(self.config.check_interval_sec)
                
            except Exception as e:
                logger.error(f"Error in adaptive sizing monitor: {e}")
                time.sleep(self.config.check_interval_sec)
    
    def _make_sizing_decision(
        self,
        mem_stats: MemoryStats,
        workload_stats: WorkloadStats
    ) -> SizingDecision:
        """Make a cache sizing decision based on current stats."""
        
        # Check cooldown
        if time.time() - self.last_resize_time < self.config.cooldown_sec:
            return SizingDecision(
                action="maintain",
                target_size_mb=self.current_size_mb,
                reason="In cooldown period"
            )
        
        pressure = mem_stats.pressure_level
        trend = self.memory_monitor.get_pressure_trend()
        
        # Critical: Emergency shrink
        if pressure == MemoryPressureLevel.CRITICAL:
            target = self.current_size_mb * self.config.emergency_shrink_factor
            target = max(target, self.config.min_cache_size_mb)
            return SizingDecision(
                action="emergency_shrink",
                target_size_mb=target,
                reason=f"Critical memory pressure: {mem_stats.usage_percent:.1f}%",
                priority=10
            )
        
        # High pressure: Shrink
        if pressure == MemoryPressureLevel.HIGH:
            target = self.current_size_mb * self.config.shrink_factor
            target = max(target, self.config.min_cache_size_mb)
            return SizingDecision(
                action="shrink",
                target_size_mb=target,
                reason=f"High memory pressure: {mem_stats.usage_percent:.1f}%",
                priority=5
            )
        
        # Moderate pressure with increasing trend: Shrink conservatively
        if pressure == MemoryPressureLevel.MODERATE and trend == "increasing":
            target = self.current_size_mb * 0.9
            target = max(target, self.config.min_cache_size_mb)
            return SizingDecision(
                action="shrink",
                target_size_mb=target,
                reason="Moderate pressure with increasing trend",
                priority=3
            )
        
        # Low pressure with good hit rate: Consider growing
        if pressure == MemoryPressureLevel.LOW:
            if workload_stats.hit_rate >= self.config.min_hit_rate_for_growth:
                # Check if growth would help (not already at max)
                if self.current_size_mb < self.config.max_cache_size_mb:
                    target = min(
                        self.current_size_mb * self.config.growth_factor,
                        self.config.max_cache_size_mb
                    )
                    return SizingDecision(
                        action="expand",
                        target_size_mb=target,
                        reason=f"Low pressure, good hit rate: {workload_stats.hit_rate:.1%}",
                        priority=1
                    )
        
        # Default: Maintain current size
        return SizingDecision(
            action="maintain",
            target_size_mb=self.current_size_mb,
            reason="No sizing change needed"
        )
    
    def _execute_decision(self, decision: SizingDecision) -> None:
        """Execute a sizing decision."""
        with self._lock:
            old_size = self.current_size_mb
            self.current_size_mb = decision.target_size_mb
            self.last_resize_time = time.time()
        
        logger.info(
            f"Cache resize: {old_size:.1f}MB -> {decision.target_size_mb:.1f}MB "
            f"({decision.action}: {decision.reason})"
        )
        
        if self.resize_callback:
            try:
                self.resize_callback(decision.target_size_mb)
            except Exception as e:
                logger.error(f"Resize callback failed: {e}")
    
    def _handle_emergency(self, mem_stats: MemoryStats) -> None:
        """Handle emergency memory situation."""
        logger.warning(
            f"EMERGENCY: Memory usage at {mem_stats.usage_percent:.1f}%, "
            f"only {mem_stats.available_mb:.1f}MB available"
        )
        
        # Force immediate shrink
        with self._lock:
            target = self.config.min_cache_size_mb
            if self.current_size_mb > target:
                self.current_size_mb = target
                self.last_resize_time = time.time()
                
                if self.resize_callback:
                    try:
                        self.resize_callback(target)
                    except Exception as e:
                        logger.error(f"Emergency resize failed: {e}")
    
    def record_hit(self, latency_ms: float) -> None:
        """Record a cache hit for workload analysis."""
        self.workload_analyzer.record_hit(latency_ms)
    
    def record_miss(self, latency_ms: float) -> None:
        """Record a cache miss for workload analysis."""
        self.workload_analyzer.record_miss(latency_ms)
    
    def record_eviction(self) -> None:
        """Record an eviction for workload analysis."""
        self.workload_analyzer.record_eviction()
    
    def get_current_size_mb(self) -> float:
        """Get current cache size limit."""
        with self._lock:
            return self.current_size_mb
    
    def force_resize(self, size_mb: float) -> None:
        """Force a cache resize (bypasses normal decision making)."""
        size_mb = max(self.config.min_cache_size_mb, size_mb)
        size_mb = min(self.config.max_cache_size_mb, size_mb)
        
        with self._lock:
            self.current_size_mb = size_mb
            self.last_resize_time = time.time()
        
        if self.resize_callback:
            self.resize_callback(size_mb)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sizing statistics."""
        mem_stats = self.memory_monitor.get_memory_stats()
        workload_stats = self.workload_analyzer.get_stats(self.current_size_mb)
        
        return {
            "current_size_mb": self.current_size_mb,
            "min_size_mb": self.config.min_cache_size_mb,
            "max_size_mb": self.config.max_cache_size_mb,
            "memory": {
                "total_mb": mem_stats.total_mb,
                "available_mb": mem_stats.available_mb,
                "used_mb": mem_stats.used_mb,
                "usage_percent": mem_stats.usage_percent,
                "pressure_level": mem_stats.pressure_level.value,
                "trend": self.memory_monitor.get_pressure_trend()
            },
            "workload": {
                "hit_rate": workload_stats.hit_rate,
                "miss_rate": workload_stats.miss_rate,
                "avg_latency_ms": workload_stats.avg_latency_ms,
                "p99_latency_ms": workload_stats.p99_latency_ms,
                "requests_per_sec": workload_stats.requests_per_sec,
                "evictions_per_sec": workload_stats.evictions_per_sec
            },
            "decision_history": [
                {"action": d.action, "target_mb": d.target_size_mb, "reason": d.reason}
                for d in list(self.decision_history)[-10:]
            ]
        }


# Factory function
def create_adaptive_sizer(
    min_size_mb: float = 64,
    max_size_mb: float = 8192,
    initial_size_mb: float = 512,
    resize_callback: Optional[Callable[[float], None]] = None,
    **kwargs
) -> AdaptiveCacheSizer:
    """Create an adaptive cache sizer.
    
    Args:
        min_size_mb: Minimum cache size in MB
        max_size_mb: Maximum cache size in MB
        initial_size_mb: Initial cache size in MB
        resize_callback: Callback when resize is needed
        **kwargs: Additional config options
        
    Returns:
        Configured AdaptiveCacheSizer
    """
    config = AdaptiveSizingConfig(
        min_cache_size_mb=min_size_mb,
        max_cache_size_mb=max_size_mb,
        initial_cache_size_mb=initial_size_mb,
        **kwargs
    )
    
    return AdaptiveCacheSizer(config, resize_callback)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Adaptive Cache Sizing...")
    
    def on_resize(size_mb: float):
        print(f"Resize callback: {size_mb:.1f} MB")
    
    sizer = create_adaptive_sizer(
        min_size_mb=64,
        max_size_mb=2048,
        initial_size_mb=256,
        resize_callback=on_resize
    )
    
    # Simulate workload
    import random
    
    for i in range(100):
        if random.random() < 0.7:  # 70% hit rate
            sizer.record_hit(random.uniform(0.5, 5.0))
        else:
            sizer.record_miss(random.uniform(10, 50))
        
        if random.random() < 0.1:
            sizer.record_eviction()
    
    stats = sizer.get_stats()
    print(f"\nCurrent size: {stats['current_size_mb']:.1f} MB")
    print(f"Memory usage: {stats['memory']['usage_percent']:.1f}%")
    print(f"Pressure: {stats['memory']['pressure_level']}")
    print(f"Hit rate: {stats['workload']['hit_rate']:.1%}")
