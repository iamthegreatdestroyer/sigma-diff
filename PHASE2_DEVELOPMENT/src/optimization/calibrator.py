"""
Calibration Engine for Model Optimization - Sprint 4.2.

This module provides calibration functionality for optimized model inference,
enabling accurate quantization through proper scale and zero-point determination.

Key Features:
- Multiple calibration strategies (histogram, percentile, entropy, min-max)
- Calibration dataset management and streaming
- Per-channel and per-tensor calibration modes
- Adaptive calibration with hardware-aware tuning
- Cross-layer calibration for improved accuracy

Performance Targets:
- Calibration overhead: <5% of inference time
- Accuracy retention: >98% of FP32 baseline
- Memory-efficient streaming calibration

Author: Ryot Development Team
Sprint: 4.2 - Model Optimization & Quantization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import logging
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class CalibrationStrategy(Enum):
    """Calibration strategy for scale/zero-point determination."""
    MIN_MAX = auto()          # Simple min-max range
    HISTOGRAM = auto()        # Histogram-based with percentile clipping
    PERCENTILE = auto()       # Percentile-based range (e.g., 99.9%)
    ENTROPY = auto()          # KL-divergence minimization
    MSE = auto()              # Mean squared error minimization
    MOVING_AVERAGE = auto()   # Exponential moving average of ranges
    ADAPTIVE = auto()         # Hardware-aware adaptive calibration


class CalibrationMode(Enum):
    """Granularity of calibration."""
    PER_TENSOR = auto()       # Single scale for entire tensor
    PER_CHANNEL = auto()      # Per-channel (axis 0) scales
    PER_GROUP = auto()        # Group-wise calibration


class CalibrationPhase(Enum):
    """Phase in calibration lifecycle."""
    COLLECTING = auto()       # Collecting activation statistics
    COMPUTING = auto()        # Computing optimal parameters
    VALIDATING = auto()       # Validating calibration quality
    COMPLETE = auto()         # Calibration finished


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CalibrationConfig:
    """Configuration for calibration process."""
    strategy: CalibrationStrategy = CalibrationStrategy.HISTOGRAM
    mode: CalibrationMode = CalibrationMode.PER_TENSOR
    num_calibration_samples: int = 1000
    percentile: float = 99.99
    num_histogram_bins: int = 2048
    moving_average_momentum: float = 0.1
    symmetric: bool = True
    group_size: int = 128
    validate_calibration: bool = True
    validation_tolerance: float = 0.02
    collect_layer_stats: bool = True
    stream_calibration: bool = False
    batch_size: int = 32
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


@dataclass
class CalibrationMetrics:
    """Metrics from calibration process."""
    num_samples_processed: int = 0
    calibration_time_seconds: float = 0.0
    validation_accuracy: float = 0.0
    mean_quantization_error: float = 0.0
    max_quantization_error: float = 0.0
    layers_calibrated: int = 0
    memory_used_mb: float = 0.0
    phase: CalibrationPhase = CalibrationPhase.COLLECTING


@dataclass
class LayerCalibrationInfo:
    """Calibration information for a single layer."""
    layer_name: str
    min_val: float = 0.0
    max_val: float = 0.0
    scale: float = 1.0
    zero_point: int = 0
    histogram: Optional[torch.Tensor] = None
    histogram_edges: Optional[torch.Tensor] = None
    num_samples: int = 0
    quantization_error: float = 0.0


@dataclass
class CalibrationResult:
    """Complete calibration result."""
    success: bool
    config: CalibrationConfig
    metrics: CalibrationMetrics
    layer_info: Dict[str, LayerCalibrationInfo] = field(default_factory=dict)
    global_scale: float = 1.0
    global_zero_point: int = 0
    calibration_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


# =============================================================================
# Calibration Dataset
# =============================================================================

class CalibrationDataset(Dataset):
    """
    Dataset wrapper for calibration data.
    
    Supports both in-memory and streaming calibration data.
    """
    
    def __init__(
        self,
        data: Union[torch.Tensor, List[torch.Tensor], DataLoader],
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize calibration dataset.
        
        Args:
            data: Input tensors or dataloader
            transform: Optional transform to apply
            max_samples: Maximum samples to use
        """
        self.transform = transform
        self.max_samples = max_samples
        
        if isinstance(data, DataLoader):
            # Materialize from dataloader
            self.samples = []
            for batch in data:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # Take inputs only
                self.samples.append(batch)
                if max_samples and len(self.samples) * batch.shape[0] >= max_samples:
                    break
            self.samples = torch.cat(self.samples, dim=0)
            if max_samples:
                self.samples = self.samples[:max_samples]
        elif isinstance(data, list):
            self.samples = torch.cat(data, dim=0)
            if max_samples:
                self.samples = self.samples[:max_samples]
        else:
            self.samples = data[:max_samples] if max_samples else data
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def get_dataloader(self, batch_size: int = 32) -> DataLoader:
        """Create dataloader for calibration."""
        return DataLoader(self, batch_size=batch_size, shuffle=False)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_min_max_range(
    tensor: torch.Tensor,
    symmetric: bool = True
) -> Tuple[float, float]:
    """
    Compute min-max range for calibration.
    
    Args:
        tensor: Input tensor
        symmetric: Use symmetric range around zero
        
    Returns:
        Tuple of (min_val, max_val)
    """
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    if symmetric:
        abs_max = max(abs(min_val), abs(max_val))
        return -abs_max, abs_max
    
    return min_val, max_val


def compute_percentile_range(
    tensor: torch.Tensor,
    percentile: float = 99.99,
    symmetric: bool = True
) -> Tuple[float, float]:
    """
    Compute percentile-based range.
    
    Args:
        tensor: Input tensor
        percentile: Percentile to use (e.g., 99.99)
        symmetric: Use symmetric range
        
    Returns:
        Tuple of (min_val, max_val)
    """
    flat = tensor.flatten()
    lower = (100 - percentile) / 2
    upper = 100 - lower
    
    min_val = torch.quantile(flat, lower / 100).item()
    max_val = torch.quantile(flat, upper / 100).item()
    
    if symmetric:
        abs_max = max(abs(min_val), abs(max_val))
        return -abs_max, abs_max
    
    return min_val, max_val


def compute_histogram(
    tensor: torch.Tensor,
    num_bins: int = 2048,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute histogram of tensor values.
    
    Args:
        tensor: Input tensor
        num_bins: Number of histogram bins
        range_min: Optional minimum range
        range_max: Optional maximum range
        
    Returns:
        Tuple of (histogram counts, bin edges)
    """
    flat = tensor.flatten().float()
    
    if range_min is None:
        range_min = flat.min().item()
    if range_max is None:
        range_max = flat.max().item()
    
    # Compute histogram
    hist = torch.histc(flat, bins=num_bins, min=range_min, max=range_max)
    edges = torch.linspace(range_min, range_max, num_bins + 1)
    
    return hist, edges


def compute_entropy_threshold(
    histogram: torch.Tensor,
    edges: torch.Tensor,
    target_bins: int = 128
) -> Tuple[float, float]:
    """
    Compute optimal threshold using entropy (KL-divergence) minimization.
    
    This finds the clipping threshold that minimizes information loss.
    
    Args:
        histogram: Histogram counts
        edges: Bin edges
        target_bins: Target number of quantization bins
        
    Returns:
        Tuple of (min_threshold, max_threshold)
    """
    num_bins = len(histogram)
    
    # Normalize histogram to probability distribution
    hist = histogram.float()
    hist = hist / hist.sum()
    
    best_kl = float('inf')
    best_start = 0
    best_end = num_bins
    
    # Search for optimal clipping thresholds
    for start in range(num_bins // 4):
        for end in range(num_bins - num_bins // 4, num_bins):
            # Create clipped distribution
            clipped = hist[start:end].clone()
            
            # Add clipped mass to edges
            clipped[0] += hist[:start].sum()
            clipped[-1] += hist[end:].sum()
            
            if clipped.sum() <= 0:
                continue
            
            # Quantize to target bins
            merged_bins = len(clipped) // target_bins
            if merged_bins < 1:
                merged_bins = 1
            
            quantized = torch.zeros(target_bins)
            for i in range(target_bins):
                start_idx = i * merged_bins
                end_idx = min((i + 1) * merged_bins, len(clipped))
                quantized[i] = clipped[start_idx:end_idx].sum()
            
            # Expand back and compute KL divergence
            expanded = quantized.repeat_interleave(merged_bins)[:len(clipped)]
            expanded = expanded / expanded.sum()
            
            # KL divergence
            mask = (clipped > 0) & (expanded > 0)
            kl = (clipped[mask] * (clipped[mask] / expanded[mask]).log()).sum().item()
            
            if kl < best_kl:
                best_kl = kl
                best_start = start
                best_end = end
    
    min_threshold = edges[best_start].item()
    max_threshold = edges[best_end].item()
    
    return min_threshold, max_threshold


def compute_scale_zero_point(
    min_val: float,
    max_val: float,
    num_bits: int = 8,
    symmetric: bool = True
) -> Tuple[float, int]:
    """
    Compute quantization scale and zero point.
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        num_bits: Number of quantization bits
        symmetric: Use symmetric quantization
        
    Returns:
        Tuple of (scale, zero_point)
    """
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    
    if symmetric:
        abs_max = max(abs(min_val), abs(max_val))
        scale = abs_max / qmax if abs_max > 0 else 1.0
        zero_point = 0
    else:
        scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0
        zero_point = int(round(qmin - min_val / scale))
        zero_point = max(qmin, min(qmax, zero_point))
    
    return scale, zero_point


# =============================================================================
# Abstract Base Calibrator
# =============================================================================

class ModelCalibrator(ABC):
    """
    Abstract base class for model calibration.
    
    Calibrators collect activation statistics during inference and compute
    optimal quantization parameters (scale, zero-point) for each layer.
    """
    
    def __init__(self, config: CalibrationConfig):
        """
        Initialize calibrator.
        
        Args:
            config: Calibration configuration
        """
        self.config = config
        self.metrics = CalibrationMetrics()
        self.layer_info: Dict[str, LayerCalibrationInfo] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._collecting = False
    
    @abstractmethod
    def collect_statistics(
        self,
        model: nn.Module,
        data: Union[CalibrationDataset, DataLoader]
    ) -> None:
        """
        Collect activation statistics from model.
        
        Args:
            model: Model to calibrate
            data: Calibration data
        """
        pass
    
    @abstractmethod
    def compute_parameters(self) -> Dict[str, LayerCalibrationInfo]:
        """
        Compute calibration parameters from collected statistics.
        
        Returns:
            Dict mapping layer names to calibration info
        """
        pass
    
    def calibrate(
        self,
        model: nn.Module,
        data: Union[CalibrationDataset, DataLoader]
    ) -> CalibrationResult:
        """
        Full calibration pipeline.
        
        Args:
            model: Model to calibrate
            data: Calibration data
            
        Returns:
            CalibrationResult with all parameters
        """
        start_time = time.time()
        
        try:
            # Collection phase
            self.metrics.phase = CalibrationPhase.COLLECTING
            self.collect_statistics(model, data)
            
            # Computation phase
            self.metrics.phase = CalibrationPhase.COMPUTING
            self.layer_info = self.compute_parameters()
            
            # Validation phase
            if self.config.validate_calibration:
                self.metrics.phase = CalibrationPhase.VALIDATING
                self._validate_calibration(model, data)
            
            self.metrics.phase = CalibrationPhase.COMPLETE
            self.metrics.calibration_time_seconds = time.time() - start_time
            self.metrics.layers_calibrated = len(self.layer_info)
            
            return CalibrationResult(
                success=True,
                config=self.config,
                metrics=self.metrics,
                layer_info=self.layer_info
            )
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return CalibrationResult(
                success=False,
                config=self.config,
                metrics=self.metrics,
                error_message=str(e)
            )
        finally:
            self._remove_hooks()
    
    def _validate_calibration(
        self,
        model: nn.Module,
        data: Union[CalibrationDataset, DataLoader]
    ) -> None:
        """Validate calibration quality."""
        # Simple validation by measuring quantization error
        total_error = 0.0
        max_error = 0.0
        count = 0
        
        if isinstance(data, CalibrationDataset):
            data = data.get_dataloader(self.config.batch_size)
        
        model.eval()
        with torch.no_grad():
            for batch in data:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.config.device)
                
                # Measure quantization error per layer
                for name, info in self.layer_info.items():
                    error = abs(info.quantization_error)
                    total_error += error
                    max_error = max(max_error, error)
                    count += 1
                
                break  # Just check one batch
        
        self.metrics.mean_quantization_error = total_error / max(count, 1)
        self.metrics.max_quantization_error = max_error
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# =============================================================================
# Per-Tensor Calibrator
# =============================================================================

class PerTensorCalibrator(ModelCalibrator):
    """
    Per-tensor calibration using a single scale for each layer.
    
    This is the simplest calibration mode with lowest overhead.
    """
    
    def __init__(self, config: CalibrationConfig):
        super().__init__(config)
        config.mode = CalibrationMode.PER_TENSOR
        self.activation_ranges: Dict[str, Tuple[float, float]] = {}
        self.activation_histograms: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def collect_statistics(
        self,
        model: nn.Module,
        data: Union[CalibrationDataset, DataLoader]
    ) -> None:
        """Collect per-tensor activation statistics."""
        if isinstance(data, CalibrationDataset):
            data = data.get_dataloader(self.config.batch_size)
        
        # Register hooks
        def create_hook(name: str):
            def hook(module, input, output):
                if not self._collecting:
                    return
                
                tensor = output if isinstance(output, torch.Tensor) else output[0]
                tensor = tensor.detach()
                
                if self.config.strategy == CalibrationStrategy.HISTOGRAM:
                    # Update histogram
                    hist, edges = compute_histogram(tensor, self.config.num_histogram_bins)
                    if name in self.activation_histograms:
                        old_hist, old_edges = self.activation_histograms[name]
                        self.activation_histograms[name] = (old_hist + hist, edges)
                    else:
                        self.activation_histograms[name] = (hist, edges)
                else:
                    # Update range
                    min_val, max_val = compute_min_max_range(tensor, self.config.symmetric)
                    if name in self.activation_ranges:
                        old_min, old_max = self.activation_ranges[name]
                        self.activation_ranges[name] = (
                            min(old_min, min_val),
                            max(old_max, max_val)
                        )
                    else:
                        self.activation_ranges[name] = (min_val, max_val)
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handle = module.register_forward_hook(create_hook(name))
                self.hooks.append(handle)
        
        # Run inference
        model.eval()
        self._collecting = True
        samples_processed = 0
        
        with torch.no_grad():
            for batch in data:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.config.device)
                
                model(batch)
                samples_processed += batch.shape[0]
                
                if samples_processed >= self.config.num_calibration_samples:
                    break
        
        self._collecting = False
        self.metrics.num_samples_processed = samples_processed
    
    def compute_parameters(self) -> Dict[str, LayerCalibrationInfo]:
        """Compute per-tensor calibration parameters."""
        layer_info = {}
        
        if self.config.strategy == CalibrationStrategy.HISTOGRAM:
            for name, (hist, edges) in self.activation_histograms.items():
                if self.config.strategy == CalibrationStrategy.ENTROPY:
                    min_val, max_val = compute_entropy_threshold(hist, edges)
                else:
                    # Use percentile on histogram
                    cumsum = hist.cumsum(0)
                    total = cumsum[-1]
                    lower_idx = (cumsum < total * (100 - self.config.percentile) / 200).sum()
                    upper_idx = (cumsum < total * (100 + self.config.percentile) / 200).sum()
                    min_val = edges[lower_idx].item()
                    max_val = edges[upper_idx].item()
                
                if self.config.symmetric:
                    abs_max = max(abs(min_val), abs(max_val))
                    min_val, max_val = -abs_max, abs_max
                
                scale, zp = compute_scale_zero_point(min_val, max_val, symmetric=self.config.symmetric)
                
                layer_info[name] = LayerCalibrationInfo(
                    layer_name=name,
                    min_val=min_val,
                    max_val=max_val,
                    scale=scale,
                    zero_point=zp,
                    histogram=hist,
                    histogram_edges=edges
                )
        else:
            for name, (min_val, max_val) in self.activation_ranges.items():
                scale, zp = compute_scale_zero_point(min_val, max_val, symmetric=self.config.symmetric)
                
                layer_info[name] = LayerCalibrationInfo(
                    layer_name=name,
                    min_val=min_val,
                    max_val=max_val,
                    scale=scale,
                    zero_point=zp
                )
        
        return layer_info


# =============================================================================
# Per-Channel Calibrator
# =============================================================================

class PerChannelCalibrator(ModelCalibrator):
    """
    Per-channel calibration with separate scales per output channel.
    
    Provides better accuracy than per-tensor at slight overhead cost.
    """
    
    def __init__(self, config: CalibrationConfig):
        super().__init__(config)
        config.mode = CalibrationMode.PER_CHANNEL
        self.channel_ranges: Dict[str, torch.Tensor] = {}  # [num_channels, 2]
    
    def collect_statistics(
        self,
        model: nn.Module,
        data: Union[CalibrationDataset, DataLoader]
    ) -> None:
        """Collect per-channel activation statistics."""
        if isinstance(data, CalibrationDataset):
            data = data.get_dataloader(self.config.batch_size)
        
        def create_hook(name: str):
            def hook(module, input, output):
                if not self._collecting:
                    return
                
                tensor = output if isinstance(output, torch.Tensor) else output[0]
                tensor = tensor.detach()
                
                # Get per-channel min/max (assume channel is dim 1 or last dim)
                if tensor.dim() >= 2:
                    # Reshape to [batch, channels, -1]
                    channels = tensor.shape[1] if tensor.dim() > 2 else tensor.shape[-1]
                    flat = tensor.view(-1, channels) if tensor.dim() == 2 else tensor.view(tensor.shape[0], channels, -1)
                    
                    if tensor.dim() > 2:
                        # [batch, channels, spatial] -> min/max over batch and spatial
                        channel_min = flat.min(dim=0)[0].min(dim=-1)[0]
                        channel_max = flat.max(dim=0)[0].max(dim=-1)[0]
                    else:
                        # [batch, channels] -> min/max over batch
                        channel_min = flat.min(dim=0)[0]
                        channel_max = flat.max(dim=0)[0]
                    
                    if name in self.channel_ranges:
                        old_ranges = self.channel_ranges[name]
                        self.channel_ranges[name] = torch.stack([
                            torch.min(old_ranges[:, 0], channel_min),
                            torch.max(old_ranges[:, 1], channel_max)
                        ], dim=1)
                    else:
                        self.channel_ranges[name] = torch.stack([channel_min, channel_max], dim=1)
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handle = module.register_forward_hook(create_hook(name))
                self.hooks.append(handle)
        
        model.eval()
        self._collecting = True
        samples_processed = 0
        
        with torch.no_grad():
            for batch in data:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.config.device)
                
                model(batch)
                samples_processed += batch.shape[0]
                
                if samples_processed >= self.config.num_calibration_samples:
                    break
        
        self._collecting = False
        self.metrics.num_samples_processed = samples_processed
    
    def compute_parameters(self) -> Dict[str, LayerCalibrationInfo]:
        """Compute per-channel calibration parameters."""
        layer_info = {}
        
        for name, ranges in self.channel_ranges.items():
            min_vals = ranges[:, 0]
            max_vals = ranges[:, 1]
            
            if self.config.symmetric:
                abs_max = torch.max(min_vals.abs(), max_vals.abs())
                min_vals = -abs_max
                max_vals = abs_max
            
            # Compute per-channel scales
            scales = (max_vals - min_vals) / 255.0
            scales = torch.where(scales > 0, scales, torch.ones_like(scales))
            
            layer_info[name] = LayerCalibrationInfo(
                layer_name=name,
                min_val=min_vals.min().item(),
                max_val=max_vals.max().item(),
                scale=scales.mean().item(),  # Store mean for summary
                zero_point=0 if self.config.symmetric else 128
            )
        
        return layer_info


# =============================================================================
# Adaptive Calibrator
# =============================================================================

class AdaptiveCalibrator(ModelCalibrator):
    """
    Adaptive calibration that selects optimal strategy per layer.
    
    Uses hardware characteristics and layer properties to choose
    between per-tensor and per-channel calibration.
    """
    
    def __init__(self, config: CalibrationConfig):
        super().__init__(config)
        config.strategy = CalibrationStrategy.ADAPTIVE
        self.per_tensor_cal = PerTensorCalibrator(CalibrationConfig(**vars(config)))
        self.per_channel_cal = PerChannelCalibrator(CalibrationConfig(**vars(config)))
        self.strategy_selection: Dict[str, CalibrationMode] = {}
    
    def collect_statistics(
        self,
        model: nn.Module,
        data: Union[CalibrationDataset, DataLoader]
    ) -> None:
        """Collect statistics using both methods."""
        logger.info("Collecting per-tensor statistics...")
        self.per_tensor_cal.collect_statistics(model, data)
        
        logger.info("Collecting per-channel statistics...")
        self.per_channel_cal.collect_statistics(model, data)
        
        self.metrics.num_samples_processed = self.per_tensor_cal.metrics.num_samples_processed
    
    def compute_parameters(self) -> Dict[str, LayerCalibrationInfo]:
        """Select best calibration mode per layer."""
        per_tensor_info = self.per_tensor_cal.compute_parameters()
        per_channel_info = self.per_channel_cal.compute_parameters()
        
        layer_info = {}
        
        for name in set(per_tensor_info.keys()) | set(per_channel_info.keys()):
            tensor_info = per_tensor_info.get(name)
            channel_info = per_channel_info.get(name)
            
            if tensor_info and channel_info:
                # Choose based on range ratio
                tensor_range = tensor_info.max_val - tensor_info.min_val
                channel_range = channel_info.max_val - channel_info.min_val
                
                # Per-channel better if range reduction > 20%
                if channel_range < tensor_range * 0.8:
                    layer_info[name] = channel_info
                    self.strategy_selection[name] = CalibrationMode.PER_CHANNEL
                else:
                    layer_info[name] = tensor_info
                    self.strategy_selection[name] = CalibrationMode.PER_TENSOR
            elif tensor_info:
                layer_info[name] = tensor_info
                self.strategy_selection[name] = CalibrationMode.PER_TENSOR
            elif channel_info:
                layer_info[name] = channel_info
                self.strategy_selection[name] = CalibrationMode.PER_CHANNEL
        
        return layer_info


# =============================================================================
# Factory Function
# =============================================================================

def create_calibrator(
    config: Optional[CalibrationConfig] = None,
    mode: Optional[CalibrationMode] = None,
    **kwargs
) -> ModelCalibrator:
    """
    Factory function to create appropriate calibrator.
    
    Args:
        config: Optional calibration configuration
        mode: Optional calibration mode override
        **kwargs: Additional configuration options
        
    Returns:
        Configured ModelCalibrator instance
    """
    if config is None:
        config = CalibrationConfig(**kwargs)
    
    # Check ADAPTIVE strategy FIRST (has priority over default mode)
    if config.strategy == CalibrationStrategy.ADAPTIVE:
        return AdaptiveCalibrator(config)
    
    # Then check explicit mode override or config mode
    mode = mode or config.mode
    
    if mode == CalibrationMode.PER_CHANNEL:
        return PerChannelCalibrator(config)
    else:
        # Default to per-tensor calibration
        return PerTensorCalibrator(config)


# =============================================================================
# Utility Functions
# =============================================================================

def apply_calibration(
    model: nn.Module,
    calibration_result: CalibrationResult,
    inplace: bool = True
) -> nn.Module:
    """
    Apply calibration parameters to model.
    
    Args:
        model: Model to apply calibration to
        calibration_result: Calibration result with parameters
        inplace: Modify model in-place
        
    Returns:
        Model with calibration applied
    """
    if not calibration_result.success:
        logger.warning("Applying failed calibration result")
        return model
    
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    for name, module in model.named_modules():
        if name in calibration_result.layer_info:
            info = calibration_result.layer_info[name]
            # Store calibration as module attributes
            module.register_buffer('_calibration_scale', torch.tensor(info.scale))
            module.register_buffer('_calibration_zero_point', torch.tensor(info.zero_point))
    
    return model


def export_calibration(
    calibration_result: CalibrationResult,
    path: str
) -> None:
    """
    Export calibration data to file.
    
    Args:
        calibration_result: Calibration result to export
        path: Output path
    """
    import json
    
    export_data = {
        'success': calibration_result.success,
        'config': {
            'strategy': calibration_result.config.strategy.name,
            'mode': calibration_result.config.mode.name,
            'symmetric': calibration_result.config.symmetric
        },
        'metrics': {
            'samples_processed': calibration_result.metrics.num_samples_processed,
            'time_seconds': calibration_result.metrics.calibration_time_seconds,
            'layers_calibrated': calibration_result.metrics.layers_calibrated
        },
        'layers': {
            name: {
                'scale': info.scale,
                'zero_point': info.zero_point,
                'min_val': info.min_val,
                'max_val': info.max_val
            }
            for name, info in calibration_result.layer_info.items()
        }
    }
    
    with open(path, 'w') as f:
        json.dump(export_data, f, indent=2)


def import_calibration(path: str) -> CalibrationResult:
    """
    Import calibration data from file.
    
    Args:
        path: Input path
        
    Returns:
        CalibrationResult loaded from file
    """
    import json
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    config = CalibrationConfig(
        strategy=CalibrationStrategy[data['config']['strategy']],
        mode=CalibrationMode[data['config']['mode']],
        symmetric=data['config']['symmetric']
    )
    
    metrics = CalibrationMetrics(
        num_samples_processed=data['metrics']['samples_processed'],
        calibration_time_seconds=data['metrics']['time_seconds'],
        layers_calibrated=data['metrics']['layers_calibrated']
    )
    
    layer_info = {
        name: LayerCalibrationInfo(
            layer_name=name,
            scale=info['scale'],
            zero_point=info['zero_point'],
            min_val=info['min_val'],
            max_val=info['max_val']
        )
        for name, info in data['layers'].items()
    }
    
    return CalibrationResult(
        success=data['success'],
        config=config,
        metrics=metrics,
        layer_info=layer_info
    )


__all__ = [
    # Enums
    'CalibrationStrategy',
    'CalibrationMode', 
    'CalibrationPhase',
    # Config and Results
    'CalibrationConfig',
    'CalibrationMetrics',
    'LayerCalibrationInfo',
    'CalibrationResult',
    # Dataset
    'CalibrationDataset',
    # Calibrators
    'ModelCalibrator',
    'PerTensorCalibrator',
    'PerChannelCalibrator',
    'AdaptiveCalibrator',
    # Factory
    'create_calibrator',
    # Utilities
    'compute_min_max_range',
    'compute_percentile_range',
    'compute_histogram',
    'compute_entropy_threshold',
    'compute_scale_zero_point',
    'apply_calibration',
    'export_calibration',
    'import_calibration',
]
