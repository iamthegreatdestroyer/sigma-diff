"""
Weight Loader with Transparent Quantization Integration
[PHASE2-004] - Weight Loading with QuantizationEngine Integration

This module provides a unified weight loading interface that automatically
applies quantization to model weights during load, supporting:
- SafeTensors format
- PyTorch .pth format
- GGUF format (future)
- Transparent quantization via QuantizationEngine
- Layer-wise error measurement
- Compression statistics
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import struct
import io
from enum import Enum

# Try importing safetensors
try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# Try importing torch for bfloat16 handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .quantization import (
    QuantizationEngine,
    QuantizationConfig,
    create_default_config,
    create_aggressive_config
)


class WeightFormat(Enum):
    """Supported weight file formats."""
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    GGUF = "gguf"
    CUSTOM = "custom"


@dataclass
class CompressionStats:
    """Statistics for weight compression."""
    total_parameters: int = 0
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 0.0
    
    # Per-layer statistics
    layer_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Quantization error
    total_error: float = 0.0
    mean_layer_error: float = 0.0
    max_layer_error: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"CompressionStats(\n"
            f"  total_params={self.total_parameters:,}\n"
            f"  original={self.original_size_mb:.2f}MB\n"
            f"  quantized={self.quantized_size_mb:.2f}MB\n"
            f"  compression={self.compression_ratio:.2f}x\n"
            f"  mean_error={self.mean_layer_error:.6f}\n"
            f"  max_error={self.max_layer_error:.6f}\n"
            f")"
        )


@dataclass
class WeightLoaderConfig:
    """Configuration for weight loading and quantization."""
    # Quantization settings
    quantize: bool = True
    quantization_config: Optional[QuantizationConfig] = None
    auto_aggressive: bool = False  # Use aggressive quantization for large models
    
    # Loading settings
    device: str = "cpu"
    dtype: np.dtype = field(default_factory=lambda: np.float32)
    
    # Validation
    validate_shapes: bool = True
    compute_error: bool = True
    
    # Caching
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize quantization config if needed."""
        if self.quantize and self.quantization_config is None:
            if self.auto_aggressive:
                self.quantization_config = create_aggressive_config()
            else:
                self.quantization_config = create_default_config()


class WeightLoader:
    """
    Unified weight loader with transparent quantization.
    
    Supports loading model weights from various formats and automatically
    applies quantization to reduce model size while maintaining accuracy.
    
    Example:
        >>> loader = WeightLoader()
        >>> weights = await loader.load_safetensors(
        ...     "model.safetensors",
        ...     quantize=True
        ... )
        >>> print(weights.stats)
    """
    
    def __init__(self, config: Optional[WeightLoaderConfig] = None):
        """Initialize weight loader.
        
        Args:
            config: Loader configuration (uses defaults if None)
        """
        self.config = config or WeightLoaderConfig()
        self.quantizer = QuantizationEngine(
            self.config.quantization_config
        ) if self.config.quantize else None
        
        # Statistics tracking
        self.stats = CompressionStats()
        self.loaded_weights: Dict[str, Any] = {}
    
    def detect_format(self, file_path: Union[str, Path]) -> WeightFormat:
        """Detect weight file format from extension.
        
        Args:
            file_path: Path to weight file
            
        Returns:
            Detected format
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == ".safetensors":
            return WeightFormat.SAFETENSORS
        elif suffix == ".pth":
            return WeightFormat.PYTORCH
        elif suffix == ".gguf":
            return WeightFormat.GGUF
        else:
            return WeightFormat.CUSTOM
    
    def load_safetensors(
        self,
        file_path: Union[str, Path],
        quantize: Optional[bool] = None,
        layer_filter: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Load weights from SafeTensors format.
        
        Args:
            file_path: Path to .safetensors file
            quantize: Override config quantize setting
            layer_filter: List of layer names to load (None = load all)
            
        Returns:
            Dictionary mapping layer names to weight arrays (or TernaryWeight objects if quantized)
            
        Raises:
            ImportError: If safetensors is not installed
            FileNotFoundError: If weights file not found
            ValueError: If invalid SafeTensors format
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors package required for SafeTensors loading. "
                "Install with: pip install safetensors"
            )
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Weights file not found: {file_path}")
        
        quantize = quantize if quantize is not None else self.config.quantize
        
        # Load weights using safetensors library
        weights = {}
        with safe_open(str(file_path), framework="pt") as f:  # Use PyTorch framework to handle bfloat16
            for key in f.keys():
                # Apply layer filter if specified
                if layer_filter and not any(layer in key for layer in layer_filter):
                    continue
                
                # Load weight as PyTorch tensor first (handles bfloat16)
                weight_pt = f.get_tensor(key)
                
                # Convert to numpy, handling bfloat16
                if weight_pt.dtype == torch.bfloat16:
                    weight_pt = weight_pt.float()  # Convert bfloat16 to float32
                
                # Convert to numpy
                weight = weight_pt.numpy()
                
                # Ensure correct dtype
                if weight.dtype != self.config.dtype:
                    weight = weight.astype(self.config.dtype)
                
                weights[key] = weight
        
        # Apply quantization if enabled
        if quantize and self.quantizer:
            weights = self._quantize_weights(weights)
        
        self.loaded_weights = weights
        return weights
    
    def load_pytorch(
        self,
        file_path: Union[str, Path],
        quantize: Optional[bool] = None,
        layer_filter: Optional[List[str]] = None,
        map_location: str = "cpu",
    ) -> Dict[str, np.ndarray]:
        """Load weights from PyTorch .pth format.
        
        Args:
            file_path: Path to .pth file
            quantize: Override config quantize setting
            layer_filter: List of layer names to load (None = load all)
            map_location: Device to map weights to
            
        Returns:
            Dictionary mapping layer names to weight arrays
            
        Raises:
            ImportError: If torch is not installed
            FileNotFoundError: If weights file not found
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch package required for PyTorch loading. "
                "Install with: pip install torch"
            )
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Weights file not found: {file_path}")
        
        quantize = quantize if quantize is not None else self.config.quantize
        
        # Load PyTorch state dict
        state_dict = torch.load(str(file_path), map_location=map_location)
        
        # Convert to numpy
        weights = {}
        for key, tensor in state_dict.items():
            # Apply layer filter if specified
            if layer_filter and not any(layer in key for layer in layer_filter):
                continue
            
            # Convert tensor to numpy
            if isinstance(tensor, torch.Tensor):
                weight = tensor.detach().cpu().numpy()
            else:
                weight = np.array(tensor)
            
            weights[key] = weight.astype(self.config.dtype)
        
        # Apply quantization if enabled
        if quantize and self.quantizer:
            weights = self._quantize_weights(weights)
        
        self.loaded_weights = weights
        return weights
    
    def load_gguf(
        self,
        file_path: Union[str, Path],
        quantize: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """Load weights from GGUF format.
        
        Args:
            file_path: Path to .gguf file
            quantize: Override config quantize setting
            
        Returns:
            Dictionary mapping layer names to weight arrays
            
        Raises:
            NotImplementedError: GGUF loading not yet implemented
        """
        raise NotImplementedError(
            "GGUF loading not yet implemented. "
            "Use SafeTensors or PyTorch format for now."
        )
    
    def load(
        self,
        file_path: Union[str, Path],
        quantize: Optional[bool] = None,
        layer_filter: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Load weights from any supported format.
        
        Automatically detects format and loads accordingly.
        
        Args:
            file_path: Path to weight file
            quantize: Override config quantize setting
            layer_filter: List of layer names to load (None = load all)
            
        Returns:
            Dictionary mapping layer names to weight arrays
            
        Raises:
            ValueError: If format is not supported
        """
        format_type = self.detect_format(file_path)
        
        if format_type == WeightFormat.SAFETENSORS:
            return self.load_safetensors(
                file_path, quantize=quantize, layer_filter=layer_filter
            )
        elif format_type == WeightFormat.PYTORCH:
            return self.load_pytorch(
                file_path, quantize=quantize, layer_filter=layer_filter
            )
        elif format_type == WeightFormat.GGUF:
            return self.load_gguf(file_path, quantize=quantize)
        else:
            raise ValueError(f"Unsupported weight format: {format_type}")
    
    def _quantize_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Apply quantization to loaded weights.
        
        Args:
            weights: Dictionary of weight arrays
            
        Returns:
            Dictionary of quantized weights (TernaryWeight objects for 2D weights)
        """
        if not self.quantizer:
            return weights
        
        quantized = {}
        layer_errors = []
        total_original = 0
        total_quantized = 0
        
        for name, weight in weights.items():
            # Skip non-tensor data
            if not isinstance(weight, np.ndarray):
                quantized[name] = weight
                continue
            
            # Count parameters
            num_params = weight.size
            total_original += weight.nbytes
            
            try:
                # Quantize weights (handles 1D and 2D)
                ternary = self.quantizer.quantize_weights(weight, name)
                quantized[name] = ternary
                
                # Calculate compression
                quantized_bytes = ternary.size() + len(ternary.scales) * 4
                total_quantized += quantized_bytes
                
                # Compute error if enabled
                if self.config.compute_error:
                    recovered = self.quantizer.dequantize_weights(ternary)
                    error = self.quantizer.compute_error(weight, recovered)
                    
                    # Track per-layer statistics
                    self.stats.layer_stats[name] = {
                        'num_params': num_params,
                        'original_mb': weight.nbytes / (1024 * 1024),
                        'quantized_mb': quantized_bytes / (1024 * 1024),
                        'compression_ratio': weight.nbytes / quantized_bytes if quantized_bytes > 0 else 0,
                        'error': error,
                    }
                    layer_errors.append(error)
                
            except Exception as e:
                # If quantization fails, keep original
                print(f"Warning: Quantization failed for {name}: {e}")
                quantized[name] = weight
                total_quantized += weight.nbytes
        
        # Update global statistics
        self.stats.total_parameters = sum(
            w.size for w in weights.values() if isinstance(w, np.ndarray)
        )
        self.stats.original_size_mb = total_original / (1024 * 1024)
        self.stats.quantized_size_mb = total_quantized / (1024 * 1024)
        self.stats.compression_ratio = (
            total_original / total_quantized if total_quantized > 0 else 0
        )
        
        if layer_errors:
            self.stats.total_error = sum(layer_errors)
            self.stats.mean_layer_error = np.mean(layer_errors)
            self.stats.max_layer_error = np.max(layer_errors)
        
        return quantized
    
    def get_stats(self) -> CompressionStats:
        """Get compression statistics from last load.
        
        Returns:
            CompressionStats with details about quantization
        """
        return self.stats
    
    def save_stats(self, output_path: Union[str, Path]) -> None:
        """Save compression statistics to JSON file.
        
        Args:
            output_path: Path to save stats JSON
        """
        output_path = Path(output_path)
        
        stats_dict = {
            'total_parameters': self.stats.total_parameters,
            'original_size_mb': float(self.stats.original_size_mb),
            'quantized_size_mb': float(self.stats.quantized_size_mb),
            'compression_ratio': float(self.stats.compression_ratio),
            'total_error': float(self.stats.total_error),
            'mean_layer_error': float(self.stats.mean_layer_error),
            'max_layer_error': float(self.stats.max_layer_error),
            'layer_stats': self.stats.layer_stats,
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
    
    def clear_cache(self) -> None:
        """Clear loaded weights from memory."""
        self.loaded_weights.clear()
        if self.quantizer:
            self.quantizer.clear_cache()


# Convenience functions for quick loading

def load_weights(
    file_path: Union[str, Path],
    quantize: bool = True,
    aggressive: bool = False,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], CompressionStats]:
    """Quick load weights with optional quantization.
    
    Args:
        file_path: Path to weight file
        quantize: Enable quantization
        aggressive: Use aggressive quantization settings
        device: Target device
        
    Returns:
        Tuple of (weights dictionary, compression statistics)
        
    Example:
        >>> weights, stats = load_weights("model.safetensors", quantize=True)
        >>> print(f"Compression: {stats.compression_ratio:.2f}x")
    """
    config = WeightLoaderConfig(
        quantize=quantize,
        auto_aggressive=aggressive,
        device=device,
    )
    loader = WeightLoader(config)
    weights = loader.load(file_path)
    return weights, loader.get_stats()


def load_and_quantize(
    file_path: Union[str, Path],
    config: Optional[QuantizationConfig] = None,
) -> Tuple[Dict[str, Any], CompressionStats]:
    """Load weights and apply quantization with custom config.
    
    Args:
        file_path: Path to weight file
        config: Custom quantization configuration
        
    Returns:
        Tuple of (weights dictionary, compression statistics)
    """
    loader_config = WeightLoaderConfig(
        quantize=True,
        quantization_config=config or create_default_config(),
    )
    loader = WeightLoader(loader_config)
    weights = loader.load(file_path)
    return weights, loader.get_stats()


if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
        
        print(f"Loading weights from: {weights_path}")
        weights, stats = load_weights(weights_path, quantize=True)
        
        print(f"\n{stats}")
        print(f"\nLoaded {len(weights)} weight tensors")
        for name, weight in list(weights.items())[:3]:
            print(f"  {name}: shape={getattr(weight, 'shape', 'N/A')}")
    else:
        print("Usage: python weight_loader.py <weights_file>")
        print("Supported formats: .safetensors, .pth")
