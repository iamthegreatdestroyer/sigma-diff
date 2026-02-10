"""
BitNet Model Loader
===================

Loads and initializes BitNet models for inference.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import json
import numpy as np

from .config import BitNetConfig
from .quantization import QuantizedTensor, quantize_ternary
from ...api.api_types import ModelInfo, ModelType, QuantizationType
from ...api.exceptions import ModelNotLoadedError


class ModelLoader:
    """
    Loads BitNet models from disk.
    
    Supports:
    - SafeTensors format
    - PyTorch .bin format (converted)
    - Pre-quantized ternary format
    """
    
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.config: Optional[BitNetConfig] = None
        self.weights: Dict[str, QuantizedTensor] = {}
        self.embeddings: Optional[np.ndarray] = None
        self._loaded = False
    
    def load(self) -> "ModelLoader":
        """Load model from disk."""
        if not self.model_path.exists():
            raise ModelNotLoadedError(str(self.model_path))
        
        # Load config
        self.config = BitNetConfig.from_pretrained(self.model_path)
        
        # Find weight files
        weight_files = list(self.model_path.glob("*.safetensors"))
        if not weight_files:
            weight_files = list(self.model_path.glob("*.bin"))
        
        if not weight_files:
            # Try loading pre-quantized format
            weight_files = list(self.model_path.glob("*.ternary"))
        
        if not weight_files:
            raise ModelNotLoadedError(f"No weight files found in {self.model_path}")
        
        # Load weights
        for weight_file in weight_files:
            self._load_weight_file(weight_file)
        
        self._loaded = True
        return self
    
    def _load_weight_file(self, path: Path) -> None:
        """Load a single weight file."""
        if path.suffix == ".safetensors":
            self._load_safetensors(path)
        elif path.suffix == ".bin":
            self._load_pytorch(path)
        elif path.suffix == ".ternary":
            self._load_ternary(path)
        else:
            raise ValueError(f"Unknown weight format: {path.suffix}")
    
    def _load_safetensors(self, path: Path) -> None:
        """Load SafeTensors format."""
        try:
            from safetensors.numpy import load_file
            tensors = load_file(str(path))
            
            for name, tensor in tensors.items():
                if "embed" in name.lower():
                    # Keep embeddings as float
                    self.embeddings = tensor.astype(np.float32)
                else:
                    # Quantize to ternary
                    self.weights[name] = quantize_ternary(
                        tensor,
                        group_size=self.config.quantization_group_size
                    )
        except ImportError:
            raise ImportError("safetensors package required: pip install safetensors")
    
    def _load_pytorch(self, path: Path) -> None:
        """Load PyTorch .bin format."""
        try:
            import torch
            state_dict = torch.load(path, map_location="cpu")
            
            for name, tensor in state_dict.items():
                np_tensor = tensor.numpy()
                
                if "embed" in name.lower():
                    self.embeddings = np_tensor.astype(np.float32)
                else:
                    self.weights[name] = quantize_ternary(
                        np_tensor,
                        group_size=self.config.quantization_group_size
                    )
        except ImportError:
            raise ImportError("torch package required for .bin files")
    
    def _load_ternary(self, path: Path) -> None:
        """Load pre-quantized ternary format."""
        data = np.load(path, allow_pickle=True)
        
        for name in data.files:
            item = data[name].item()
            
            if isinstance(item, dict) and "packed_weights" in item:
                self.weights[name] = QuantizedTensor(
                    packed_weights=item["packed_weights"],
                    scales=item["scales"],
                    shape=tuple(item["shape"]),
                    group_size=item.get("group_size", 128),
                )
            else:
                # Assume embeddings
                self.embeddings = item.astype(np.float32)
    
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        if not self._loaded:
            raise ModelNotLoadedError("Model not loaded")
        
        # Calculate model size
        total_bytes = 0
        if self.embeddings is not None:
            total_bytes += self.embeddings.nbytes
        for weight in self.weights.values():
            total_bytes += weight.nbytes
        
        return ModelInfo(
            model_id=self.model_path.name,
            model_type=ModelType.BITNET,
            parameter_count=self._count_parameters(),
            context_window=self.config.max_position_embeddings,
            vocab_size=self.config.vocab_size,
            quantization=QuantizationType.TERNARY,
            bits_per_weight=self.config.bits_per_weight,
            model_size_bytes=total_bytes,
            estimated_memory_mb=total_bytes / (1024 * 1024),
            supports_streaming=True,
            supports_batching=True,
            supports_sigma_compression=True,
            supports_kv_recycling=True,
        )
    
    def _count_parameters(self) -> int:
        """Count total parameters."""
        total = 0
        
        if self.embeddings is not None:
            total += self.embeddings.size
        
        for weight in self.weights.values():
            total += int(np.prod(weight.shape))
        
        return total
    
    def get_weight(self, name: str) -> QuantizedTensor:
        """Get a specific weight tensor."""
        if name not in self.weights:
            raise KeyError(f"Weight not found: {name}")
        return self.weights[name]
    
    def get_embeddings(self) -> np.ndarray:
        """Get embedding matrix."""
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")
        return self.embeddings
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

