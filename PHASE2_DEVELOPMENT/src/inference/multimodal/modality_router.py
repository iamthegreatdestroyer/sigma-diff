"""
Modality Router
===============

Intelligent routing for multi-modal inputs.
Detects input modalities and routes to appropriate encoders.

Features:
- Automatic modality detection (image, text, audio, video)
- Intelligent routing to specialized encoders
- Support for mixed modality batches
- Priority-based processing for optimal throughput

Sprint 2.1 - Multi-Modal Inference
Created: 2025-12-26
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Modality(Enum):
    """Supported input modalities."""
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    UNKNOWN = auto()


@dataclass
class ModalityInput:
    """Generic input container for any modality."""
    modality: Modality
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more urgent
    request_id: Optional[str] = None


@dataclass
class RouterConfig:
    """Configuration for modality router."""
    enable_auto_detection: bool = True
    max_batch_size: int = 32
    enable_mixed_batching: bool = True
    priority_scheduling: bool = True
    timeout_seconds: float = 30.0
    supported_modalities: List[Modality] = field(
        default_factory=lambda: [Modality.TEXT, Modality.IMAGE]
    )


@dataclass
class RoutedRequest:
    """Request after routing decision."""
    inputs: List[ModalityInput]
    target_encoder: str
    modality: Modality
    batch_size: int
    requires_fusion: bool = False


@dataclass
class RouterStats:
    """Statistics for router performance."""
    total_requests: int = 0
    requests_by_modality: Dict[str, int] = field(default_factory=dict)
    average_batch_size: float = 0.0
    routing_latency_ms: float = 0.0


class ModalityDetector:
    """
    Detect modality from input data.
    
    Uses heuristics and content inspection to determine input type.
    """
    
    def __init__(self):
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    def detect(self, data: Any) -> Modality:
        """Detect modality from input data."""
        # Check if it's a tensor
        if isinstance(data, torch.Tensor):
            return self._detect_from_tensor(data)
        
        # Check if it's a string (could be text or path)
        if isinstance(data, str):
            return self._detect_from_string(data)
        
        # Check if it's a dict with type hints
        if isinstance(data, dict):
            return self._detect_from_dict(data)
        
        # Check if it's a list
        if isinstance(data, list):
            if len(data) > 0:
                return self.detect(data[0])
        
        return Modality.UNKNOWN
    
    def _detect_from_tensor(self, tensor: torch.Tensor) -> Modality:
        """Detect modality from tensor shape."""
        ndim = tensor.dim()
        
        if ndim == 1:
            # Could be text tokens or 1D audio
            return Modality.TEXT
        elif ndim == 2:
            # [seq_len, dim] - likely text embeddings
            return Modality.TEXT
        elif ndim == 3:
            # [C, H, W] - image or [B, seq, dim]
            if tensor.shape[0] in [1, 3, 4]:  # Channels
                return Modality.IMAGE
            return Modality.TEXT
        elif ndim == 4:
            # [B, C, H, W] - batched images
            return Modality.IMAGE
        elif ndim == 5:
            # [B, T, C, H, W] - video
            return Modality.VIDEO
        
        return Modality.UNKNOWN
    
    def _detect_from_string(self, text: str) -> Modality:
        """Detect modality from string (text or file path)."""
        text_lower = text.lower()
        
        # Check for file extensions
        for ext in self.image_extensions:
            if text_lower.endswith(ext):
                return Modality.IMAGE
        
        for ext in self.audio_extensions:
            if text_lower.endswith(ext):
                return Modality.AUDIO
        
        for ext in self.video_extensions:
            if text_lower.endswith(ext):
                return Modality.VIDEO
        
        # Default to text
        return Modality.TEXT
    
    def _detect_from_dict(self, data: dict) -> Modality:
        """Detect modality from dict with type hints."""
        if 'modality' in data:
            modality_str = data['modality'].lower()
            modality_map = {
                'text': Modality.TEXT,
                'image': Modality.IMAGE,
                'audio': Modality.AUDIO,
                'video': Modality.VIDEO
            }
            return modality_map.get(modality_str, Modality.UNKNOWN)
        
        if 'pixel_values' in data or 'image' in data:
            return Modality.IMAGE
        
        if 'input_ids' in data or 'text' in data:
            return Modality.TEXT
        
        if 'audio' in data or 'waveform' in data:
            return Modality.AUDIO
        
        return Modality.UNKNOWN


class EncoderRegistry:
    """
    Registry for modality-specific encoders.
    
    Maps modalities to their corresponding encoder functions.
    """
    
    def __init__(self):
        self.encoders: Dict[Modality, Callable] = {}
        self.encoder_configs: Dict[Modality, Dict[str, Any]] = {}
    
    def register(
        self,
        modality: Modality,
        encoder: Callable,
        config: Optional[Dict[str, Any]] = None
    ):
        """Register an encoder for a modality."""
        self.encoders[modality] = encoder
        self.encoder_configs[modality] = config or {}
        logger.info(f"Registered encoder for {modality.name}")
    
    def get_encoder(self, modality: Modality) -> Optional[Callable]:
        """Get encoder for a modality."""
        return self.encoders.get(modality)
    
    def get_config(self, modality: Modality) -> Dict[str, Any]:
        """Get encoder config for a modality."""
        return self.encoder_configs.get(modality, {})
    
    def has_encoder(self, modality: Modality) -> bool:
        """Check if encoder exists for modality."""
        return modality in self.encoders
    
    def list_modalities(self) -> List[Modality]:
        """List all registered modalities."""
        return list(self.encoders.keys())


class ModalityRouter:
    """
    Main router for multi-modal inputs.
    
    Handles:
    - Modality detection
    - Encoder routing
    - Batch organization
    - Priority scheduling
    """
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.detector = ModalityDetector()
        self.registry = EncoderRegistry()
        self.stats = RouterStats()
        self.pending_requests: Dict[Modality, List[ModalityInput]] = {
            m: [] for m in Modality
        }
        
        logger.info("ModalityRouter initialized")
    
    def register_encoder(
        self,
        modality: Modality,
        encoder: Callable,
        config: Optional[Dict[str, Any]] = None
    ):
        """Register an encoder for a modality."""
        self.registry.register(modality, encoder, config)
    
    def detect_modality(self, data: Any) -> Modality:
        """Detect modality of input data."""
        if not self.config.enable_auto_detection:
            return Modality.UNKNOWN
        return self.detector.detect(data)
    
    def route(
        self,
        inputs: Union[Any, List[Any]],
        modality: Optional[Modality] = None,
        priority: int = 0,
        request_id: Optional[str] = None
    ) -> List[RoutedRequest]:
        """
        Route inputs to appropriate encoders.
        
        Args:
            inputs: Single input or list of inputs
            modality: Optional modality override
            priority: Request priority (higher = more urgent)
            request_id: Optional request identifier
        
        Returns:
            List of RoutedRequest objects
        """
        # Normalize to list
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        # Detect modalities and create ModalityInputs
        modality_inputs: Dict[Modality, List[ModalityInput]] = {
            m: [] for m in Modality
        }
        
        for data in inputs:
            detected = modality if modality else self.detect_modality(data)
            
            modal_input = ModalityInput(
                modality=detected,
                data=data,
                priority=priority,
                request_id=request_id
            )
            modality_inputs[detected].append(modal_input)
        
        # Update stats
        self.stats.total_requests += len(inputs)
        for m, items in modality_inputs.items():
            if items:
                self.stats.requests_by_modality[m.name] = \
                    self.stats.requests_by_modality.get(m.name, 0) + len(items)
        
        # Create routed requests
        routed = []
        for mod, items in modality_inputs.items():
            if not items:
                continue
            
            if not self.registry.has_encoder(mod):
                logger.warning(f"No encoder registered for {mod.name}, skipping")
                continue
            
            # Batch the inputs
            batches = self._create_batches(items, self.config.max_batch_size)
            
            for batch in batches:
                routed.append(RoutedRequest(
                    inputs=batch,
                    target_encoder=f"{mod.name.lower()}_encoder",
                    modality=mod,
                    batch_size=len(batch),
                    requires_fusion=len(modality_inputs) > 1
                ))
        
        # Sort by priority if enabled
        if self.config.priority_scheduling:
            routed.sort(key=lambda r: max(i.priority for i in r.inputs), reverse=True)
        
        return routed
    
    def _create_batches(
        self,
        inputs: List[ModalityInput],
        batch_size: int
    ) -> List[List[ModalityInput]]:
        """Create batches from inputs."""
        batches = []
        for i in range(0, len(inputs), batch_size):
            batches.append(inputs[i:i + batch_size])
        return batches
    
    def process(
        self,
        inputs: Union[Any, List[Any]],
        modality: Optional[Modality] = None
    ) -> Dict[Modality, torch.Tensor]:
        """
        Process inputs through registered encoders.
        
        Args:
            inputs: Input data
            modality: Optional modality override
        
        Returns:
            Dict mapping modality to encoded features
        """
        routed = self.route(inputs, modality)
        
        results: Dict[Modality, List[torch.Tensor]] = {}
        
        for request in routed:
            encoder = self.registry.get_encoder(request.modality)
            if encoder is None:
                continue
            
            # Extract raw data from ModalityInputs
            batch_data = [inp.data for inp in request.inputs]
            
            # Process through encoder
            with torch.no_grad():
                encoded = encoder(batch_data)
            
            if request.modality not in results:
                results[request.modality] = []
            results[request.modality].append(encoded)
        
        # Concatenate results
        final_results = {}
        for mod, tensors in results.items():
            if len(tensors) == 1:
                final_results[mod] = tensors[0]
            else:
                final_results[mod] = torch.cat(tensors, dim=0)
        
        return final_results
    
    def get_stats(self) -> RouterStats:
        """Get router statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset router statistics."""
        self.stats = RouterStats()


class MultiModalRouter(ModalityRouter):
    """
    Extended router with built-in fusion support.
    
    Handles complete multi-modal processing including:
    - Encoding each modality
    - Fusing multi-modal representations
    - Returning unified output
    """
    
    def __init__(
        self,
        config: RouterConfig,
        fusion_layer: Optional[nn.Module] = None
    ):
        super().__init__(config)
        self.fusion_layer = fusion_layer
    
    def set_fusion_layer(self, fusion_layer: nn.Module):
        """Set the fusion layer for multi-modal processing."""
        self.fusion_layer = fusion_layer
    
    def process_multimodal(
        self,
        vision_input: Optional[Any] = None,
        text_input: Optional[Any] = None,
        audio_input: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Process multi-modal inputs and return fused representation.
        
        Args:
            vision_input: Image/video input
            text_input: Text input
            audio_input: Audio input
        
        Returns:
            Fused feature tensor
        """
        encoded = {}
        
        # Process each modality
        if vision_input is not None:
            result = self.process(vision_input, Modality.IMAGE)
            if Modality.IMAGE in result:
                encoded['vision'] = result[Modality.IMAGE]
        
        if text_input is not None:
            result = self.process(text_input, Modality.TEXT)
            if Modality.TEXT in result:
                encoded['text'] = result[Modality.TEXT]
        
        if audio_input is not None:
            result = self.process(audio_input, Modality.AUDIO)
            if Modality.AUDIO in result:
                encoded['audio'] = result[Modality.AUDIO]
        
        # Fuse if multiple modalities and fusion layer available
        if len(encoded) > 1 and self.fusion_layer is not None:
            if 'vision' in encoded and 'text' in encoded:
                fused = self.fusion_layer.fuse(
                    vision_features=encoded['vision'],
                    text_features=encoded['text']
                )
                return fused
        
        # Return first available encoding if no fusion
        if encoded:
            return next(iter(encoded.values()))
        
        raise ValueError("No valid inputs provided")


def create_router(
    enable_auto_detection: bool = True,
    max_batch_size: int = 32,
    **kwargs
) -> ModalityRouter:
    """
    Factory function to create a modality router.
    
    Args:
        enable_auto_detection: Enable automatic modality detection
        max_batch_size: Maximum batch size per modality
        **kwargs: Additional config options
    
    Returns:
        Configured ModalityRouter
    """
    config = RouterConfig(
        enable_auto_detection=enable_auto_detection,
        max_batch_size=max_batch_size,
        **kwargs
    )
    return ModalityRouter(config)


if __name__ == "__main__":
    # Test modality router
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Modality Router...")
    
    # Create router
    config = RouterConfig(
        enable_auto_detection=True,
        max_batch_size=8
    )
    router = ModalityRouter(config)
    
    # Register dummy encoders
    def dummy_image_encoder(images):
        batch_size = len(images) if isinstance(images, list) else 1
        return torch.randn(batch_size, 196, 768)
    
    def dummy_text_encoder(texts):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return torch.randn(batch_size, 32, 1024)
    
    router.register_encoder(Modality.IMAGE, dummy_image_encoder)
    router.register_encoder(Modality.TEXT, dummy_text_encoder)
    
    # Test routing
    test_inputs = [
        "This is a text prompt",
        "image.jpg",
        "Another text",
        "photo.png"
    ]
    
    routed = router.route(test_inputs)
    
    print(f"Number of routed requests: {len(routed)}")
    for req in routed:
        print(f"  - {req.modality.name}: {req.batch_size} items -> {req.target_encoder}")
    
    print(f"Stats: {router.get_stats()}")
    print("Modality router test passed!")
