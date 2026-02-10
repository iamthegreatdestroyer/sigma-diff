"""
Multi-Modal Inference Module
============================

Production-ready multi-modal inference for vision-language models.

Components:
- VisionEncoder: CLIP, DINOv2, ViT vision encoders
- CrossModalFusionLayer: Attention-based modality fusion
- ModalityRouter: Intelligent input routing
- AdaptiveBatcher: Dynamic batching optimization
- MultiModalPipeline: Unified inference pipeline

Sprint 2.1 - Multi-Modal Inference
Created: 2025-12-26
"""

from .vision_encoder import (
    VisionEncoder,
    VisionEncoderConfig,
    VisionEncoderType,
    VisionOutput,
    ImageInput,
    create_vision_encoder
)

from .fusion_layer import (
    CrossModalFusionLayer,
    FusionConfig,
    FusionStrategy,
    FusionInput,
    FusionOutput,
    create_fusion_layer
)

from .modality_router import (
    ModalityRouter,
    MultiModalRouter,
    RouterConfig,
    Modality,
    ModalityInput,
    RoutedRequest,
    create_router
)

from .adaptive_batcher import (
    AdaptiveBatcher,
    ContinuousBatcher,
    BatcherConfig,
    Batch,
    BatchRequest,
    create_batcher
)

from .pipeline import (
    MultiModalPipeline,
    PipelineConfig,
    MultiModalInput,
    MultiModalOutput,
    create_pipeline
)

__all__ = [
    # Vision Encoder
    "VisionEncoder",
    "VisionEncoderConfig",
    "VisionEncoderType",
    "VisionOutput",
    "ImageInput",
    "create_vision_encoder",
    
    # Fusion Layer
    "CrossModalFusionLayer",
    "FusionConfig",
    "FusionStrategy",
    "FusionInput",
    "FusionOutput",
    "create_fusion_layer",
    
    # Modality Router
    "ModalityRouter",
    "MultiModalRouter",
    "RouterConfig",
    "Modality",
    "ModalityInput",
    "RoutedRequest",
    "create_router",
    
    # Adaptive Batcher
    "AdaptiveBatcher",
    "ContinuousBatcher",
    "BatcherConfig",
    "Batch",
    "BatchRequest",
    "create_batcher",
    
    # Pipeline
    "MultiModalPipeline",
    "PipelineConfig",
    "MultiModalInput",
    "MultiModalOutput",
    "create_pipeline",
]

__version__ = "2.1.0"
__sprint__ = "Sprint 2.1 - Multi-Modal Inference"
