"""
Model Pruning Engine for Ryzanstein LLM Optimization.

Sprint 4.2: Model Optimization & Quantization
Component: Structured and Unstructured Model Pruning

This module implements pruning strategies for model size reduction:
- Structured pruning (filter/channel/head removal)
- Unstructured pruning (weight magnitude-based)
- Gradual pruning with scheduling
- Importance scoring (magnitude, gradient, Taylor expansion)

Performance Targets:
- 30-50% parameter reduction with <3% accuracy loss
- Support for iterative fine-tuning
- Hardware-aware structured pruning
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import logging
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class PruningStrategy(Enum):
    """Pruning strategy selection."""
    MAGNITUDE = "magnitude"           # Prune smallest magnitude weights
    GRADIENT = "gradient"             # Prune based on gradient information
    TAYLOR = "taylor"                 # Taylor expansion importance
    RANDOM = "random"                 # Random pruning (baseline)
    L1_NORM = "l1_norm"               # L1-norm based filter pruning
    L2_NORM = "l2_norm"               # L2-norm based filter pruning
    GEOMETRIC_MEDIAN = "geometric_median"  # Geometric median distance
    ACTIVATION = "activation"         # Activation-based importance
    HYBRID = "hybrid"                 # Combination of strategies


class PruningSchedule(Enum):
    """Gradual pruning schedule types."""
    ONE_SHOT = "one_shot"             # Prune all at once
    LINEAR = "linear"                 # Linear sparsity increase
    POLYNOMIAL = "polynomial"         # Polynomial (cubic) schedule
    EXPONENTIAL = "exponential"       # Exponential decay schedule
    COSINE = "cosine"                 # Cosine annealing schedule
    LOTTERY_TICKET = "lottery_ticket" # Iterative magnitude pruning


class PruningGranularity(Enum):
    """Pruning granularity levels."""
    WEIGHT = "weight"                 # Individual weight pruning
    ROW = "row"                       # Row-wise pruning (output neurons)
    COLUMN = "column"                 # Column-wise pruning (input neurons)
    FILTER = "filter"                 # Filter/channel pruning (conv)
    HEAD = "head"                     # Attention head pruning
    LAYER = "layer"                   # Entire layer pruning
    BLOCK = "block"                   # Transformer block pruning


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PruningConfig:
    """Configuration for model pruning."""
    target_sparsity: float = 0.5              # Target sparsity ratio
    strategy: PruningStrategy = PruningStrategy.MAGNITUDE
    schedule: PruningSchedule = PruningSchedule.ONE_SHOT
    granularity: PruningGranularity = PruningGranularity.WEIGHT
    
    # Schedule parameters
    num_pruning_steps: int = 10               # Steps for gradual pruning
    initial_sparsity: float = 0.0             # Starting sparsity
    polynomial_degree: int = 3                # For polynomial schedule
    
    # Layer selection
    prune_embeddings: bool = False            # Prune embedding layers
    prune_attention: bool = True              # Prune attention weights
    prune_mlp: bool = True                    # Prune MLP/FFN layers
    prune_output: bool = False                # Prune output projection
    layer_wise_sparsity: Optional[Dict[str, float]] = None
    
    # Importance scoring
    importance_accumulation: str = "mean"     # mean, sum, max
    global_pruning: bool = True               # Global vs per-layer thresholds
    
    # Fine-tuning
    rewinding_epoch: Optional[int] = None     # For lottery ticket
    
    def __post_init__(self):
        if not 0.0 <= self.target_sparsity < 1.0:
            raise ValueError(f"target_sparsity must be in [0, 1): {self.target_sparsity}")
        if not 0.0 <= self.initial_sparsity <= self.target_sparsity:
            raise ValueError(f"initial_sparsity must be <= target_sparsity")


@dataclass
class PruningMetrics:
    """Metrics from pruning operation."""
    original_params: int = 0
    pruned_params: int = 0
    remaining_params: int = 0
    achieved_sparsity: float = 0.0
    target_sparsity: float = 0.0
    
    # Per-layer breakdown
    layer_sparsities: Dict[str, float] = field(default_factory=dict)
    layer_param_counts: Dict[str, int] = field(default_factory=dict)
    
    # Performance impact estimates
    estimated_speedup: float = 1.0
    estimated_memory_reduction: float = 0.0
    
    # Accuracy tracking
    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None
    accuracy_drop: Optional[float] = None


@dataclass
class PruningResult:
    """Result of model pruning."""
    model: nn.Module
    masks: Dict[str, Tensor]
    metrics: PruningMetrics
    config: PruningConfig
    pruning_step: int = 0
    is_final: bool = True
    
    # Original weights for rewinding
    original_weights: Optional[Dict[str, Tensor]] = None


@dataclass
class PruningMask:
    """Pruning mask for a single parameter."""
    name: str
    mask: Tensor                      # Binary mask (1=keep, 0=prune)
    importance_scores: Optional[Tensor] = None
    sparsity: float = 0.0
    num_pruned: int = 0
    num_total: int = 0


# =============================================================================
# Importance Scoring Functions
# =============================================================================


def compute_magnitude_importance(weight: Tensor) -> Tensor:
    """Compute importance based on weight magnitude."""
    return weight.abs()


def compute_l1_norm_importance(weight: Tensor, dim: int = 0) -> Tensor:
    """Compute L1-norm importance along specified dimension."""
    return weight.abs().sum(dim=dim)


def compute_l2_norm_importance(weight: Tensor, dim: int = 0) -> Tensor:
    """Compute L2-norm importance along specified dimension."""
    return (weight ** 2).sum(dim=dim).sqrt()


def compute_gradient_importance(
    weight: Tensor,
    gradient: Optional[Tensor] = None
) -> Tensor:
    """Compute importance based on gradient magnitude."""
    if gradient is None:
        if weight.grad is not None:
            gradient = weight.grad
        else:
            logger.warning("No gradient available, falling back to magnitude")
            return compute_magnitude_importance(weight)
    return gradient.abs()


def compute_taylor_importance(
    weight: Tensor,
    gradient: Optional[Tensor] = None
) -> Tensor:
    """
    Compute Taylor expansion importance: |w * dL/dw|
    
    This approximates the impact of removing a weight on the loss.
    """
    if gradient is None:
        if weight.grad is not None:
            gradient = weight.grad
        else:
            logger.warning("No gradient for Taylor, falling back to magnitude")
            return compute_magnitude_importance(weight)
    return (weight * gradient).abs()


def compute_geometric_median_importance(
    weight: Tensor,
    dim: int = 0,
    num_iterations: int = 50
) -> Tensor:
    """
    Compute geometric median distance for filter pruning.
    
    Filters similar to the geometric median are pruned as redundant.
    """
    if weight.dim() < 2:
        return compute_magnitude_importance(weight)
    
    # Reshape for filter-wise computation
    if dim == 0:
        filters = weight.view(weight.size(0), -1)
    else:
        filters = weight.view(-1, weight.size(-1))
    
    # Weiszfeld algorithm for geometric median
    median = filters.mean(dim=0)
    
    for _ in range(num_iterations):
        distances = (filters - median.unsqueeze(0)).norm(dim=1, keepdim=True)
        distances = distances.clamp(min=1e-8)
        weights = 1.0 / distances
        weights = weights / weights.sum()
        median = (filters * weights).sum(dim=0)
    
    # Distance from geometric median (higher = more unique = more important)
    importance = (filters - median.unsqueeze(0)).norm(dim=1)
    return importance


# =============================================================================
# Pruning Schedule Functions
# =============================================================================


def compute_sparsity_at_step(
    step: int,
    total_steps: int,
    initial_sparsity: float,
    target_sparsity: float,
    schedule: PruningSchedule,
    polynomial_degree: int = 3
) -> float:
    """Compute sparsity for current step given schedule."""
    if total_steps <= 1:
        return target_sparsity
    
    t = min(step / (total_steps - 1), 1.0)
    delta = target_sparsity - initial_sparsity
    
    if schedule == PruningSchedule.ONE_SHOT:
        return target_sparsity if step >= total_steps - 1 else initial_sparsity
    
    elif schedule == PruningSchedule.LINEAR:
        return initial_sparsity + delta * t
    
    elif schedule == PruningSchedule.POLYNOMIAL:
        return target_sparsity - delta * ((1 - t) ** polynomial_degree)
    
    elif schedule == PruningSchedule.EXPONENTIAL:
        if t == 0:
            return initial_sparsity
        return target_sparsity - delta * math.exp(-5 * t)
    
    elif schedule == PruningSchedule.COSINE:
        return initial_sparsity + delta * (1 - math.cos(math.pi * t)) / 2
    
    elif schedule == PruningSchedule.LOTTERY_TICKET:
        # Iterative magnitude pruning with rewind
        return initial_sparsity + delta * t
    
    else:
        return target_sparsity


# =============================================================================
# Mask Generation
# =============================================================================


def generate_unstructured_mask(
    weight: Tensor,
    sparsity: float,
    importance_fn: Callable[[Tensor], Tensor],
    global_threshold: Optional[float] = None
) -> PruningMask:
    """Generate unstructured (weight-level) pruning mask."""
    importance = importance_fn(weight)
    num_total = weight.numel()
    num_to_prune = int(num_total * sparsity)
    
    if global_threshold is not None:
        mask = (importance > global_threshold).float()
    else:
        if num_to_prune == 0:
            mask = torch.ones_like(weight)
        elif num_to_prune >= num_total:
            mask = torch.zeros_like(weight)
        else:
            threshold = importance.view(-1).kthvalue(num_to_prune).values
            mask = (importance > threshold).float()
    
    num_pruned = (mask == 0).sum().item()
    achieved_sparsity = num_pruned / num_total if num_total > 0 else 0.0
    
    return PruningMask(
        name="",
        mask=mask,
        importance_scores=importance,
        sparsity=achieved_sparsity,
        num_pruned=num_pruned,
        num_total=num_total
    )


def generate_structured_mask(
    weight: Tensor,
    sparsity: float,
    granularity: PruningGranularity,
    importance_fn: Callable[[Tensor], Tensor]
) -> PruningMask:
    """Generate structured pruning mask (rows, columns, filters)."""
    if granularity == PruningGranularity.WEIGHT:
        return generate_unstructured_mask(weight, sparsity, importance_fn)
    
    # Determine dimension and compute importance
    if granularity == PruningGranularity.ROW:
        dim = 0
        importance = compute_l2_norm_importance(weight, dim=1)  # Sum over input
    elif granularity == PruningGranularity.COLUMN:
        dim = 1
        importance = compute_l2_norm_importance(weight, dim=0)  # Sum over output
    elif granularity == PruningGranularity.FILTER:
        dim = 0
        if weight.dim() >= 4:  # Conv weight
            importance = compute_l2_norm_importance(
                weight.view(weight.size(0), -1), dim=1
            )
        else:
            importance = compute_l2_norm_importance(weight, dim=1)
    else:
        # Default to weight-level
        return generate_unstructured_mask(weight, sparsity, importance_fn)
    
    num_elements = importance.numel()
    num_to_prune = int(num_elements * sparsity)
    
    if num_to_prune == 0:
        element_mask = torch.ones_like(importance)
    elif num_to_prune >= num_elements:
        element_mask = torch.zeros_like(importance)
    else:
        threshold = importance.kthvalue(num_to_prune).values
        element_mask = (importance > threshold).float()
    
    # Expand mask to full weight shape
    if dim == 0:
        mask = element_mask.view(-1, *([1] * (weight.dim() - 1))).expand_as(weight)
    else:
        mask = element_mask.view(*([1] * dim), -1, *([1] * (weight.dim() - dim - 1)))
        mask = mask.expand_as(weight)
    
    num_pruned = (mask == 0).sum().item()
    num_total = weight.numel()
    achieved_sparsity = num_pruned / num_total if num_total > 0 else 0.0
    
    return PruningMask(
        name="",
        mask=mask,
        importance_scores=importance,
        sparsity=achieved_sparsity,
        num_pruned=num_pruned,
        num_total=num_total
    )


# =============================================================================
# Abstract Base Pruner
# =============================================================================


class ModelPruner(ABC):
    """Abstract base class for model pruners."""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.masks: Dict[str, PruningMask] = {}
        self.original_weights: Optional[Dict[str, Tensor]] = None
        self.current_step = 0
        self.importance_accumulator: Dict[str, List[Tensor]] = {}
        
    @abstractmethod
    def compute_importance(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None
    ) -> Dict[str, Tensor]:
        """Compute importance scores for all parameters."""
        pass
    
    @abstractmethod
    def generate_masks(
        self,
        model: nn.Module,
        importance_scores: Dict[str, Tensor]
    ) -> Dict[str, PruningMask]:
        """Generate pruning masks based on importance scores."""
        pass
    
    def prune(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None,
        step: Optional[int] = None
    ) -> PruningResult:
        """Execute pruning on the model."""
        if step is not None:
            self.current_step = step
        
        # Save original weights for lottery ticket
        if self.original_weights is None and self.config.schedule == PruningSchedule.LOTTERY_TICKET:
            self.original_weights = {
                name: param.data.clone()
                for name, param in model.named_parameters()
            }
        
        # Compute current sparsity target
        current_sparsity = compute_sparsity_at_step(
            self.current_step,
            self.config.num_pruning_steps,
            self.config.initial_sparsity,
            self.config.target_sparsity,
            self.config.schedule,
            self.config.polynomial_degree
        )
        
        # Compute importance and generate masks
        importance_scores = self.compute_importance(model, dataloader)
        self.masks = self.generate_masks(model, importance_scores)
        
        # Apply masks
        metrics = self._apply_masks(model, current_sparsity)
        
        self.current_step += 1
        is_final = self.current_step >= self.config.num_pruning_steps
        
        return PruningResult(
            model=model,
            masks={name: m.mask for name, m in self.masks.items()},
            metrics=metrics,
            config=self.config,
            pruning_step=self.current_step,
            is_final=is_final,
            original_weights=self.original_weights
        )
    
    def _apply_masks(
        self,
        model: nn.Module,
        target_sparsity: float
    ) -> PruningMetrics:
        """Apply pruning masks to model parameters."""
        original_params = 0
        pruned_params = 0
        layer_sparsities = {}
        layer_param_counts = {}
        
        for name, param in model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                param.data.mul_(mask.mask.to(param.device))
                
                original_params += mask.num_total
                pruned_params += mask.num_pruned
                layer_sparsities[name] = mask.sparsity
                layer_param_counts[name] = mask.num_total
        
        remaining_params = original_params - pruned_params
        achieved_sparsity = pruned_params / original_params if original_params > 0 else 0.0
        
        # Estimate performance impact
        # Structured pruning: linear speedup potential
        # Unstructured: depends on hardware sparse support
        if self.config.granularity in [PruningGranularity.FILTER, PruningGranularity.ROW]:
            estimated_speedup = 1.0 / (1.0 - achieved_sparsity + 1e-8)
        else:
            estimated_speedup = 1.0 + achieved_sparsity * 0.3  # Conservative for unstructured
        
        return PruningMetrics(
            original_params=original_params,
            pruned_params=pruned_params,
            remaining_params=remaining_params,
            achieved_sparsity=achieved_sparsity,
            target_sparsity=target_sparsity,
            layer_sparsities=layer_sparsities,
            layer_param_counts=layer_param_counts,
            estimated_speedup=min(estimated_speedup, 4.0),
            estimated_memory_reduction=achieved_sparsity
        )
    
    def should_prune_layer(self, name: str) -> bool:
        """Check if a layer should be pruned based on config."""
        name_lower = name.lower()
        
        # Embedding layers
        if "embed" in name_lower:
            return self.config.prune_embeddings
        
        # Attention layers
        if any(x in name_lower for x in ["attn", "attention", "query", "key", "value"]):
            return self.config.prune_attention
        
        # MLP/FFN layers
        if any(x in name_lower for x in ["mlp", "ffn", "fc", "linear", "dense"]):
            return self.config.prune_mlp
        
        # Output projection
        if any(x in name_lower for x in ["output", "proj", "head", "lm_head"]):
            return self.config.prune_output
        
        return True  # Default: prune
    
    def rewind_weights(self, model: nn.Module) -> None:
        """Rewind weights to initial values (lottery ticket hypothesis)."""
        if self.original_weights is None:
            logger.warning("No original weights saved for rewinding")
            return
        
        for name, param in model.named_parameters():
            if name in self.original_weights:
                param.data.copy_(self.original_weights[name])
                if name in self.masks:
                    param.data.mul_(self.masks[name].mask.to(param.device))


# =============================================================================
# Concrete Pruner Implementations
# =============================================================================


class UnstructuredPruner(ModelPruner):
    """Unstructured (weight-level) magnitude pruning."""
    
    def compute_importance(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None
    ) -> Dict[str, Tensor]:
        """Compute importance based on weight magnitudes."""
        importance_scores = {}
        
        for name, param in model.named_parameters():
            if not self.should_prune_layer(name):
                continue
            if param.dim() < 2:  # Skip biases and norms
                continue
            
            if self.config.strategy == PruningStrategy.MAGNITUDE:
                importance = compute_magnitude_importance(param.data)
            elif self.config.strategy == PruningStrategy.GRADIENT:
                importance = compute_gradient_importance(param.data, param.grad)
            elif self.config.strategy == PruningStrategy.TAYLOR:
                importance = compute_taylor_importance(param.data, param.grad)
            elif self.config.strategy == PruningStrategy.RANDOM:
                importance = torch.rand_like(param.data)
            else:
                importance = compute_magnitude_importance(param.data)
            
            importance_scores[name] = importance
        
        return importance_scores
    
    def generate_masks(
        self,
        model: nn.Module,
        importance_scores: Dict[str, Tensor]
    ) -> Dict[str, PruningMask]:
        """Generate unstructured masks based on importance."""
        masks = {}
        
        # Compute current target sparsity
        current_sparsity = compute_sparsity_at_step(
            self.current_step,
            self.config.num_pruning_steps,
            self.config.initial_sparsity,
            self.config.target_sparsity,
            self.config.schedule,
            self.config.polynomial_degree
        )
        
        if self.config.global_pruning:
            # Global threshold across all layers
            all_importance = torch.cat([
                imp.view(-1) for imp in importance_scores.values()
            ])
            total_params = all_importance.numel()
            num_to_prune = int(total_params * current_sparsity)
            
            if num_to_prune > 0 and num_to_prune < total_params:
                global_threshold = all_importance.kthvalue(num_to_prune).values.item()
            else:
                global_threshold = 0.0
            
            for name, importance in importance_scores.items():
                param = dict(model.named_parameters())[name]
                layer_sparsity = self.config.layer_wise_sparsity.get(name, current_sparsity) \
                    if self.config.layer_wise_sparsity else current_sparsity
                
                mask = generate_unstructured_mask(
                    param.data,
                    layer_sparsity,
                    lambda w: importance,
                    global_threshold if self.config.global_pruning else None
                )
                mask.name = name
                masks[name] = mask
        else:
            # Per-layer thresholds
            for name, importance in importance_scores.items():
                param = dict(model.named_parameters())[name]
                layer_sparsity = self.config.layer_wise_sparsity.get(name, current_sparsity) \
                    if self.config.layer_wise_sparsity else current_sparsity
                
                mask = generate_unstructured_mask(
                    param.data,
                    layer_sparsity,
                    lambda w: importance
                )
                mask.name = name
                masks[name] = mask
        
        return masks


class StructuredPruner(ModelPruner):
    """Structured pruning (filters, channels, heads)."""
    
    def compute_importance(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None
    ) -> Dict[str, Tensor]:
        """Compute structured importance scores."""
        importance_scores = {}
        
        for name, param in model.named_parameters():
            if not self.should_prune_layer(name):
                continue
            if param.dim() < 2:
                continue
            
            if self.config.strategy == PruningStrategy.L1_NORM:
                importance = compute_l1_norm_importance(param.data, dim=1)
            elif self.config.strategy == PruningStrategy.L2_NORM:
                importance = compute_l2_norm_importance(param.data, dim=1)
            elif self.config.strategy == PruningStrategy.GEOMETRIC_MEDIAN:
                importance = compute_geometric_median_importance(param.data)
            else:
                importance = compute_l2_norm_importance(param.data, dim=1)
            
            importance_scores[name] = importance
        
        return importance_scores
    
    def generate_masks(
        self,
        model: nn.Module,
        importance_scores: Dict[str, Tensor]
    ) -> Dict[str, PruningMask]:
        """Generate structured masks."""
        masks = {}
        
        current_sparsity = compute_sparsity_at_step(
            self.current_step,
            self.config.num_pruning_steps,
            self.config.initial_sparsity,
            self.config.target_sparsity,
            self.config.schedule,
            self.config.polynomial_degree
        )
        
        for name, importance in importance_scores.items():
            param = dict(model.named_parameters())[name]
            layer_sparsity = self.config.layer_wise_sparsity.get(name, current_sparsity) \
                if self.config.layer_wise_sparsity else current_sparsity
            
            mask = generate_structured_mask(
                param.data,
                layer_sparsity,
                self.config.granularity,
                lambda w: importance
            )
            mask.name = name
            masks[name] = mask
        
        return masks


class GradualPruner(ModelPruner):
    """Gradual pruning with scheduling and optional fine-tuning."""
    
    def __init__(self, config: PruningConfig):
        super().__init__(config)
        self.fine_tune_callback: Optional[Callable] = None
        
    def set_fine_tune_callback(self, callback: Callable[[nn.Module, int], None]):
        """Set callback for fine-tuning between pruning steps."""
        self.fine_tune_callback = callback
    
    def compute_importance(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None
    ) -> Dict[str, Tensor]:
        """Compute importance with accumulation over iterations."""
        importance_scores = {}
        
        for name, param in model.named_parameters():
            if not self.should_prune_layer(name):
                continue
            if param.dim() < 2:
                continue
            
            current_importance = self._compute_single_importance(param)
            
            # Accumulate importance over pruning steps
            if name not in self.importance_accumulator:
                self.importance_accumulator[name] = []
            self.importance_accumulator[name].append(current_importance)
            
            # Combine accumulated scores
            if self.config.importance_accumulation == "mean":
                importance = torch.stack(self.importance_accumulator[name]).mean(dim=0)
            elif self.config.importance_accumulation == "sum":
                importance = torch.stack(self.importance_accumulator[name]).sum(dim=0)
            elif self.config.importance_accumulation == "max":
                importance = torch.stack(self.importance_accumulator[name]).max(dim=0).values
            else:
                importance = current_importance
            
            importance_scores[name] = importance
        
        return importance_scores
    
    def _compute_single_importance(self, param: nn.Parameter) -> Tensor:
        """Compute importance for a single parameter."""
        if self.config.strategy == PruningStrategy.MAGNITUDE:
            return compute_magnitude_importance(param.data)
        elif self.config.strategy == PruningStrategy.GRADIENT:
            return compute_gradient_importance(param.data, param.grad)
        elif self.config.strategy == PruningStrategy.TAYLOR:
            return compute_taylor_importance(param.data, param.grad)
        else:
            return compute_magnitude_importance(param.data)
    
    def generate_masks(
        self,
        model: nn.Module,
        importance_scores: Dict[str, Tensor]
    ) -> Dict[str, PruningMask]:
        """Generate masks for current pruning step."""
        masks = {}
        
        current_sparsity = compute_sparsity_at_step(
            self.current_step,
            self.config.num_pruning_steps,
            self.config.initial_sparsity,
            self.config.target_sparsity,
            self.config.schedule,
            self.config.polynomial_degree
        )
        
        logger.info(f"Pruning step {self.current_step}/{self.config.num_pruning_steps}, "
                   f"target sparsity: {current_sparsity:.2%}")
        
        for name, importance in importance_scores.items():
            param = dict(model.named_parameters())[name]
            layer_sparsity = self.config.layer_wise_sparsity.get(name, current_sparsity) \
                if self.config.layer_wise_sparsity else current_sparsity
            
            if self.config.granularity == PruningGranularity.WEIGHT:
                mask = generate_unstructured_mask(
                    param.data,
                    layer_sparsity,
                    lambda w: importance
                )
            else:
                mask = generate_structured_mask(
                    param.data,
                    layer_sparsity,
                    self.config.granularity,
                    lambda w: importance
                )
            
            mask.name = name
            masks[name] = mask
        
        return masks
    
    def prune_with_fine_tuning(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None,
        fine_tune_epochs: int = 1
    ) -> List[PruningResult]:
        """Execute gradual pruning with fine-tuning between steps."""
        results = []
        
        for step in range(self.config.num_pruning_steps):
            # Prune
            result = self.prune(model, dataloader, step)
            results.append(result)
            
            logger.info(f"Step {step+1}: Achieved sparsity {result.metrics.achieved_sparsity:.2%}")
            
            # Fine-tune if not final step
            if not result.is_final and self.fine_tune_callback is not None:
                self.fine_tune_callback(model, fine_tune_epochs)
        
        return results


# =============================================================================
# Factory Function
# =============================================================================


def create_pruner(config: PruningConfig) -> ModelPruner:
    """
    Factory function to create appropriate pruner.
    
    Args:
        config: Pruning configuration
        
    Returns:
        Configured ModelPruner instance
    """
    # Select based on schedule and granularity
    if config.schedule != PruningSchedule.ONE_SHOT:
        logger.info(f"Creating GradualPruner with {config.schedule.value} schedule")
        return GradualPruner(config)
    
    if config.granularity == PruningGranularity.WEIGHT:
        logger.info(f"Creating UnstructuredPruner with {config.strategy.value} strategy")
        return UnstructuredPruner(config)
    
    logger.info(f"Creating StructuredPruner with {config.granularity.value} granularity")
    return StructuredPruner(config)


# =============================================================================
# Utility Functions
# =============================================================================


def compute_model_sparsity(model: nn.Module) -> Dict[str, float]:
    """Compute current sparsity of each layer in the model."""
    sparsities = {}
    total_zeros = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        
        zeros = (param.data == 0).sum().item()
        total = param.numel()
        
        sparsities[name] = zeros / total if total > 0 else 0.0
        total_zeros += zeros
        total_params += total
    
    sparsities["__total__"] = total_zeros / total_params if total_params > 0 else 0.0
    return sparsities


def remove_pruned_neurons(
    model: nn.Module,
    masks: Dict[str, Tensor]
) -> nn.Module:
    """
    Physically remove pruned neurons for structured pruning.
    
    This creates a smaller model by removing zero channels/filters.
    Only applicable for structured pruning.
    """
    # This is a complex operation that requires careful handling
    # of layer dependencies. For now, return model as-is.
    logger.warning("Physical neuron removal not yet implemented, returning masked model")
    return model


__all__ = [
    # Enums
    "PruningStrategy",
    "PruningSchedule",
    "PruningGranularity",
    # Data classes
    "PruningConfig",
    "PruningMetrics",
    "PruningResult",
    "PruningMask",
    # Pruners
    "ModelPruner",
    "UnstructuredPruner",
    "StructuredPruner",
    "GradualPruner",
    # Factory
    "create_pruner",
    # Utilities
    "compute_model_sparsity",
    "remove_pruned_neurons",
]
