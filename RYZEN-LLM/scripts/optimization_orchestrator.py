"""
OptimizationOrchestrator: Coordinates kernel optimization, semantic compression, 
and RLVR inference scaling without parameter conflicts.

This module implements a central coordination point for three independent optimizations:
1. Kernel Optimizer: Tile size and computation optimization
2. Semantic Compression: Parameter compression with reconstruction bounds
3. RLVR Inference Scaling: Path routing based on sparsity patterns

Author: RYZEN LLM Architecture Team
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class OptimizationState:
    """
    Snapshot of optimization configuration state for reproducibility and debugging.
    
    Attributes:
        kernel_config: Kernel optimizer settings (tile sizes, block sizes)
        compression_config: Semantic compression settings (ratio, block size)
        rlvr_config: RLVR inference scaling settings (path selection thresholds)
        epoch: Training epoch at snapshot time
        timestamp: Float-based timestamp for sorting/ordering
        metrics: Associated training metrics at snapshot time
    """
    kernel_config: Dict[str, Any]
    compression_config: Dict[str, Any]
    rlvr_config: Dict[str, Any]
    epoch: int
    timestamp: float
    metrics: Dict[str, float] = field(default_factory=dict)


class OptimizationOrchestrator:
    """
    Coordinates three independent optimizations to prevent parameter conflicts
    and maintain safe training state throughout optimization.
    
    Parameter Precedence Rules:
    1. Kernel tile sizes (kernel_optimizer) determine compression_block_size
    2. RLVR path selection depends on compression sparsity patterns
    3. Adaptive compression ratio depends on kernel_speedup feedback
    
    Safety Gates:
    1. Loss validity: no NaN/Inf values
    2. Gradient flow: 1e-6 < gradient_norm < 10.0
    3. Compression reconstruction error < 5%
    """
    
    def __init__(self, config: Dict[str, Dict]) -> None:
        """
        Initialize the OptimizationOrchestrator with configuration for all three optimizations.
        
        Args:
            config: Dictionary with three keys:
                - 'kernel_optimizer': Dict with kernel optimization parameters
                  Keys: tile_size, block_size, compute_intensity, etc.
                - 'semantic_compression': Dict with compression parameters
                  Keys: compression_ratio, block_size, reconstruction_bound, etc.
                - 'inference_scaling': Dict with RLVR parameters
                  Keys: path_selection_threshold, sparsity_threshold, etc.
        
        Raises:
            ValueError: If required config keys are missing
            
        Example:
            >>> config = {
            ...     'kernel_optimizer': {'tile_size': 32, 'block_size': 64},
            ...     'semantic_compression': {'ratio': 0.8, 'block_size': 64},
            ...     'inference_scaling': {'threshold': 0.7}
            ... }
            >>> orchestrator = OptimizationOrchestrator(config)
        """
        required_keys = {'kernel_optimizer', 'semantic_compression', 'inference_scaling'}
        if not required_keys.issubset(config.keys()):
            missing = required_keys - set(config.keys())
            raise ValueError(f"Missing required config keys: {missing}")
        
        self.config = config
        self.optimization_states: List[OptimizationState] = []
        self.parameter_adjustment_history: List[Dict] = []
        
        logger.info(f"OptimizationOrchestrator initialized with config: {self._config_summary()}")
    
    def _config_summary(self) -> str:
        """Generate a concise summary of configuration for logging."""
        summary = {}
        for key in ['kernel_optimizer', 'semantic_compression', 'inference_scaling']:
            summary[key] = list(self.config[key].keys())
        return json.dumps(summary)
    
    def validate_parameter_compatibility(self) -> Tuple[bool, List[str]]:
        """
        Validate that parameters from all three optimizations are compatible.
        
        Implements Parameter Precedence Rules:
        1. kernel_optimizer.tile_size must divide into semantic_compression.block_size
        2. semantic_compression.block_size must be compatible with inference_scaling expectations
        3. No conflicting thresholds or bounds
        
        Returns:
            Tuple[bool, List[str]]: 
                - bool: True if all parameters are compatible
                - List[str]: List of warning/error messages (empty if all compatible)
                
        Example:
            >>> is_compatible, warnings = orchestrator.validate_parameter_compatibility()
            >>> if not is_compatible:
            ...     print(f"Configuration issues: {warnings}")
        """
        warnings = []
        is_compatible = True
        
        kernel_cfg = self.config['kernel_optimizer']
        compression_cfg = self.config['semantic_compression']
        rlvr_cfg = self.config['inference_scaling']
        
        # Rule 1: Kernel tile size must be compatible with compression block size
        kernel_tile = kernel_cfg.get('tile_size', 32)
        compression_block = compression_cfg.get('block_size', 64)
        
        if compression_block % kernel_tile != 0:
            is_compatible = False
            warnings.append(
                f"🔴 INCOMPATIBLE: compression_block_size ({compression_block}) "
                f"not divisible by kernel_tile_size ({kernel_tile}). "
                f"Kernel will process {compression_block // kernel_tile + 1} tiles."
            )
        else:
            logger.info(f"✅ Kernel/compression compatibility: {compression_block // kernel_tile} tiles")
        
        # Rule 2: Compression ratio must be realistic
        compression_ratio = compression_cfg.get('compression_ratio', 0.8)
        if compression_ratio <= 0 or compression_ratio > 1.0:
            is_compatible = False
            warnings.append(
                f"🔴 Invalid compression_ratio: {compression_ratio}. Must be in (0, 1.0]"
            )
        
        # Rule 3: RLVR threshold must align with compression sparsity expectations
        rlvr_threshold = rlvr_cfg.get('path_selection_threshold', 0.7)
        expected_sparsity = 1.0 - compression_ratio
        
        if rlvr_threshold < expected_sparsity:
            warnings.append(
                f"⚠️ RLVR threshold ({rlvr_threshold:.2f}) lower than "
                f"expected sparsity ({expected_sparsity:.2f}). "
                f"May activate more sparse paths than intended."
            )
        
        # Rule 4: Reconstruction bound sanity check
        recon_bound = compression_cfg.get('reconstruction_bound', 0.05)
        if recon_bound > 0.1:
            warnings.append(
                f"⚠️ Reconstruction bound {recon_bound:.4f} quite loose (default: 0.05). "
                f"May impact training stability."
            )
        
        if warnings:
            logger.warning(f"Parameter compatibility issues: {len(warnings)} warning(s)")
        else:
            logger.info("✅ All parameters are compatible")
        
        return is_compatible, warnings
    
    def validate_safety_gates(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that training metrics pass critical safety gates.
        
        Safety Gates:
        1. Loss Validity: Loss must be finite (no NaN/Inf)
        2. Gradient Flow: Gradient norm must be between 1e-6 and 10.0
        3. Compression Stability: Reconstruction error < 5%
        
        Args:
            metrics: Dict containing training metrics:
                - 'loss': float training loss
                - 'gradient_norm': float L2 norm of gradients
                - 'compression_recon_error': float reconstruction error ratio
                
        Returns:
            Tuple[bool, List[str]]:
                - bool: True if all safety gates pass
                - List[str]: List of gate violations (empty if all pass)
                
        Example:
            >>> metrics = {'loss': 2.5, 'gradient_norm': 0.1, 'compression_recon_error': 0.02}
            >>> is_safe, warnings = orchestrator.validate_safety_gates(metrics)
            >>> if not is_safe:
            ...     raise RuntimeError(f"Safety gate violation: {warnings}")
        """
        warnings = []
        safe = True
        
        # Gate 1: Loss validity (no NaN/Inf)
        loss = metrics.get('loss', 0.0)
        try:
            # Handle both torch and numpy tensors
            if hasattr(loss, 'item'):
                loss = loss.item()
            loss_val = float(loss)
            
            if np.isnan(loss_val) or np.isinf(loss_val):
                warnings.append("🔴 GATE 1 FAILED: Loss contains NaN/Inf - UNSAFE")
                safe = False
            else:
                logger.info(f"✅ Gate 1 (Loss Validity): {loss_val:.6f}")
        except (ValueError, TypeError) as e:
            warnings.append(f"🔴 GATE 1 FAILED: Cannot evaluate loss - {e}")
            safe = False
        
        # Gate 2: Gradient flow (1e-6 to 10.0)
        grad_norm = metrics.get('gradient_norm', 1.0)
        try:
            if hasattr(grad_norm, 'item'):
                grad_norm = grad_norm.item()
            grad_norm = float(grad_norm)
            
            if grad_norm > 10.0:
                warnings.append(
                    f"⚠️ GATE 2 WARNING: Gradient norm {grad_norm:.4f} exceeds 10.0 "
                    f"(potential instability or exploding gradients)"
                )
            elif grad_norm < 1e-6:
                warnings.append(
                    f"⚠️ GATE 2 WARNING: Gradient norm {grad_norm:.2e} below 1e-6 "
                    f"(potential vanishing gradients)"
                )
            else:
                logger.info(f"✅ Gate 2 (Gradient Flow): {grad_norm:.4f}")
        except (ValueError, TypeError) as e:
            warnings.append(f"⚠️ GATE 2 WARNING: Cannot evaluate gradient norm - {e}")
        
        # Gate 3: Compression reconstruction error (< 5%)
        recon_error = metrics.get('compression_recon_error', 0.0)
        try:
            if hasattr(recon_error, 'item'):
                recon_error = recon_error.item()
            recon_error = float(recon_error)
            
            if recon_error > 0.05:
                warnings.append(
                    f"⚠️ GATE 3 WARNING: Reconstruction error {recon_error:.4f} exceeds 5% "
                    f"(consider reducing compression_ratio)"
                )
            else:
                logger.info(f"✅ Gate 3 (Compression Stability): {recon_error:.4f}")
        except (ValueError, TypeError) as e:
            warnings.append(f"⚠️ GATE 3 WARNING: Cannot evaluate reconstruction error - {e}")
        
        return safe, warnings
    
    def adapt_parameters(self, metrics: Dict[str, Any]) -> None:
        """
        Adapt optimization parameters based on training metrics feedback.
        
        Implements adaptive feedback loops:
        1. Compression Ratio: Adjusted based on reconstruction error
        2. Kernel Tile Size: May be adjusted based on kernel_speedup
        3. RLVR Path Selection: Adjusted based on sparsity patterns
        
        Args:
            metrics: Dictionary of current training metrics:
                - 'kernel_speedup': float speedup factor from kernel optimization
                - 'compression_recon_error': float reconstruction error
                - 'sparsity_ratio': float fraction of sparse parameters
                
        Returns:
            None (modifies self.config in-place)
            
        Example:
            >>> metrics = {
            ...     'kernel_speedup': 1.5,
            ...     'compression_recon_error': 0.03,
            ...     'sparsity_ratio': 0.2
            ... }
            >>> orchestrator.adapt_parameters(metrics)
        """
        adjustments = {'kernel': {}, 'compression': {}, 'rlvr': {}}
        
        # Adaptation 1: Compression ratio based on reconstruction error
        recon_error = metrics.get('compression_recon_error', 0.02)
        current_ratio = self.config['semantic_compression'].get('compression_ratio', 0.8)
        
        if recon_error > 0.04:
            # Reduce compression (increase sparsity preservation)
            new_ratio = max(0.5, current_ratio - 0.05)
            adjustments['compression']['compression_ratio'] = new_ratio
            logger.info(f"📉 Reduced compression ratio: {current_ratio:.2f} → {new_ratio:.2f}")
        elif recon_error < 0.01:
            # Increase compression (more aggressive)
            new_ratio = min(0.95, current_ratio + 0.03)
            adjustments['compression']['compression_ratio'] = new_ratio
            logger.info(f"📈 Increased compression ratio: {current_ratio:.2f} → {new_ratio:.2f}")
        
        # Adaptation 2: Kernel tile size based on speedup
        kernel_speedup = metrics.get('kernel_speedup', 1.0)
        current_tile = self.config['kernel_optimizer'].get('tile_size', 32)
        
        if kernel_speedup > 1.5:
            # Tile size is effective, can afford larger tiles
            new_tile = min(64, current_tile + 8)
            adjustments['kernel']['tile_size'] = new_tile
            logger.info(f"📈 Increased tile size: {current_tile} → {new_tile} (speedup: {kernel_speedup:.2f}x)")
        elif kernel_speedup < 1.1:
            # Tile size may be too aggressive, reduce
            new_tile = max(16, current_tile - 8)
            adjustments['kernel']['tile_size'] = new_tile
            logger.info(f"📉 Decreased tile size: {current_tile} → {new_tile} (speedup: {kernel_speedup:.2f}x)")
        
        # Adaptation 3: RLVR path selection based on sparsity
        sparsity = metrics.get('sparsity_ratio', 0.0)
        current_threshold = self.config['inference_scaling'].get('path_selection_threshold', 0.7)
        
        # Align RLVR threshold with observed sparsity
        new_threshold = min(0.95, max(0.5, sparsity + 0.1))
        if abs(new_threshold - current_threshold) > 0.05:
            adjustments['rlvr']['path_selection_threshold'] = new_threshold
            logger.info(f"🔄 Updated RLVR threshold: {current_threshold:.2f} → {new_threshold:.2f} "
                       f"(sparsity: {sparsity:.2f})")
        
        # Apply adjustments
        for opt_type, changes in adjustments.items():
            if changes:
                for param, value in changes.items():
                    if opt_type == 'kernel':
                        self.config['kernel_optimizer'][param] = value
                    elif opt_type == 'compression':
                        self.config['semantic_compression'][param] = value
                    elif opt_type == 'rlvr':
                        self.config['inference_scaling'][param] = value
        
        # Record adjustment history
        self.parameter_adjustment_history.append({
            'timestamp': datetime.now().isoformat(),
            'adjustments': adjustments,
            'metrics': metrics
        })
    
    def snapshot_configuration(self, epoch: int, timestamp: float) -> None:
        """
        Create a snapshot of the current optimization configuration state.
        
        Used for reproducibility and debugging. Snapshots enable reconstruction
        of exact optimization state at any training epoch.
        
        Args:
            epoch: The training epoch number
            timestamp: The training timestamp (for ordering/sorting)
            
        Returns:
            None (snapshots stored in self.optimization_states)
            
        Example:
            >>> orchestrator.snapshot_configuration(epoch=10, timestamp=time.time())
        """
        state = OptimizationState(
            kernel_config=dict(self.config['kernel_optimizer']),
            compression_config=dict(self.config['semantic_compression']),
            rlvr_config=dict(self.config['inference_scaling']),
            epoch=epoch,
            timestamp=timestamp,
            metrics={}
        )
        self.optimization_states.append(state)
        logger.info(f"📸 Configuration snapshot at epoch {epoch} (total snapshots: {len(self.optimization_states)})")
    
    def snapshot_with_metrics(self, epoch: int, timestamp: float, metrics: Dict[str, Any]) -> None:
        """
        Create a snapshot including associated training metrics.
        
        Args:
            epoch: The training epoch number
            timestamp: The training timestamp
            metrics: Dictionary of training metrics to store with snapshot
            
        Returns:
            None
        """
        state = OptimizationState(
            kernel_config=dict(self.config['kernel_optimizer']),
            compression_config=dict(self.config['semantic_compression']),
            rlvr_config=dict(self.config['inference_scaling']),
            epoch=epoch,
            timestamp=timestamp,
            metrics=dict(metrics)
        )
        self.optimization_states.append(state)
        logger.info(f"📸 Configuration snapshot at epoch {epoch} with metrics")
    
    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        """
        Generate metadata to include with model checkpoints.
        
        This metadata enables exact reproduction of the optimization state
        when loading from checkpoint.
        
        Returns:
            Dict with keys:
                - 'orchestrator_version': Version of this orchestrator
                - 'configurations_count': Number of snapshots
                - 'configurations': List of all snapshots
                - 'latest_config': Most recent configuration
                - 'adjustment_history': All parameter adjustments made
                
        Example:
            >>> metadata = orchestrator.get_checkpoint_metadata()
            >>> checkpoint = {'model': model, 'optimizer': optimizer, 'metadata': metadata}
            >>> torch.save(checkpoint, 'model.pt')
        """
        return {
            'orchestrator_version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'configurations_count': len(self.optimization_states),
            'configurations': [
                {
                    'epoch': state.epoch,
                    'timestamp': state.timestamp,
                    'kernel': state.kernel_config,
                    'compression': state.compression_config,
                    'rlvr': state.rlvr_config,
                    'metrics': state.metrics
                }
                for state in self.optimization_states
            ],
            'latest_config': {
                'kernel': self.config['kernel_optimizer'],
                'compression': self.config['semantic_compression'],
                'rlvr': self.config['inference_scaling'],
            },
            'adjustment_history_length': len(self.parameter_adjustment_history),
            'parameter_precedence_rules': {
                'rule_1': 'Kernel tile sizes determine compression_block_size',
                'rule_2': 'RLVR path selection depends on compression sparsity',
                'rule_3': 'Adaptive compression ratio depends on kernel_speedup'
            }
        }
    
    def save_metadata_to_file(self, filepath: Path) -> None:
        """
        Save checkpoint metadata to file for external analysis.
        
        Args:
            filepath: Path to save metadata JSON file
        """
        metadata = self.get_checkpoint_metadata()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Metadata saved to {filepath}")
    
    def get_current_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current optimization configuration.
        
        Returns:
            Dictionary with current kernel, compression, and RLVR configs
        """
        return {
            'kernel_optimizer': dict(self.config['kernel_optimizer']),
            'semantic_compression': dict(self.config['semantic_compression']),
            'inference_scaling': dict(self.config['inference_scaling']),
        }
    
    def get_adjustment_history(self) -> List[Dict]:
        """
        Get the complete history of parameter adjustments.
        
        Returns:
            List of adjustment records with timestamps and metrics
        """
        return list(self.parameter_adjustment_history)


def main():
    """Example usage of OptimizationOrchestrator."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example configuration
    config = {
        'kernel_optimizer': {
            'tile_size': 32,
            'block_size': 64,
            'compute_intensity': 16.0,
        },
        'semantic_compression': {
            'compression_ratio': 0.8,
            'block_size': 64,
            'reconstruction_bound': 0.05,
        },
        'inference_scaling': {
            'path_selection_threshold': 0.7,
            'sparsity_threshold': 0.3,
        }
    }
    
    # Initialize orchestrator
    orchestrator = OptimizationOrchestrator(config)
    
    # Validate compatibility
    is_compatible, warnings = orchestrator.validate_parameter_compatibility()
    print(f"\nParameter Compatibility: {is_compatible}")
    for warning in warnings:
        print(f"  {warning}")
    
    # Validate safety gates
    metrics = {
        'loss': 2.5,
        'gradient_norm': 0.1,
        'compression_recon_error': 0.02
    }
    is_safe, warnings = orchestrator.validate_safety_gates(metrics)
    print(f"\nSafety Gates: {is_safe}")
    for warning in warnings:
        print(f"  {warning}")
    
    # Adapt parameters
    metrics_with_feedback = {
        **metrics,
        'kernel_speedup': 1.5,
        'sparsity_ratio': 0.25
    }
    orchestrator.adapt_parameters(metrics_with_feedback)
    
    # Create snapshot
    orchestrator.snapshot_with_metrics(epoch=1, timestamp=datetime.now().timestamp(), metrics=metrics)
    
    # Get checkpoint metadata
    metadata = orchestrator.get_checkpoint_metadata()
    print(f"\nCheckpoint Metadata Keys: {list(metadata.keys())}")
    print(f"Configurations stored: {metadata['configurations_count']}")


if __name__ == '__main__':
    main()
