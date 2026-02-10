"""
Unit Tests for Distributed Model Loading

Task 1.1.7: Comprehensive test coverage for:
  - DistributedCheckpointLoader loading and metadata
  - WeightDistributor sharding strategies
  - CheckpointSaver checkpoint saving
  - ModelDistributor orchestration
  - Integration scenarios

Test Categories:
  1. Metadata Management: Loading and broadcasting
  2. Weight Distribution: Row/column sharding
  3. Checkpoint I/O: Loading and saving
  4. Orchestration: End-to-end loading
  5. Integration: Multiple components together
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

# Import model loader components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.distributed.model_loader import (
    ModelLoadConfig,
    CheckpointMetadata,
    DistributedCheckpointLoader,
    WeightDistributor,
    CheckpointSaver,
    ModelDistributor,
)


# ============================================================================
# Test Utilities
# ============================================================================

class BaseModelLoaderTest(unittest.TestCase):
    """Base class for model loader tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# ============================================================================
# Configuration & Metadata Tests
# ============================================================================

class TestModelLoadConfig(BaseModelLoaderTest):
    """Test ModelLoadConfig configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelLoadConfig()
        
        self.assertEqual(config.checkpoint_format, "distributed")
        self.assertTrue(config.use_memory_map)
        self.assertTrue(config.enable_prefetch)
        self.assertEqual(config.tp_size, 1)
        self.assertEqual(config.num_load_threads, 4)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelLoadConfig(
            checkpoint_dir="/custom/path",
            tp_size=4,
            enable_prefetch=False
        )
        
        self.assertEqual(config.checkpoint_dir, "/custom/path")
        self.assertEqual(config.tp_size, 4)
        self.assertFalse(config.enable_prefetch)


class TestCheckpointMetadata(BaseModelLoaderTest):
    """Test CheckpointMetadata."""
    
    def test_initialization(self):
        """Test metadata initialization."""
        metadata = CheckpointMetadata()
        
        self.assertEqual(metadata.model_name, "")
        self.assertEqual(metadata.model_size, 0)
        self.assertEqual(metadata.step, 0)
    
    def test_dict_conversion(self):
        """Test conversion to/from dictionary."""
        metadata = CheckpointMetadata()
        metadata.model_name = "llama-7b"
        metadata.model_size = 7_000_000_000
        metadata.tp_size = 4
        
        d = metadata.to_dict()
        
        self.assertIn("model_name", d)
        self.assertEqual(d["model_name"], "llama-7b")
        self.assertEqual(d["model_size"], 7_000_000_000)
    
    def test_from_dict(self):
        """Test creating metadata from dictionary."""
        d = {
            "model_name": "llama-7b",
            "model_size": 7_000_000_000,
            "hidden_dim": 4096,
            "num_layers": 32,
            "tp_size": 4,
        }
        
        metadata = CheckpointMetadata.from_dict(d)
        
        self.assertEqual(metadata.model_name, "llama-7b")
        self.assertEqual(metadata.model_size, 7_000_000_000)
        self.assertEqual(metadata.tp_size, 4)


# ============================================================================
# DistributedCheckpointLoader Tests
# ============================================================================

class TestDistributedCheckpointLoader(BaseModelLoaderTest):
    """Test DistributedCheckpointLoader."""
    
    def test_initialization(self):
        """Test loader initialization."""
        loader = DistributedCheckpointLoader(
            checkpoint_dir=self.temp_dir,
            rank=0,
            world_size=4
        )
        
        self.assertEqual(loader.rank, 0)
        self.assertEqual(loader.world_size, 4)
        self.assertTrue(loader.use_memory_map)
        self.assertTrue(loader.enable_prefetch)
    
    def test_find_latest_checkpoint_empty(self):
        """Test finding checkpoint in empty directory."""
        loader = DistributedCheckpointLoader(
            checkpoint_dir=self.temp_dir,
            rank=0,
            world_size=1
        )
        
        result = loader.find_latest_checkpoint()
        
        self.assertIsNone(result)
    
    def test_find_latest_checkpoint_multiple(self):
        """Test finding latest checkpoint among multiple."""
        # Create checkpoint directories
        for step in [100, 200, 300]:
            os.makedirs(os.path.join(self.temp_dir, f"model-step-{step}"))
        
        loader = DistributedCheckpointLoader(
            checkpoint_dir=self.temp_dir,
            rank=0,
            world_size=1
        )
        
        result = loader.find_latest_checkpoint()
        
        self.assertIsNotNone(result)
        self.assertIn("300", str(result))
    
    def test_save_and_load_metadata(self):
        """Test saving and loading metadata."""
        # Create checkpoint directory
        ckpt_dir = os.path.join(self.temp_dir, "model-step-1000")
        os.makedirs(ckpt_dir)
        
        # Save metadata
        metadata = CheckpointMetadata()
        metadata.model_name = "llama-7b"
        metadata.model_size = 7_000_000_000
        metadata.num_layers = 32
        
        metadata_file = os.path.join(ckpt_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f)
        
        # Load metadata
        loader = DistributedCheckpointLoader(
            checkpoint_dir=self.temp_dir,
            rank=0,
            world_size=1
        )
        
        loaded = loader.load_metadata(Path(ckpt_dir))
        
        self.assertEqual(loaded.model_name, "llama-7b")
        self.assertEqual(loaded.model_size, 7_000_000_000)
        self.assertEqual(loaded.num_layers, 32)
    
    def test_load_rank_weights(self):
        """Test loading rank-specific weights."""
        # Create checkpoint directory with weight file
        ckpt_dir = os.path.join(self.temp_dir, "model-step-1000")
        os.makedirs(ckpt_dir)
        
        # Create and save weights for rank 0
        weights = {
            "layer1.weight": torch.randn(1024, 4096),
            "layer1.bias": torch.randn(1024),
        }
        
        weight_file = os.path.join(ckpt_dir, "weights_rank0.pt")
        torch.save(weights, weight_file)
        
        # Load weights
        loader = DistributedCheckpointLoader(
            checkpoint_dir=self.temp_dir,
            rank=0,
            world_size=1
        )
        
        loaded = loader.load_rank_weights(Path(ckpt_dir))
        
        self.assertIn("layer1.weight", loaded)
        self.assertIn("layer1.bias", loaded)
        self.assertEqual(loaded["layer1.weight"].shape, (1024, 4096))


# ============================================================================
# WeightDistributor Tests
# ============================================================================

class TestWeightDistributor(BaseModelLoaderTest):
    """Test WeightDistributor weight sharding."""
    
    def test_initialization(self):
        """Test distributor initialization."""
        distributor = WeightDistributor(rank=0, world_size=4, tp_size=4)
        
        self.assertEqual(distributor.rank, 0)
        self.assertEqual(distributor.world_size, 4)
        self.assertEqual(distributor.tp_size, 4)
    
    def test_tp_size_validation(self):
        """Test TP size validation."""
        with self.assertRaises(ValueError):
            WeightDistributor(rank=0, world_size=2, tp_size=4)
    
    def test_shard_row_wise(self):
        """Test row-wise weight sharding."""
        distributor = WeightDistributor(rank=0, world_size=4, tp_size=4)
        
        # Create weight matrix (4096, 4096)
        weight = torch.randn(4096, 4096)
        
        sharded, local_out = distributor.shard_row_wise(weight)
        
        # Each rank gets 1/4 of output dimension
        self.assertEqual(sharded.shape, (1024, 4096))
        self.assertEqual(local_out, 1024)
    
    def test_shard_row_wise_different_rank(self):
        """Test row-wise sharding for different ranks."""
        weight = torch.randn(4096, 4096)
        
        for rank in range(4):
            distributor = WeightDistributor(rank=rank, world_size=4, tp_size=4)
            sharded, local_out = distributor.shard_row_wise(weight)
            
            # Verify non-overlapping shards
            self.assertEqual(sharded.shape, (1024, 4096))
            start_idx = rank * 1024
            end_idx = (rank + 1) * 1024
            
            expected = weight[start_idx:end_idx, :]
            self.assertTrue(torch.allclose(sharded, expected))
    
    def test_shard_column_wise(self):
        """Test column-wise weight sharding."""
        distributor = WeightDistributor(rank=0, world_size=4, tp_size=4)
        
        weight = torch.randn(4096, 4096)
        
        sharded, local_in = distributor.shard_column_wise(weight)
        
        # Each rank gets 1/4 of input dimension
        self.assertEqual(sharded.shape, (4096, 1024))
        self.assertEqual(local_in, 1024)
    
    def test_shard_bias_row_wise(self):
        """Test row-wise bias sharding."""
        distributor = WeightDistributor(rank=0, world_size=4, tp_size=4)
        
        bias = torch.randn(4096)
        
        sharded = distributor.shard_bias_row_wise(bias)
        
        # Each rank gets 1/4 of bias
        self.assertEqual(sharded.shape, (1024,))
    
    def test_distribute_linear_weights_row_wise(self):
        """Test distributing linear layer weights row-wise."""
        distributor = WeightDistributor(rank=0, world_size=4, tp_size=4)
        
        weight = torch.randn(4096, 4096)
        bias = torch.randn(4096)
        
        sharded_w, sharded_b = distributor.distribute_linear_weights(
            weight, bias, sharding_type="row_wise"
        )
        
        self.assertEqual(sharded_w.shape, (1024, 4096))
        self.assertEqual(sharded_b.shape, (1024,))
    
    def test_distribute_linear_weights_column_wise(self):
        """Test distributing linear layer weights column-wise."""
        distributor = WeightDistributor(rank=0, world_size=4, tp_size=4)
        
        weight = torch.randn(4096, 4096)
        bias = torch.randn(4096)
        
        sharded_w, sharded_b = distributor.distribute_linear_weights(
            weight, bias, sharding_type="column_wise"
        )
        
        self.assertEqual(sharded_w.shape, (4096, 1024))
        self.assertEqual(sharded_b.shape, (4096,))


# ============================================================================
# CheckpointSaver Tests
# ============================================================================

class TestCheckpointSaver(BaseModelLoaderTest):
    """Test CheckpointSaver."""
    
    def test_initialization(self):
        """Test saver initialization."""
        saver = CheckpointSaver(
            checkpoint_dir=self.temp_dir,
            rank=0,
            world_size=4
        )
        
        self.assertEqual(saver.rank, 0)
        self.assertEqual(saver.world_size, 4)
        self.assertTrue(Path(self.temp_dir).exists())
    
    def test_save_metadata(self):
        """Test saving metadata."""
        saver = CheckpointSaver(
            checkpoint_dir=self.temp_dir,
            rank=0,
            world_size=1
        )
        
        model = nn.Linear(10, 10)
        metadata = CheckpointMetadata()
        metadata.model_name = "test-model"
        
        ckpt_path = saver.save_checkpoint(model, metadata, step=100)
        
        # Verify metadata file exists
        metadata_file = ckpt_path / "metadata.json"
        self.assertTrue(metadata_file.exists())
        
        # Verify content
        with open(metadata_file, 'r') as f:
            saved_meta = json.load(f)
        self.assertEqual(saved_meta["model_name"], "test-model")
    
    def test_save_weights(self):
        """Test saving weights."""
        saver = CheckpointSaver(
            checkpoint_dir=self.temp_dir,
            rank=0,
            world_size=1
        )
        
        model = nn.Linear(10, 10)
        metadata = CheckpointMetadata()
        
        ckpt_path = saver.save_checkpoint(model, metadata, step=100)
        
        # Verify weight file exists
        weight_file = ckpt_path / "weights_rank0.pt"
        self.assertTrue(weight_file.exists())


# ============================================================================
# Integration Tests
# ============================================================================

class TestModelDistributorIntegration(BaseModelLoaderTest):
    """Integration tests for ModelDistributor."""
    
    def test_initialization(self):
        """Test distributor initialization."""
        config = ModelLoadConfig(checkpoint_dir=self.temp_dir)
        
        with patch('torch.distributed.is_initialized', return_value=False):
            distributor = ModelDistributor(config)
        
        self.assertIsNotNone(distributor.checkpoint_loader)
        self.assertIsNotNone(distributor.weight_distributor)
        self.assertIsNotNone(distributor.checkpoint_saver)
    
    def test_load_model_integration(self):
        """Test end-to-end model loading."""
        # Create a checkpoint
        ckpt_dir = os.path.join(self.temp_dir, "model-step-100")
        os.makedirs(ckpt_dir)
        
        # Create and save metadata
        metadata = CheckpointMetadata()
        metadata.model_name = "test-model"
        metadata_file = os.path.join(ckpt_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f)
        
        # Create and save weights
        model = nn.Linear(10, 10)
        weight_file = os.path.join(ckpt_dir, "weights_rank0.pt")
        torch.save(model.state_dict(), weight_file)
        
        # Load model
        config = ModelLoadConfig(checkpoint_dir=self.temp_dir)
        
        with patch('torch.distributed.is_initialized', return_value=False):
            distributor = ModelDistributor(config)
        
        new_model = nn.Linear(10, 10)
        success = distributor.load_model(new_model)
        
        self.assertTrue(success)
        # Verify weights were loaded
        self.assertTrue(torch.allclose(
            model.weight, new_model.weight
        ))


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
