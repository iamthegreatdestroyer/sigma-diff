#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

print("Step 1: Imports complete")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Step 2: Logging configured")

# Load config
import yaml

CONFIG_PATH = Path('s:\\Ryot\\training_configuration.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
    
print(f"Step 3: Config loaded: {CONFIG_PATH}")

# Create simple model like in training_loop.py
class SimpleModel(nn.Module):
    def __init__(self, vocab_size=2048, embedding_dim=256, num_heads=4, num_layers=2, ff_dim=512, max_seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embeds = self.embedding(input_ids)
        pos_embeds = self.pos_embedding(positions)
        hidden = embeds + pos_embeds
        encoded = self.transformer_encoder(hidden)
        logits = self.output_projection(encoded)
        return logits

print("Step 4: Model class defined")

# Create model
device = "cpu"
model_config = config['model']
model = SimpleModel(
    vocab_size=model_config.get('vocab_size', 2048),
    embedding_dim=model_config.get('embedding_dim', 256),
    num_heads=model_config.get('num_heads', 4),
    num_layers=model_config.get('num_layers', 2),
    ff_dim=model_config.get('ff_dim', 512),
    max_seq_len=model_config.get('max_seq_len', 128)
).to(device)

print(f"Step 5: Model created on {device}")

# Create data
vocab_size = model_config['vocab_size']
max_seq_len = model_config['max_seq_len']
batch_size = config['training']['batch_size']
num_samples = batch_size * 10

print(f"Step 6: Creating synthetic data... (vocab_size={vocab_size}, seq_len={max_seq_len}, batch_size={batch_size}, num_samples={num_samples})")

train_input_ids = torch.randint(1, vocab_size, (num_samples, max_seq_len))
train_labels = torch.randint(0, vocab_size, (num_samples, max_seq_len))

print("Step 7: Data created")

train_dataset = TensorDataset(train_input_ids, train_labels)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

print("Step 8: DataLoader created")

# Do one batch
print("Step 9: Iterating through first batch...")
for i, (input_ids, labels) in enumerate(train_loader):
    print(f"  Got batch {i}: input_ids.shape={input_ids.shape}, labels.shape={labels.shape}")
    if i == 0:
        break

print("Step 10: First batch processed successfully!")
print("All steps completed!")
