#!/usr/bin/env python3
import torch
import sys

print("Testing PyTorch environment...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print("Creating tensor on CUDA...")
    t = torch.randn(10, 10).cuda()
    print(f"Tensor created successfully: {t.device}")
else:
    print("CUDA not available, using CPU")

print("Environment check complete!")
sys.exit(0)
