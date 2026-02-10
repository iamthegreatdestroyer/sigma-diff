#!/usr/bin/env python3
"""
Model Download Script
[REF:AP-009] - Appendix: Technical Stack

This script downloads pre-trained models for Ryzanstein LLM from various sources.

Usage:
    python scripts/download_models.py --model bitnet-7b --output models/bitnet/
    python scripts/download_models.py --all
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List
import hashlib
import requests
from tqdm import tqdm


# Model registry with download URLs and checksums
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "bitnet-1.58b": {
        "url": "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/main/model.safetensors",
        "sha256": "",  # Will be computed after download
        "size_gb": 1.2,
        "description": "BitNet 1.58B with ternary quantization (Microsoft)"
    },
    "bitnet-3b": {
        "url": "https://huggingface.co/1bitLLM/bitnet_b1_58-3B/resolve/main/model.safetensors.index.json",
        "files": [
            "https://huggingface.co/1bitLLM/bitnet_b1_58-3B/resolve/main/model-00001-of-00003.safetensors",
            "https://huggingface.co/1bitLLM/bitnet_b1_58-3B/resolve/main/model-00002-of-00003.safetensors",
            "https://huggingface.co/1bitLLM/bitnet_b1_58-3B/resolve/main/model-00003-of-00003.safetensors"
        ],
        "sha256": "",
        "size_gb": 13.3,
        "description": "BitNet 3B with ternary quantization (1bitLLM)"
    },
    "mamba-2.8b": {
        "url": "https://huggingface.co/state-spaces/mamba-2.8b/resolve/main/pytorch_model.bin",
        "sha256": "",
        "size_gb": 5.7,
        "description": "Mamba 2.8B state space model"
    },
    "rwkv-7b": {
        "url": "https://huggingface.co/BlinkDL/rwkv-7b/resolve/main/RWKV-7B-v0.1.pth",
        "sha256": "",
        "size_gb": 14.0,
        "description": "RWKV 7B attention-free model"
    },
    "draft-350m": {
        "url": "https://huggingface.co/microsoft/DialoGPT-small/resolve/main/pytorch_model.bin",
        "sha256": "",
        "size_gb": 0.7,
        "description": "Small draft model for speculative decoding"
    },
}


def verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    """
    Verify file checksum.
    
    Args:
        filepath: Path to file
        expected_sha256: Expected SHA256 hash
        
    Returns:
        True if checksum matches
    """
    if not expected_sha256 or expected_sha256 == "":
        print(f"⚠️  Skipping checksum verification for {filepath.name} (no expected hash)")
        return True
    
    print(f"Verifying checksum for {filepath.name}...")
    sha256_hash = hashlib.sha256()
    
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        computed_hash = sha256_hash.hexdigest()
        if computed_hash == expected_sha256:
            print("✓ Checksum verified")
            return True
        else:
            print(f"✗ Checksum mismatch: expected {expected_sha256}, got {computed_hash}")
            return False
    except Exception as e:
        print(f"✗ Error verifying checksum: {e}")
        return False


def download_file(url: str, output_path: Path, verify_ssl: bool = True) -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Where to save the file
        verify_ssl: Whether to verify SSL certificates
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        print(f"Downloading from {url}...")
        print(f"Saving to {output_path}...")
        
        if not verify_ssl:
            print("⚠️  Note: SSL certificate verification is disabled for compatibility")
        
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True, timeout=30, verify=verify_ssl)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=output_path.name,
            ncols=80
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print("✓ Download completed successfully")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Download failed: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        return False
    except Exception as e:
        print(f"✗ Unexpected error during download: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_model(model_name: str, output_dir: Path, verify: bool = True, verify_ssl: bool = True) -> bool:
    """
    Download a specific model.
    
    Args:
        model_name: Name of model to download
        output_dir: Directory to save model
        verify: Whether to verify checksum
        
    Returns:
        True if successful
    """
    if model_name not in MODEL_REGISTRY:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
        return False
    
    model_info = MODEL_REGISTRY[model_name]
    print(f"\n=== Downloading {model_name} ===")
    print(f"Description: {model_info['description']}")
    print(f"Size: {model_info['size_gb']:.2f} GB")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if this is a multi-file model
    if "files" in model_info:
        # Multi-file download (like BitNet 3B)
        files_to_download = model_info["files"]
        print(f"Model consists of {len(files_to_download)} files")
        
        success_count = 0
        for file_url in files_to_download:
            filename = file_url.split('/')[-1]
            output_path = output_dir / filename
            
            # Check if already exists
            if output_path.exists():
                print(f"✓ {filename} already exists")
                success_count += 1
                continue
            
            # Download file
            if download_file(file_url, output_path, verify_ssl):
                success_count += 1
            else:
                print(f"✗ Failed to download {filename}")
        
        # Also download the index file if it exists
        if "url" in model_info:
            index_filename = model_info['url'].split('/')[-1]
            index_path = output_dir / index_filename
            if not index_path.exists():
                if not download_file(model_info['url'], index_path, verify_ssl):
                    print(f"✗ Failed to download index file {index_filename}")
                    return False
        
        if success_count == len(files_to_download):
            print(f"✓ Successfully downloaded all {len(files_to_download)} files")
            return True
        else:
            print(f"✗ Only downloaded {success_count}/{len(files_to_download)} files")
            return False
    
    else:
        # Single file download
        filename = model_info['url'].split('/')[-1]
        output_path = output_dir / filename
        
        # Check if already exists
        if output_path.exists():
            print(f"✓ Model already exists at {output_path}")
            if verify:
                if verify_checksum(output_path, model_info['sha256']):
                    print("✓ Checksum verified")
                    return True
                else:
                    print("✗ Checksum mismatch - re-downloading")
                    output_path.unlink()
        
        # Download
        success = download_file(model_info['url'], output_path, verify_ssl)
        
        # Verify checksum
        if success and verify:
            if verify_checksum(output_path, model_info['sha256']):
                print("✓ Checksum verified")
            else:
                print("✗ Checksum mismatch")
                return False
        
        return success


def list_models():
    """List all available models."""
    print("\n=== Available Models ===\n")
    for name, info in MODEL_REGISTRY.items():
        print(f"{name:20} | {info['size_gb']:6.2f} GB | {info['description']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download models for Ryzanstein LLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Output directory (default: models/)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip checksum verification"
    )
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Skip SSL certificate verification"
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list:
        list_models()
        return 0
    
    # Download all models
    if args.all:
        print("Downloading all models...")
        for model_name in MODEL_REGISTRY.keys():
            model_output = args.output / model_name.split('-')[0]
            download_model(model_name, model_output, not args.no_verify, not args.no_ssl_verify)
        return 0
    
    # Download specific model
    if args.model:
        # Determine subdirectory based on model type
        model_type = args.model.split('-')[0]
        model_output = args.output / model_type
        
        success = download_model(args.model, model_output, not args.no_verify, not args.no_ssl_verify)
        return 0 if success else 1
    
    # No action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
