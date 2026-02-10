#!/usr/bin/env python3
"""
Training Artifact Sync System
Synchronizes model checkpoints and metrics to S3 storage backend.
Supports versioning, metadata tracking, and integrity verification.
"""

import os
import json
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """Calculate file hash for integrity verification."""
    hasher = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def sync_to_s3(
    local_path: str,
    s3_bucket: str,
    s3_key: str,
    metadata: Optional[Dict[str, str]] = None
) -> bool:
    """
    Upload file to S3 with metadata and versioning.
    
    Args:
        local_path: Local file path
        s3_bucket: S3 bucket name
        s3_key: S3 object key (path in bucket)
        metadata: Optional metadata dict
        
    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client = boto3.client('s3')
        
        # Calculate file hash
        file_hash = calculate_file_hash(local_path)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        default_metadata = {
            'uploaded-by': 'github-actions',
            'git-sha': os.getenv('GITHUB_SHA', 'unknown'),
            'git-ref': os.getenv('GITHUB_REF', 'unknown'),
            'run-number': os.getenv('GITHUB_RUN_NUMBER', 'unknown'),
            'timestamp': datetime.utcnow().isoformat(),
            'file-hash': file_hash,
        }
        default_metadata.update(metadata)
        
        # Upload to S3
        with open(local_path, 'rb') as f:
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=f,
                Metadata=default_metadata,
                ContentType='application/octet-stream' if local_path.endswith('.pt') else 'application/json',
                CacheControl='max-age=3600',
                ServerSideEncryption='AES256'
            )
        
        # Get file size
        file_size_mb = Path(local_path).stat().st_size / (1024 * 1024)
        
        logger.info(f"✅ Uploaded {local_path} ({file_size_mb:.1f} MB) to s3://{s3_bucket}/{s3_key}")
        logger.info(f"   SHA256: {file_hash}")
        return True
        
    except ClientError as e:
        logger.error(f"❌ S3 upload failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during upload: {e}")
        return False


def create_manifest(
    checkpoint_path: str,
    metrics_path: str,
    s3_bucket: str,
    run_number: int
) -> Dict[str, Any]:
    """Create manifest file tracking checkpoint and metrics."""
    manifest = {
        'run_number': run_number,
        'timestamp': datetime.utcnow().isoformat(),
        'git_sha': os.getenv('GITHUB_SHA', 'unknown'),
        'git_ref': os.getenv('GITHUB_REF', 'unknown'),
        'checkpoint': {
            'local_path': checkpoint_path,
            'hash': calculate_file_hash(checkpoint_path),
            'size_mb': Path(checkpoint_path).stat().st_size / (1024 * 1024),
            's3_path': f's3://{s3_bucket}/phase2/checkpoint-{run_number}.pt'
        },
        'metrics': {
            'local_path': metrics_path,
            'hash': calculate_file_hash(metrics_path),
            's3_path': f's3://{s3_bucket}/metrics/metrics-{run_number}.json'
        }
    }
    
    # Load and include actual metrics
    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
            manifest['metrics']['data'] = {
                'final_loss': metrics_data.get('final_loss'),
                'total_steps': metrics_data.get('total_steps'),
                'total_epochs': metrics_data.get('total_epochs'),
                'duration_hours': metrics_data.get('total_duration_hours'),
            }
    except Exception as e:
        logger.warning(f"Could not load metrics data: {e}")
    
    return manifest


def upload_manifest(
    manifest: Dict[str, Any],
    s3_bucket: str,
    run_number: int
) -> bool:
    """Upload manifest file to S3."""
    try:
        s3_client = boto3.client('s3')
        manifest_key = f'manifests/run-{run_number}-manifest.json'
        
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2),
            ContentType='application/json',
            ServerSideEncryption='AES256'
        )
        
        logger.info(f"✅ Uploaded manifest to s3://{s3_bucket}/{manifest_key}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Manifest upload failed: {e}")
        return False


def verify_s3_upload(s3_bucket: str, s3_key: str, local_hash: str) -> bool:
    """Verify uploaded file integrity via S3 metadata."""
    try:
        s3_client = boto3.client('s3')
        response = s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
        
        remote_hash = response.get('Metadata', {}).get('file-hash', '')
        if remote_hash == local_hash:
            logger.info(f"✅ Integrity verified: {s3_key}")
            return True
        else:
            logger.warning(f"⚠️  Hash mismatch for {s3_key}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Sync training artifacts to S3',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--metrics',
        required=True,
        help='Path to metrics JSON file'
    )
    parser.add_argument(
        '--s3-bucket',
        default='ryzen-llm-checkpoints',
        help='S3 bucket name (default: ryzen-llm-checkpoints)'
    )
    parser.add_argument(
        '--run-number',
        type=int,
        default=int(os.getenv('GITHUB_RUN_NUMBER', '0')),
        help='GitHub run number'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify upload integrity'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.checkpoint).exists():
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1
    
    if not Path(args.metrics).exists():
        logger.error(f"❌ Metrics not found: {args.metrics}")
        return 1
    
    logger.info("🚀 Starting artifact sync to S3...")
    
    # Upload checkpoint
    checkpoint_key = f'phase2/checkpoint-{args.run_number}.pt'
    checkpoint_uploaded = sync_to_s3(
        args.checkpoint,
        args.s3_bucket,
        checkpoint_key,
        metadata={'artifact-type': 'model-checkpoint'}
    )
    
    # Upload metrics
    metrics_key = f'metrics/metrics-{args.run_number}.json'
    metrics_uploaded = sync_to_s3(
        args.metrics,
        args.s3_bucket,
        metrics_key,
        metadata={'artifact-type': 'training-metrics'}
    )
    
    # Create and upload manifest
    manifest = create_manifest(
        args.checkpoint,
        args.metrics,
        args.s3_bucket,
        args.run_number
    )
    manifest_uploaded = upload_manifest(manifest, args.s3_bucket, args.run_number)
    
    # Verify uploads if requested
    if args.verify:
        checkpoint_hash = calculate_file_hash(args.checkpoint)
        verify_checkpoint = verify_s3_upload(
            args.s3_bucket,
            checkpoint_key,
            checkpoint_hash
        )
        
        metrics_hash = calculate_file_hash(args.metrics)
        verify_metrics = verify_s3_upload(
            args.s3_bucket,
            metrics_key,
            metrics_hash
        )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 Sync Summary:")
    logger.info(f"  Checkpoint: {'✅' if checkpoint_uploaded else '❌'}")
    logger.info(f"  Metrics:    {'✅' if metrics_uploaded else '❌'}")
    logger.info(f"  Manifest:   {'✅' if manifest_uploaded else '❌'}")
    logger.info(f"  S3 Bucket:  {args.s3_bucket}")
    logger.info(f"  Run Number: {args.run_number}")
    logger.info("="*60 + "\n")
    
    # Return success only if all uploads succeeded
    success = checkpoint_uploaded and metrics_uploaded and manifest_uploaded
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
