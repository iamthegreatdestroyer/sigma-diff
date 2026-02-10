# Self-Hosted GPU Runner Configuration

# Phase 2 GPU Training CI/CD Setup

## Environment Configuration

Create `.env` file in the runner directory:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
CUDA_HOME=/usr/local/cuda-12.1
CUDA_PATH=/usr/local/cuda-12.1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# PyTorch Configuration
TORCH_HOME=/mnt/storage/torch_cache
TORCH_DEVICE_INDEX=0
PYTORCH_ENABLE_MPS_FALLBACK=1

# HuggingFace Configuration
HUGGINGFACE_HUB_CACHE=/mnt/storage/hf_cache
HF_HOME=/mnt/storage/huggingface

# S3/AWS Configuration
AWS_REGION=us-east-1
AWS_DEFAULT_REGION=us-east-1

# Training Configuration
TRAINING_DATA_DIR=/mnt/data/training
CHECKPOINT_DIR=/mnt/storage/checkpoints
RESULT_DIR=/mnt/storage/results

# System Configuration
NUM_WORKERS=4
PREFETCH_FACTOR=2
```

## Ubuntu/Linux Runner Setup

### 1. Install CUDA 12.1

```bash
# Add NVIDIA repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Install CUDA
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-12-1

# Set PATH
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### 2. Install cuDNN 8.9

```bash
# Download and install cuDNN (requires NVIDIA developer account)
# https://developer.nvidia.com/cudnn

tar -xzvf cudnn-linux-x86_64-8.9.x.tgz
sudo cp cudnn-linux-x86_64-8.9.x/include/cudnn*.h /usr/local/cuda-12.1/include
sudo cp -P cudnn-linux-x86_64-8.9.x/lib/libcudnn* /usr/local/cuda-12.1/lib64
sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h
sudo chmod a+r /usr/local/cuda-12.1/lib64/libcudnn*
```

### 3. Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Test with PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"
```

### 4. Configure Storage Paths

```bash
# Create cache directories
sudo mkdir -p /mnt/storage/{torch_cache,hf_cache,checkpoints,results,training_data}
sudo chown -R $USER:$USER /mnt/storage

# Create data directory
sudo mkdir -p /mnt/data/training
sudo chown -R $USER:$USER /mnt/data
```

### 5. Setup GitHub Runner

```bash
# Create runner directory
mkdir -p ~/github-runner
cd ~/github-runner

# Download runner
curl -o actions-runner-linux-x64-2.312.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.312.0/actions-runner-linux-x64-2.312.0.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.312.0.tar.gz

# Configure (interactive)
./config.sh --url https://github.com/iamthegreatdestroyer/Ryzanstein --token <TOKEN>

# Run as service
sudo ./svc.sh install
sudo ./svc.sh start

# Verify runner
ps aux | grep runsvc.sh
```

### 6. Add Runner Labels

During configuration, add labels:

- `self-hosted`
- `gpu`
- `cuda-12`
- `ubuntu-latest`

```bash
# Or update existing runner:
cd ~/github-runner
./config.sh --url https://github.com/iamthegreatdestroyer/Ryzanstein --token <TOKEN> --labels "self-hosted,gpu,cuda-12"
```

## Windows GPU Runner Setup

### 1. Install CUDA 12.1 on Windows

```powershell
# Download from https://developer.nvidia.com/cuda-12-1-0-download-archive
# Select: Windows, x86_64, 11 or later

# Run installer
.\cuda_12.1.0_536.40_windows.exe
```

### 2. Verify Installation

```powershell
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Test PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"
```

### 3. Setup GitHub Actions Runner on Windows

```powershell
# Create runner directory
New-Item -ItemType Directory -Path "$env:USERPROFILE\github-runner"
cd "$env:USERPROFILE\github-runner"

# Download runner (check for latest version)
Invoke-WebRequest -Uri "https://github.com/actions/runner/releases/download/v2.312.0/actions-runner-win-x64-2.312.0.zip" -OutFile "runner.zip"

# Extract
Expand-Archive -Path "runner.zip" -DestinationPath "."

# Configure runner
.\config.cmd --url "https://github.com/iamthegreatdestroyer/Ryzanstein" --token "<TOKEN>" --labels "self-hosted,gpu,cuda-12,windows"

# Install as service
.\svc.cmd install

# Start service
.\svc.cmd start
```

## Docker GPU Image (Optional)

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-training.txt .
RUN pip install --no-cache-dir -r requirements-training.txt

# Setup cache directories
RUN mkdir -p /cache/{torch,huggingface} && \
    chmod 777 /cache/{torch,huggingface}

# Environment
ENV TORCH_HOME=/cache/torch \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface \
    CUDA_VISIBLE_DEVICES=0

WORKDIR /app
```

Build and run:

```bash
docker build -t ryzen-llm-trainer:latest .
docker run --gpus all -v /mnt/data:/data -v /mnt/storage:/storage ryzen-llm-trainer:latest python scripts/training_loop.py
```

## Performance Tuning

### GPU Memory Optimization

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Profile training
python -m torch.utils.bottleneck scripts/training_loop.py --config configs/training_configuration.yaml
```

### Network Optimization

```bash
# Enable GPU-GPU P2P (if multi-GPU later)
nvidia-smi -pm 1

# Check peer access
nvidia-smi -lgc 1000  # Lock GPU clock to max
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `batch_size` in `configs/training_configuration.yaml`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Enable `gradient_checkpointing: true`
4. Set `mixed_precision: bf16` for FP32 models

### CUDA Driver Issues

```bash
# Check driver version
nvidia-smi | grep "Driver Version"

# Update driver (Ubuntu)
sudo apt-get install --only-upgrade nvidia-driver-545

# Restart GPU
sudo nvidia-smi -pm 1
nvidia-smi -c DEFAULT
```

### Connection Timeout

1. Check runner logs: `~/github-runner/_diag/*.log`
2. Verify network: `ping github.com`
3. Check authentication token
4. Restart runner: `./svc.sh restart`

## Monitoring

### Real-time GPU Monitoring

```bash
# Install nvtop
sudo apt-get install nvtop

# Monitor
nvtop
```

### Integration with Prometheus/Grafana

See main observability documentation for metrics collection setup.
