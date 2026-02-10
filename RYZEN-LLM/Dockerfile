# Ryzanstein LLM Docker Image
# Multi-stage build for development and production
# [REF:AP-009] - Appendix: Technical Stack

# ============================================================================
# Stage 1: Builder - Compile C++ components
# ============================================================================
FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy source files
COPY src/ ./src/
COPY CMakeLists.txt ./

# Build C++ components
RUN mkdir -p build && cd build && \
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF \
        .. && \
    ninja && \
    ninja install

# ============================================================================
# Stage 2: Python runtime
# ============================================================================
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Copy compiled libraries from builder
COPY --from=builder /usr/local/lib/ /usr/local/lib/

# Copy Python source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -e .

# Create directories for models and storage
RUN mkdir -p models storage logs

# Set environment variables
ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Expose API port
EXPOSE 8000

# Expose metrics port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/')"

# Default command
CMD ["python3", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Stage 3: Development image with additional tools
# ============================================================================
FROM runtime AS development

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    curl \
    gdb \
    valgrind \
    && rm -rf /var/lib/apt/lists/*

# Install Python development dependencies
RUN python3 -m pip install --no-cache-dir -e ".[dev,benchmark]"

# Enable debug mode
ENV DEBUG=1

# Development command with auto-reload
CMD ["python3", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================================================
# Usage:
# 
# Build production image:
#   docker build --target runtime -t ryzanstein-llm:latest .
#
# Build development image:
#   docker build --target development -t ryzanstein-llm:dev .
#
# Run with Qdrant:
#   docker-compose up
#
# Run standalone:
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models ryzanstein-llm:latest
# ============================================================================
