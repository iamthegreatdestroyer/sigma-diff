#!/bin/bash
# Ryzanstein LLM Environment Setup Script
# [REF:AP-009] - Appendix: Technical Stack

set -e  # Exit on error

echo "=== Ryzanstein LLM Environment Setup ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
        return 1
    fi
}

# Check OS
echo "Checking operating system..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${GREEN}✓${NC} Linux detected"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${GREEN}✓${NC} macOS detected"
else
    echo -e "${RED}✗${NC} Unsupported OS: $OSTYPE"
    exit 1
fi

echo ""
echo "=== Checking System Requirements ==="
echo ""

# Check Python version
echo "Checking Python version..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    REQUIRED_VERSION="3.11"
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
        print_status "Python $PYTHON_VERSION (>= 3.11 required)"
    else
        echo -e "${RED}✗${NC} Python $PYTHON_VERSION found, but 3.11+ required"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} Python 3 not found"
    exit 1
fi

# Check CMake version
echo "Checking CMake version..."
if command_exists cmake; then
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    print_status "CMake $CMAKE_VERSION"
else
    echo -e "${YELLOW}!${NC} CMake not found - C++ components will not build"
fi

# Check for C++ compiler
echo "Checking C++ compiler..."
if command_exists g++; then
    GCC_VERSION=$(g++ --version | head -n1)
    print_status "g++ found: $GCC_VERSION"
elif command_exists clang++; then
    CLANG_VERSION=$(clang++ --version | head -n1)
    print_status "clang++ found: $CLANG_VERSION"
else
    echo -e "${YELLOW}!${NC} No C++ compiler found"
fi

# Check for Ninja
echo "Checking Ninja build system..."
if command_exists ninja; then
    NINJA_VERSION=$(ninja --version)
    print_status "Ninja $NINJA_VERSION"
else
    echo -e "${YELLOW}!${NC} Ninja not found - install for faster builds"
fi

# Check CPU features
echo ""
echo "=== Checking CPU Features ==="
echo ""

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command_exists lscpu; then
        CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
        echo "CPU: $CPU_MODEL"
        
        # Check for AVX-512
        if grep -q avx512 /proc/cpuinfo; then
            print_status "AVX-512 support detected"
        else
            echo -e "${YELLOW}!${NC} AVX-512 not detected - some optimizations unavailable"
        fi
        
        # Check for VNNI
        if grep -q avx512_vnni /proc/cpuinfo; then
            print_status "AVX-512 VNNI support detected"
        else
            echo -e "${YELLOW}!${NC} VNNI not detected"
        fi
    fi
fi

echo ""
echo "=== Installing Python Dependencies ==="
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
print_status "pip upgraded"

# Install Python dependencies
if [ -f "pyproject.toml" ]; then
    echo "Installing Python dependencies from pyproject.toml..."
    pip install -e . --quiet
    print_status "Python dependencies installed"
else
    echo -e "${YELLOW}!${NC} pyproject.toml not found - skipping Python dependencies"
fi

echo ""
echo "=== Checking External Dependencies ==="
echo ""

# Check for Qdrant
echo "Checking for Qdrant..."
if command_exists docker; then
    echo "Docker found - Qdrant can be run via: docker run -p 6333:6333 qdrant/qdrant"
else
    echo -e "${YELLOW}!${NC} Docker not found - install to run Qdrant"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Build C++ components: mkdir build && cd build && cmake .. && make"
echo "3. Download models: python scripts/download_models.py"
echo "4. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
echo "5. Run API server: python -m uvicorn src.api.server:app --reload"
echo ""
