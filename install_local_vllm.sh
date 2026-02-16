#!/bin/bash
set -e

VENV_DIR="$HOME/vllm-tools/venv"
VLLM_DIR="${1:-$HOME/vllm-tools/vllm}"

echo "Installing local vLLM build..."

# Check if vLLM directory exists
if [ ! -d "$VLLM_DIR" ]; then
    echo "Error: vLLM directory not found at $VLLM_DIR"
    echo "Usage: $0 [path-to-vllm-directory]"
    exit 1
fi

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Please run setup_venv.sh first"
    exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Set CUDA environment variables (use 12.3 explicitly)
export CUDA_HOME=/usr/local/cuda-12.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Uninstall existing vLLM if present
pip uninstall -y vllm 2>/dev/null || true

# Clean previous builds
cd "$VLLM_DIR"
rm -rf build/ dist/ *.egg-info

# Build and install from source with reduced parallelism
export MAX_JOBS=8
pip install --no-build-isolation . -v

echo "vLLM installed successfully from: $VLLM_DIR"
echo "Verify with: python -c 'import vllm; print(vllm.__version__)'"

