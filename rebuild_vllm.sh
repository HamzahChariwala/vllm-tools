#!/bin/bash
set -e

VENV_DIR="$HOME/vllm-tooling/venv"
VLLM_DIR="${1:-$HOME/vllm-tooling/vllm}"

echo "Performing incremental rebuild of vLLM..."

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

# Set parallelism
export MAX_JOBS=8

cd "$VLLM_DIR"

# Incremental build - keeps existing build artifacts
# Only recompiles changed files
pip install --no-build-isolation .

echo "Incremental rebuild complete!"
echo "Verify with: python -c 'import vllm; print(vllm.__version__)'"


