#!/bin/bash
set -e

VENV_DIR="$HOME/vllm-tools/venv"

echo "Setting up Python virtual environment for vLLM..."

# Install Python and venv if not present
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv "$VENV_DIR"

# Activate and upgrade pip
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# Install latest PyTorch nightly with CUDA 12.4 support (compatible with CUDA 12.3)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

# Install xformers (will be built from source during vLLM installation if needed)

# Install build dependencies for vLLM
pip install \
    ninja \
    packaging \
    wheel \
    setuptools-scm

# Install common dependencies for vLLM
pip install \
    transformers \
    ray \
    fastapi \
    uvicorn \
    pydantic \
    sentencepiece \
    psutil \
    accelerate

echo "Virtual environment setup complete!"
echo "Activate with: source $VENV_DIR/bin/activate"

