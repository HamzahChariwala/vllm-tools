#!/bin/bash
set -e

echo "Installing CUDA toolkit for vLLM..."

# Update system
apt-get update

# Install prerequisites
apt-get install -y \
    build-essential \
    wget \
    git

# Install CMake 3.26+ (required for vLLM)
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null
apt-get update
apt-get install -y cmake

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

# Update package list
apt-get update

# Install CUDA 12.3 toolkit (required for vLLM's fabric memory features)
# Note: Driver 535.288.01 supports up to CUDA 12.2, but CUDA 12.3 is backward compatible
apt-get install -y cuda-toolkit-12-3

# Set up environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh

# Create symlink for CUDA 12.1 (vLLM compatibility)
ln -sf /usr/local/cuda-12.3 /usr/local/cuda-12.1

echo "Installation complete!"
echo "Run 'source /etc/profile.d/cuda.sh' or restart your shell, then verify with: nvcc --version"
