FROM mcr.microsoft.com/devcontainers/base:jammy

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install uv - Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
RUN chmod +x /bin/uv && uv --version

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
RUN python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
