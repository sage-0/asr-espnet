ARG CUDA_IMAGE="12.5.0-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y sudo vim nano emacs less wget curl git htop make swig && \
    apt-get install -y --reinstall build-essential && \
    apt-get install -y python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN apt-get install -y mecab libmecab-dev mecab-ipadic-utf8


# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1

# Install additional dependencies
RUN python3 -m pip install pytest cmake scikit-build setuptools tk_tk