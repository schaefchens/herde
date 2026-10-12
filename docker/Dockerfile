# syntax=docker/dockerfile:1.7
# herde's images. Build from the repository's root:
#   docker build -f docker/Dockerfile --target base -t herde:base .
#   docker build -f docker/Dockerfile --target cuda -t herde:cuda .   (amd64; several GB)
#   docker build -f docker/Dockerfile --target rocm -t herde:rocm .   (amd64; about 15 GB)
# or through docker/compose.yaml. A Mac runs herde natively instead: Docker cannot reach its GPU.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04
# Plain Ubuntu: PyTorch's ROCm wheel brings ROCm along.
ARG ROCM_IMAGE=ubuntu:24.04

# --- base: the worker and ffmpeg, no speech model. Text through Ollama beside it; speech only as the
# silent "stub" (a project's own end-to-end tests). -------------------------------------------------------
FROM ${PYTHON_IMAGE} AS base
# HERDE_CONTAINER: file modes on a mounted folder mean nothing here (Docker Desktop on Windows shows 0777).
# HOME in /tmp (a tmpfs): the root file system is read-only.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HERDE_CONTAINER=1 \
    HERDE_CONFIG=/config/config.toml \
    HERDE_SERVE_DATA=/data \
    HERDE_SERVE_HOST=0.0.0.0 \
    HERDE_UI_HOST=0.0.0.0 \
    HERDE_OLLAMA=http://ollama:11434 \
    HOME=/tmp \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin herde \
 && mkdir -p /config /cache /data \
 && chown herde:herde /config /cache /data
WORKDIR /opt/herde
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY herde ./herde
RUN pip install .
USER herde
VOLUME ["/config", "/cache"]
EXPOSE 8737
ENTRYPOINT ["herde"]
CMD ["run"]

# --- cuda: Qwen3-TTS on an NVIDIA GPU (Linux, or Windows through Docker Desktop and WSL2) -----------
FROM ${CUDA_IMAGE} AS cuda
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HERDE_CONTAINER=1 \
    HERDE_CONFIG=/config/config.toml \
    HERDE_SERVE_DATA=/data \
    HERDE_SERVE_HOST=0.0.0.0 \
    HERDE_UI_HOST=0.0.0.0 \
    HERDE_OLLAMA=http://ollama:11434 \
    HERDE_TTS_BACKEND=torch \
    HOME=/tmp \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    NUMBA_CACHE_DIR=/tmp \
    MPLCONFIGDIR=/tmp \
    PATH=/opt/venv/bin:$PATH
# Ubuntu 24.04's python3 is 3.12, the version qwen-tts is made for. sox: its audio helpers call the program.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg sox ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv \
 && useradd --uid 10001 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin herde \
 && mkdir -p /config /cache /data \
 && chown herde:herde /config /cache /data
# torch from PyTorch's CUDA index first, so qwen-tts's requirements find it there.
RUN pip install torch torchaudio --index-url ${TORCH_INDEX}
WORKDIR /opt/herde
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY herde ./herde
RUN pip install ".[cuda]" --extra-index-url ${TORCH_INDEX}
USER herde
VOLUME ["/config", "/cache"]
EXPOSE 8737
ENTRYPOINT ["herde"]
CMD ["run"]

# --- rocm: Qwen3-TTS on an AMD GPU (Linux: Radeon RX 6800 and newer, Ryzen AI 300 and Ryzen AI Max) --------
# The host needs only its amdgpu driver. ROCm's torch calls an AMD GPU "cuda" too: the torch backend as it is.
FROM ${ROCM_IMAGE} AS rocm
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2
# MIOpen compiles its kernels on first use: kept in /cache, not in HOME's tmpfs, so a restart needs no second time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HERDE_CONTAINER=1 \
    HERDE_CONFIG=/config/config.toml \
    HERDE_SERVE_DATA=/data \
    HERDE_SERVE_HOST=0.0.0.0 \
    HERDE_UI_HOST=0.0.0.0 \
    HERDE_OLLAMA=http://ollama:11434 \
    HERDE_TTS_BACKEND=torch \
    HOME=/tmp \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    MIOPEN_USER_DB_PATH=/cache/miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen \
    NUMBA_CACHE_DIR=/tmp \
    MPLCONFIGDIR=/tmp \
    PATH=/opt/venv/bin:$PATH
# Ubuntu 24.04's python3 is 3.12, the version qwen-tts is made for. sox: its audio helpers call the program.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg sox ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv \
 && useradd --uid 10001 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin herde \
 && mkdir -p /config /cache /data \
 && chown herde:herde /config /cache /data
# torch from PyTorch's ROCm index first, so qwen-tts's requirements find it there. 2.11 is the last torch with a
# torchaudio, which qwen-tts needs; pinned, as pip would otherwise fetch every newer torch (6 GB each) to learn that.
RUN pip install torch==2.11.0 torchaudio==2.11.0 --index-url ${TORCH_INDEX}
WORKDIR /opt/herde
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY herde ./herde
RUN pip install ".[rocm]" --extra-index-url ${TORCH_INDEX}
USER herde
VOLUME ["/config", "/cache"]
EXPOSE 8737
ENTRYPOINT ["herde"]
CMD ["run"]
