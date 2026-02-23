# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 from deadsnakes PPA and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        curl git build-essential \
        libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock .latest-version.generated.txt ./

# Install dependencies (cached unless pyproject.toml or uv.lock change)
RUN uv sync --frozen --no-dev --no-install-project

# Copy model config directories (model.safetensors are downloaded at runtime)
COPY t3-model/ t3-model/
COPY t3-model-multilingual/ t3-model-multilingual/

# Copy source code and voice clone files
COPY src/ src/
COPY server.py .
COPY turkish_voice_clone_male.wav .

# Install the project itself
RUN uv sync --frozen --no-dev

ENV VLLM_USE_V1=0

EXPOSE 4123

CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4123"]
