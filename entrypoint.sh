#!/bin/bash
set -e

# Build TensorRT engine for the CFM estimator on first startup.
# The engine is GPU-architecture-specific, so it must be built on the target GPU.
# Once built, the engine file is cached in the HF cache dir and reused on subsequent starts.

CKPT_DIR="${HF_HOME:-${HOME}/.cache/huggingface}/hub"

if [ "${SKIP_TRT_BUILD:-0}" = "1" ]; then
    echo "[TRT] Skipping TRT engine build (SKIP_TRT_BUILD=1)"
else
    # Download s3gen.safetensors if not already cached
    echo "[TRT] Ensuring s3gen.safetensors is downloaded..."
    uv run python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='ResembleAI/chatterbox',
    filename='s3gen.safetensors',
    revision='05e904af2b5c7f8e482687a9d7336c5c824467d9',
)
print(path)
" > /tmp/s3gen_path.txt

    S3GEN_PATH=$(cat /tmp/s3gen_path.txt | tail -1)
    rm -f /tmp/s3gen_path.txt

    if [ -z "$S3GEN_PATH" ] || [ ! -f "$S3GEN_PATH" ]; then
        echo "[TRT] Failed to download s3gen.safetensors, skipping TRT build"
    else
        S3GEN_DIR=$(dirname "$S3GEN_PATH")
        ENGINE_PATH="${S3GEN_DIR}/conditional_decoder.engine"

        if [ -f "$ENGINE_PATH" ]; then
            echo "[TRT] Engine already exists at ${ENGINE_PATH}, skipping build"
        else
            ONNX_PATH="/tmp/decoder.onnx"

            echo "[TRT] Step 1/2: Exporting ONNX model..."
            uv run python /app/scripts/export_decoder_onnx.py \
                --ckpt-dir "$S3GEN_DIR" \
                --output "$ONNX_PATH"

            echo "[TRT] Step 2/2: Building TensorRT engine..."
            uv run python /app/scripts/build_trt_engine.py \
                --onnx "$ONNX_PATH" \
                --output "$ENGINE_PATH"

            rm -f "$ONNX_PATH"
            echo "[TRT] Engine built successfully at ${ENGINE_PATH}"
        fi
    fi
fi

echo "Starting server..."
exec uv run uvicorn server:app --host 0.0.0.0 --port 4123
