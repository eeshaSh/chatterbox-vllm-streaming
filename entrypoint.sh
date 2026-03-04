#!/bin/bash
set -e

# Build TensorRT engine for the CFM estimator on first startup.
# The engine is GPU-architecture-specific, so it must be built on the target GPU.
# Once built, the engine file is cached in the HF cache dir and reused on subsequent starts.

CKPT_DIR="${HF_HOME:-${HOME}/.cache/huggingface}/hub"

if [ "${SKIP_TRT_BUILD:-0}" = "1" ]; then
    echo "[TRT] Skipping TRT engine build (SKIP_TRT_BUILD=1)"
else
    # Find s3gen.safetensors in the HF cache
    S3GEN_PATH=$(find "$CKPT_DIR" -name "s3gen.safetensors" -type f 2>/dev/null | head -1)

    if [ -z "$S3GEN_PATH" ]; then
        echo "[TRT] s3gen.safetensors not found yet — model will be downloaded at server start."
        echo "[TRT] Skipping TRT build; will use PyTorch eager mode this run."
        echo "[TRT] Re-run the container after model download to build the TRT engine."
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
