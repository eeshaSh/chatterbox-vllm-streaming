#!/usr/bin/env python3
"""Export the ConditionalDecoder (CFM estimator) to ONNX format.

Usage:
    python scripts/export_decoder_onnx.py --ckpt-dir /path/to/checkpoints --output decoder.onnx

The exported ONNX model has dynamic axes on batch (dim 0) and time (dim 2).
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from chatterbox_vllm.models.s3gen.decoder import ConditionalDecoder


def build_estimator() -> ConditionalDecoder:
    """Build ConditionalDecoder with production config (matches S3Token2Mel.__init__)."""
    return ConditionalDecoder(
        in_channels=320,
        out_channels=80,
        causal=True,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
    )


def extract_estimator_weights(s3gen_weights: dict) -> dict:
    """Extract flow.decoder.estimator.* keys and strip the prefix."""
    prefix = "flow.decoder.estimator."
    return {
        k[len(prefix):]: v
        for k, v in s3gen_weights.items()
        if k.startswith(prefix)
    }


def main():
    parser = argparse.ArgumentParser(description="Export ConditionalDecoder to ONNX")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Directory containing s3gen.safetensors")
    parser.add_argument("--output", type=str, default="decoder.onnx",
                        help="Output ONNX file path (default: decoder.onnx)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX model with onnxruntime after export")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    s3gen_path = ckpt_dir / "s3gen.safetensors"
    if not s3gen_path.exists():
        raise FileNotFoundError(f"s3gen.safetensors not found at {s3gen_path}")

    print(f"Loading weights from {s3gen_path}")
    s3gen_weights = load_file(s3gen_path)
    estimator_weights = extract_estimator_weights(s3gen_weights)
    print(f"Extracted {len(estimator_weights)} estimator weight keys")

    print("Building ConditionalDecoder")
    model = build_estimator()
    model.load_state_dict(estimator_weights)
    model.eval()
    # Export in float32 — let TensorRT handle FP16 optimization
    model.float()

    # Create dummy inputs matching production shapes
    # B2 = 2*batch for CFG, T = typical mel timestep length
    B2, T = 2, 400
    device = "cpu"
    dummy_x = torch.randn(B2, 80, T, device=device)
    dummy_mask = torch.ones(B2, 1, T, device=device)
    dummy_mu = torch.randn(B2, 80, T, device=device)
    dummy_t = torch.rand(B2, device=device)
    dummy_spks = torch.randn(B2, 80, device=device)
    dummy_cond = torch.randn(B2, 80, T, device=device)

    print(f"Exporting to ONNX (opset {args.opset}): {args.output}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_x, dummy_mask, dummy_mu, dummy_t, dummy_spks, dummy_cond),
            args.output,
            input_names=["x", "mask", "mu", "t", "spks", "cond"],
            output_names=["output"],
            dynamic_axes={
                "x": {0: "batch", 2: "time"},
                "mask": {0: "batch", 2: "time"},
                "mu": {0: "batch", 2: "time"},
                "t": {0: "batch"},
                "spks": {0: "batch"},
                "cond": {0: "batch", 2: "time"},
                "output": {0: "batch", 2: "time"},
            },
            opset_version=args.opset,
        )

    # Validate the ONNX model
    import onnx
    print("Validating ONNX model...")
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")

    if args.verify:
        print("Verifying numerical accuracy with onnxruntime...")
        import onnxruntime as ort
        import numpy as np

        # Get PyTorch reference output
        with torch.no_grad():
            ref_output = model(dummy_x, dummy_mask, dummy_mu, dummy_t, dummy_spks, dummy_cond)

        # Run ONNX inference
        sess = ort.InferenceSession(args.output)
        ort_output = sess.run(None, {
            "x": dummy_x.numpy(),
            "mask": dummy_mask.numpy(),
            "mu": dummy_mu.numpy(),
            "t": dummy_t.numpy(),
            "spks": dummy_spks.numpy(),
            "cond": dummy_cond.numpy(),
        })[0]

        max_diff = np.max(np.abs(ref_output.numpy() - ort_output))
        mean_diff = np.mean(np.abs(ref_output.numpy() - ort_output))
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")

        if max_diff < 1e-2:
            print("Numerical verification PASSED (atol < 1e-2)")
        else:
            print(f"WARNING: Max difference {max_diff:.6f} exceeds tolerance 1e-2")

    print("Done!")


if __name__ == "__main__":
    main()
