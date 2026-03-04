#!/usr/bin/env python3
"""Build a TensorRT engine from the exported ONNX ConditionalDecoder model.

Usage:
    python scripts/build_trt_engine.py --onnx decoder.onnx --output conditional_decoder.engine

This configures dynamic shape optimization profiles and enables FP16 if supported.
Engine build can take 5-15 minutes depending on GPU.
"""

import argparse
import subprocess

import tensorrt as trt


def get_gpu_info() -> str:
    """Get GPU name and SM architecture for naming the engine file."""
    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        name = props.name.replace(" ", "_")
        sm = f"sm{props.major}{props.minor}"
        return f"{name}_{sm}"
    except Exception:
        return "unknown_gpu"


def build_engine(onnx_path: str, output_path: str, fp16: bool = True,
                 batch_min: int = 2, batch_opt: int = 2, batch_max: int = 20,
                 time_min: int = 100, time_opt: int = 400, time_max: int = 800):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    print(f"Network inputs: {network.num_inputs}, outputs: {network.num_outputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input {i}: {inp.name} shape={inp.shape} dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output {i}: {out.name} shape={out.shape} dtype={out.dtype}")

    config = builder.create_builder_config()
    # Use up to 4GB of GPU memory for tactics workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)
    elif fp16:
        print("WARNING: FP16 requested but not supported on this platform, using FP32")

    # Configure optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()

    # x: (B2, 80, T)
    profile.set_shape("x",
                       min=(batch_min, 80, time_min),
                       opt=(batch_opt, 80, time_opt),
                       max=(batch_max, 80, time_max))
    # mask: (B2, 1, T)
    profile.set_shape("mask",
                       min=(batch_min, 1, time_min),
                       opt=(batch_opt, 1, time_opt),
                       max=(batch_max, 1, time_max))
    # mu: (B2, 80, T)
    profile.set_shape("mu",
                       min=(batch_min, 80, time_min),
                       opt=(batch_opt, 80, time_opt),
                       max=(batch_max, 80, time_max))
    # t: (B2,)
    profile.set_shape("t",
                       min=(batch_min,),
                       opt=(batch_opt,),
                       max=(batch_max,))
    # spks: (B2, 80)
    profile.set_shape("spks",
                       min=(batch_min, 80),
                       opt=(batch_opt, 80),
                       max=(batch_max, 80))
    # cond: (B2, 80, T)
    profile.set_shape("cond",
                       min=(batch_min, 80, time_min),
                       opt=(batch_opt, 80, time_opt),
                       max=(batch_max, 80, time_max))

    config.add_optimization_profile(profile)

    print("Building TensorRT engine (this may take 5-15 minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    print(f"Writing engine to: {output_path}")
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    # Print engine size
    size_mb = len(serialized_engine) / (1024 * 1024)
    print(f"Engine size: {size_mb:.1f} MB")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to ONNX model file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output engine file path (default: auto-named with GPU info)")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable FP16 optimization")
    parser.add_argument("--batch-min", type=int, default=2,
                        help="Minimum batch size (default: 2, for CFG doubling)")
    parser.add_argument("--batch-opt", type=int, default=2,
                        help="Optimal batch size (default: 2)")
    parser.add_argument("--batch-max", type=int, default=20,
                        help="Maximum batch size (default: 20, 2 x max_batch_size=10)")
    parser.add_argument("--time-min", type=int, default=100,
                        help="Minimum time dimension (default: 100)")
    parser.add_argument("--time-opt", type=int, default=400,
                        help="Optimal time dimension (default: 400)")
    parser.add_argument("--time-max", type=int, default=800,
                        help="Maximum time dimension (default: 800)")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        gpu_info = get_gpu_info()
        output_path = f"conditional_decoder_{gpu_info}.engine"
        print(f"Auto-named output: {output_path}")

    build_engine(
        onnx_path=args.onnx,
        output_path=output_path,
        fp16=not args.no_fp16,
        batch_min=args.batch_min,
        batch_opt=args.batch_opt,
        batch_max=args.batch_max,
        time_min=args.time_min,
        time_opt=args.time_opt,
        time_max=args.time_max,
    )


if __name__ == "__main__":
    main()
