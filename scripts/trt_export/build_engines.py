"""
Convert DA3 ONNX models to TensorRT .engine files.

Usage:
    # Build engines for DA3-GIANT (17 frames for SLAM submap):
    python Depth-Anything-3/scripts/trt_export/build_engines.py --num-frames 17

    # Build engines including nested metric branch (if ONNX files present):
    python Depth-Anything-3/scripts/trt_export/build_engines.py --num-frames 17 \
        --components backbone depth_head floor_seg_head metric_backbone metric_head

    # Custom paths:
    python Depth-Anything-3/scripts/trt_export/build_engines.py \
        --onnx-dir checkpoints/da3_trt/onnx_17 \
        --engine-dir checkpoints/da3_trt/engines_10.16.0.72_17 \
        --fp16

    # Use trtexec CLI instead of Python API:
    python Depth-Anything-3/scripts/trt_export/build_engines.py --num-frames 17 --use-trtexec
"""
from __future__ import annotations

import argparse

from depth_anything_3.trt.build import build_all
from depth_anything_3.trt.config import (
    COMPONENTS,
    DEFAULT_NUM_FRAMES,
    METRIC_COMPONENTS,
    TRT_FP16,
    TRT_WORKSPACE_GB,
    get_engine_dir,
    get_onnx_dir,
)

ALL_COMPONENTS = COMPONENTS + METRIC_COMPONENTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DA3 TensorRT engines from ONNX models")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints",
                        help="Root checkpoint directory")
    parser.add_argument("--onnx-dir", type=str, default=None,
                        help="Override ONNX input directory")
    parser.add_argument("--engine-dir", type=str, default=None,
                        help="Override engine output directory")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help=f"Number of frames S (default: {DEFAULT_NUM_FRAMES})")
    parser.add_argument("--fp16", action="store_true", default=TRT_FP16,
                        help="Enable FP16 precision")
    parser.add_argument("--workspace", type=int, default=TRT_WORKSPACE_GB,
                        help="TRT workspace size in GB")
    parser.add_argument("--components", nargs="+", default=ALL_COMPONENTS,
                        help=f"Components to build (default: {ALL_COMPONENTS})")
    parser.add_argument("--use-trtexec", action="store_true",
                        help="Use trtexec CLI instead of Python API")
    args = parser.parse_args()

    onnx_dir = args.onnx_dir or get_onnx_dir(args.checkpoint_root, args.num_frames)
    engine_dir = args.engine_dir or get_engine_dir(args.checkpoint_root, args.num_frames)

    print(f"ONNX dir:   {onnx_dir}")
    print(f"Engine dir: {engine_dir}")

    build_all(
        onnx_dir=onnx_dir,
        engine_dir=engine_dir,
        fp16=args.fp16,
        workspace_gb=args.workspace,
        components=list(args.components),
        use_trtexec=args.use_trtexec,
    )


if __name__ == "__main__":
    main()
