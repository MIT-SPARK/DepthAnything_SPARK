"""
Export DA3 components to ONNX.

Usage:
    # DA3-GIANT (backbone + depth_head, 17 frames for SLAM submap):
    python Depth-Anything-3/scripts/trt_export/export_to_onnx.py \
        --model-name depth-anything/DA3-GIANT \
        --num-frames 17

    # DA3-GIANT with floor segmentation head:
    python Depth-Anything-3/scripts/trt_export/export_to_onnx.py \
        --model-name depth-anything/DA3-GIANT \
        --seg-checkpoint checkpoints/floor_seg_da3/best.pt \
        --num-frames 17

    # DA3NESTED-GIANT-LARGE (exports Giant + metric ViT-Large branches):
    python Depth-Anything-3/scripts/trt_export/export_to_onnx.py \
        --model-name depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
        --num-frames 17

    # Export a single component:
    python Depth-Anything-3/scripts/trt_export/export_to_onnx.py \
        --model-name depth-anything/DA3-GIANT \
        --components backbone \
        --num-frames 17

    # FP32 export (for debugging):
    python Depth-Anything-3/scripts/trt_export/export_to_onnx.py \
        --model-name depth-anything/DA3-GIANT \
        --dtype fp32 --num-frames 17
"""
from __future__ import annotations

import argparse

import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.floor_seg_head import FloorSegHead
from depth_anything_3.trt.config import COMPONENTS, DEFAULT_NUM_FRAMES, IMG_H, IMG_W, METRIC_COMPONENTS, PATCH_SIZE, get_onnx_dir
from depth_anything_3.trt.export import export_all

_DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DA3 components to ONNX")
    parser.add_argument("--model-name", type=str, default="depth-anything/DA3-GIANT",
                        help="HuggingFace model name or local model name")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints",
                        help="Root checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: auto from checkpoint-root + num-frames)")
    parser.add_argument("--seg-checkpoint", type=str, default=None,
                        help="Path to FloorSegHead checkpoint (.pt)")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help=f"Number of frames S (default: {DEFAULT_NUM_FRAMES})")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                        help="Export dtype (default: fp16)")
    parser.add_argument("--img-h", type=int, default=IMG_H,
                        help=f"Input image height, must be divisible by {PATCH_SIZE} (default: {IMG_H})")
    parser.add_argument("--img-w", type=int, default=IMG_W,
                        help=f"Input image width, must be divisible by {PATCH_SIZE} (default: {IMG_W})")
    all_components = COMPONENTS + METRIC_COMPONENTS
    parser.add_argument("--components", nargs="+", default=all_components,
                        help=f"Components to export (default: {all_components})")
    args = parser.parse_args()

    assert args.img_h % PATCH_SIZE == 0, f"--img-h={args.img_h} not divisible by {PATCH_SIZE}"
    assert args.img_w % PATCH_SIZE == 0, f"--img-w={args.img_w} not divisible by {PATCH_SIZE}"

    output_dir = args.output_dir or get_onnx_dir(args.checkpoint_root, args.num_frames)
    dtype = _DTYPE_MAP[args.dtype]

    # Load DA3 model
    print(f"Loading DA3 model: {args.model_name} ...")
    da3 = DepthAnything3.from_pretrained(args.model_name)
    da3 = da3.to(args.device)
    da3.eval()
    for p in da3.parameters():
        p.requires_grad_(False)

    # Extract backbone and head from the model
    inner = da3.model
    is_nested = hasattr(inner, "da3_metric")
    metric_backbone = None
    metric_head = None

    if is_nested:
        print("Detected nested model (DA3NESTED) — will export metric branch too.")
        backbone = inner.da3.backbone
        head = inner.da3.head
        metric_backbone = inner.da3_metric.backbone
        metric_head = inner.da3_metric.head
    elif hasattr(inner, "da3"):
        backbone = inner.da3.backbone
        head = inner.da3.head
    else:
        backbone = inner.backbone
        head = inner.head

    # Load floor seg head if checkpoint provided
    seg_head = None
    if args.seg_checkpoint and "floor_seg_head" in args.components:
        print(f"Loading FloorSegHead from {args.seg_checkpoint} ...")
        ckpt = torch.load(args.seg_checkpoint, map_location=args.device)
        dim_in = ckpt.get("dim_in", 2 * backbone.pretrained.embed_dim)
        seg_head = FloorSegHead(dim_in=dim_in, patch_size=14).to(args.device)
        seg_head.load_state_dict(ckpt["seg_head_state_dict"])
        seg_head.eval()

    export_all(
        backbone=backbone,
        head=head,
        seg_head=seg_head,
        output_dir=output_dir,
        num_frames=args.num_frames,
        device=args.device,
        dtype=dtype,
        components=list(args.components),
        metric_backbone=metric_backbone,
        metric_head=metric_head,
        img_h=args.img_h,
        img_w=args.img_w,
    )


if __name__ == "__main__":
    main()
