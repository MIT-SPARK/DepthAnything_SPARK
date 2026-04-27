"""
ONNX export functions for all DA3 TRT components.

Usage as library:
    from depth_anything_3.trt.export import export_all
    export_all(backbone, head, seg_head, output_dir, num_frames=4)
"""
from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn

from depth_anything_3.trt.config import (
    BATCH_SIZE,
    CONCAT_DIM,
    IMG_H,
    IMG_W,
    METRIC_EMBED_DIM,
    ONNX_OPSET,
    patch_dims,
)
from depth_anything_3.trt.wrappers.backbone_wrapper import DA3BackboneFixed
from depth_anything_3.trt.wrappers.dpt_wrapper import DPTFixed
from depth_anything_3.trt.wrappers.dualdpt_wrapper import DualDPTFixed
from depth_anything_3.trt.wrappers.metric_backbone_wrapper import DA3MetricBackboneFixed


def _validate(name: str, eager: torch.Tensor, wrapped: torch.Tensor) -> None:
    max_abs = (eager - wrapped).abs().max().item()
    rel_err = ((eager - wrapped).abs() / (eager.abs() + 1e-8)).mean().item()
    ok = max_abs < 0.05 and rel_err < 0.01
    tag = "PASS" if ok else "WARN"
    print(f"  [{tag}] {name}: max_abs={max_abs:.6f}  mean_rel={rel_err:.6f}")


def export_backbone(
    backbone: nn.Module,
    output_dir: str,
    num_frames: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    img_h: int = IMG_H,
    img_w: int = IMG_W,
) -> None:
    """Export backbone to ONNX."""
    print("\n=== Exporting DA3 Backbone ===")
    _, _, num_patches = patch_dims(img_h, img_w)
    wrapper = DA3BackboneFixed(backbone, num_frames=num_frames, img_h=img_h, img_w=img_w).to(device)
    wrapper.eval()

    B = BATCH_SIZE
    S = num_frames
    dummy = torch.randn(B, S, 3, img_h, img_w, device=device)

    # Deepcopy + cast for export
    export_wrapper = copy.deepcopy(wrapper).to(dtype)
    export_dummy = dummy.to(dtype)

    onnx_path = os.path.join(output_dir, "backbone.onnx")
    size_limit = 2 * 1024 * 1024 * 1024  # 2 GB

    # ViT-Giant may exceed 2 GB protobuf limit → use external data
    print(f"Exporting to {onnx_path} (dtype={dtype}, resolution={img_h}x{img_w})...")
    param_bytes = sum(p.numel() * p.element_size() for p in export_wrapper.parameters())
    use_external = param_bytes > size_limit

    if use_external:
        backbone_dir = os.path.join(output_dir, "backbone_data")
        os.makedirs(backbone_dir, exist_ok=True)
        onnx_path = os.path.join(backbone_dir, "backbone.onnx")
        print(f"  Model is {param_bytes / 1e9:.1f} GB — using external data")

    with torch.no_grad():
        torch.onnx.export(
            export_wrapper,
            (export_dummy,),
            onnx_path,
            input_names=["images"],
            output_names=["features", "aux_tokens", "camera_tokens"],
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
        )

    print(f"  Saved: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
    del export_wrapper


def export_depth_head(
    head: nn.Module,
    output_dir: str,
    num_frames: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    img_h: int = IMG_H,
    img_w: int = IMG_W,
) -> None:
    """Export DualDPT head to ONNX."""
    print("\n=== Exporting DA3 Depth Head ===")
    _, _, num_patches = patch_dims(img_h, img_w)
    wrapper = DualDPTFixed(head, num_frames=num_frames, img_h=img_h, img_w=img_w).to(device)
    wrapper.eval()

    BS = BATCH_SIZE * num_frames
    dummy_feats = [
        torch.randn(BS, num_patches, CONCAT_DIM, device=device) for _ in range(4)
    ]

    export_wrapper = copy.deepcopy(wrapper).to(dtype)
    # Restore fp32 weights in aux output conv (_FP32Sequential) — the blanket
    # .to(dtype) converted them to fp16 but the 128→32 conv overflows in fp16.
    from depth_anything_3.trt.wrappers.dualdpt_wrapper import _FP32Sequential
    for seq in export_wrapper.head.scratch.output_conv2_aux:
        if isinstance(seq, _FP32Sequential):
            seq.restore_fp32()
    export_feats = [f.to(dtype) for f in dummy_feats]

    onnx_path = os.path.join(output_dir, "depth_head.onnx")
    print(f"Exporting to {onnx_path} (dtype={dtype}, resolution={img_h}x{img_w})...")

    with torch.no_grad():
        torch.onnx.export(
            export_wrapper,
            tuple(export_feats),
            onnx_path,
            input_names=["feat_0", "feat_1", "feat_2", "feat_3"],
            output_names=["depth", "depth_conf", "ray", "ray_conf"],
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
        )

    print(f"  Saved: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
    del export_wrapper


def export_floor_seg(
    seg_head: nn.Module,
    output_dir: str,
    num_frames: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    img_h: int = IMG_H,
    img_w: int = IMG_W,
) -> None:
    """Export FloorSegHead to ONNX.

    The floor seg head is nearly ONNX-compatible as-is — the only fix is ensuring
    a fixed output size rather than dynamic (H, W) from the images tensor.
    """
    print("\n=== Exporting DA3 Floor Seg Head ===")
    _, _, num_patches = patch_dims(img_h, img_w)
    seg_head = copy.deepcopy(seg_head).float().to(device)
    seg_head.eval()

    B, S = BATCH_SIZE, num_frames
    dummy_tokens = torch.randn(B, S, num_patches, CONCAT_DIM, device=device)
    dummy_images = torch.randn(B, S, 3, img_h, img_w, device=device)

    onnx_path = os.path.join(output_dir, "floor_seg_head.onnx")
    print(f"Exporting to {onnx_path} (dtype=float32, head stays fp32)...")

    with torch.no_grad():
        torch.onnx.export(
            seg_head,
            (dummy_tokens, dummy_images),
            onnx_path,
            input_names=["tokens", "images"],
            output_names=["floor_logits"],
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
        )

    print(f"  Saved: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")


def export_metric_backbone(
    backbone: nn.Module,
    output_dir: str,
    num_frames: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    img_h: int = IMG_H,
    img_w: int = IMG_W,
) -> None:
    """Export metric ViT-Large backbone to ONNX."""
    print("\n=== Exporting DA3 Metric Backbone (ViT-Large) ===")
    wrapper = DA3MetricBackboneFixed(backbone, num_frames=num_frames, img_h=img_h, img_w=img_w).to(device)
    wrapper.eval()

    B = BATCH_SIZE
    S = num_frames
    dummy = torch.randn(B, S, 3, img_h, img_w, device=device)

    export_wrapper = copy.deepcopy(wrapper).to(dtype)
    export_dummy = dummy.to(dtype)

    onnx_path = os.path.join(output_dir, "metric_backbone.onnx")
    size_limit = 2 * 1024 * 1024 * 1024

    print(f"Exporting to {onnx_path} (dtype={dtype}, resolution={img_h}x{img_w})...")
    param_bytes = sum(p.numel() * p.element_size() for p in export_wrapper.parameters())
    use_external = param_bytes > size_limit

    if use_external:
        backbone_dir = os.path.join(output_dir, "metric_backbone_data")
        os.makedirs(backbone_dir, exist_ok=True)
        onnx_path = os.path.join(backbone_dir, "metric_backbone.onnx")
        print(f"  Model is {param_bytes / 1e9:.1f} GB — using external data")

    with torch.no_grad():
        torch.onnx.export(
            export_wrapper,
            (export_dummy,),
            onnx_path,
            input_names=["images"],
            output_names=["features"],
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
        )

    print(f"  Saved: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
    del export_wrapper


def export_metric_head(
    head: nn.Module,
    output_dir: str,
    num_frames: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    img_h: int = IMG_H,
    img_w: int = IMG_W,
) -> None:
    """Export metric DPT head (depth + sky) to ONNX."""
    print("\n=== Exporting DA3 Metric Head (DPT) ===")
    _, _, num_patches = patch_dims(img_h, img_w)
    wrapper = DPTFixed(head, num_frames=num_frames, img_h=img_h, img_w=img_w).to(device)
    wrapper.eval()

    BS = BATCH_SIZE * num_frames
    dummy_feats = [
        torch.randn(BS, num_patches, METRIC_EMBED_DIM, device=device) for _ in range(4)
    ]

    export_wrapper = copy.deepcopy(wrapper).to(dtype)
    export_feats = [f.to(dtype) for f in dummy_feats]

    onnx_path = os.path.join(output_dir, "metric_head.onnx")
    print(f"Exporting to {onnx_path} (dtype={dtype}, resolution={img_h}x{img_w})...")

    with torch.no_grad():
        torch.onnx.export(
            export_wrapper,
            tuple(export_feats),
            onnx_path,
            input_names=["feat_0", "feat_1", "feat_2", "feat_3"],
            output_names=["depth", "sky"],
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
        )

    print(f"  Saved: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
    del export_wrapper


def export_all(
    backbone: nn.Module,
    head: nn.Module,
    seg_head: nn.Module | None,
    output_dir: str,
    num_frames: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    components: list[str] | None = None,
    metric_backbone: nn.Module | None = None,
    metric_head: nn.Module | None = None,
    img_h: int = IMG_H,
    img_w: int = IMG_W,
) -> None:
    """Export all (or selected) DA3 components to ONNX.

    Args:
        backbone: ``DinoV2`` backbone module (Giant).
        head: ``DualDPT`` depth head module.
        seg_head: ``FloorSegHead`` module (optional, skipped if None).
        output_dir: Directory for ONNX files.
        num_frames: Fixed sequence length ``S``.
        dtype: Export precision (float16 recommended).
        components: Subset of component names. None = all available.
        metric_backbone: ``DinoV2`` backbone module for the metric branch (ViT-Large).
        metric_head: ``DPT`` head module for the metric branch.
        img_h: Input image height (must be divisible by 14).
        img_w: Input image width (must be divisible by 14).
    """
    os.makedirs(output_dir, exist_ok=True)
    all_components = components or [
        "backbone", "depth_head", "floor_seg_head",
        "metric_backbone", "metric_head",
    ]

    if "backbone" in all_components:
        export_backbone(backbone, output_dir, num_frames, device, dtype, img_h, img_w)

    if "depth_head" in all_components:
        export_depth_head(head, output_dir, num_frames, device, dtype, img_h, img_w)

    if "floor_seg_head" in all_components and seg_head is not None:
        export_floor_seg(seg_head, output_dir, num_frames, device, dtype, img_h, img_w)

    if "metric_backbone" in all_components and metric_backbone is not None:
        export_metric_backbone(metric_backbone, output_dir, num_frames, device, dtype, img_h, img_w)

    if "metric_head" in all_components and metric_head is not None:
        export_metric_head(metric_head, output_dir, num_frames, device, dtype, img_h, img_w)

    print(f"\n=== ONNX export complete → {output_dir} ===")
