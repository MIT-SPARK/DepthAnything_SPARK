"""Quick diagnostic: check backbone TRT engine outputs for NaN/Inf."""
from __future__ import annotations

import sys
import torch
from depth_anything_3.trt.engine import TRTEngine
from depth_anything_3.trt.config import IMAGENET_MEAN, IMAGENET_STD

engine_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/da3_trt/engines_10.16.0.72_32"
device = "cuda"

print(f"Loading backbone engine from {engine_dir}/backbone.engine")
backbone = TRTEngine(f"{engine_dir}/backbone.engine")

print(f"Input names: {backbone.input_names}")
print(f"Input dtypes: {backbone.input_dtypes}")
print(f"Output names: {backbone.output_names}")
print(f"Output shapes: {backbone.output_shapes}")
print(f"Output dtypes: {backbone.output_dtypes}")

# Create random input images (normalized)
images = torch.rand(1, 32, 3, 518, 518, device=device)
mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
images_norm = ((images - mean) / std).half()

print(f"\nInput dtype: {images_norm.dtype}, shape: {images_norm.shape}")
print(f"Input range: [{images_norm.min():.4f}, {images_norm.max():.4f}]")

outputs = backbone(images_norm)

for name, tensor in outputs.items():
    nan_pct = tensor.isnan().float().mean().item() * 100
    inf_pct = tensor.isinf().float().mean().item() * 100
    finite = tensor[tensor.isfinite()]
    if finite.numel() > 0:
        print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
              f"NaN={nan_pct:.1f}%, Inf={inf_pct:.1f}%, "
              f"range=[{finite.min():.4f}, {finite.max():.4f}], "
              f"mean={finite.mean():.4f}, std={finite.std():.4f}")
    else:
        print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
              f"NaN={nan_pct:.1f}%, Inf={inf_pct:.1f}% — ALL NON-FINITE")

# Detailed per-layer and per-half analysis for features
feat = outputs["features"]  # [4, B, S, N_patch, 3072]
print("\nPer-layer feature analysis:")
for layer_idx in range(4):
    layer = feat[layer_idx]  # [B, S, N_patch, 3072]
    local_half = layer[..., :1536]
    global_half = layer[..., 1536:]
    local_nan = local_half.isnan().float().mean().item() * 100
    global_nan = global_half.isnan().float().mean().item() * 100
    print(f"  Layer {layer_idx}: local_NaN={local_nan:.1f}%, global_NaN={global_nan:.1f}%")

    if local_nan < 100:
        lf = local_half[local_half.isfinite()]
        print(f"    local:  range=[{lf.min():.4f}, {lf.max():.4f}], std={lf.std():.4f}")
    if global_nan < 100:
        gf = global_half[global_half.isfinite()]
        print(f"    global: range=[{gf.min():.4f}, {gf.max():.4f}], std={gf.std():.4f}")

# Check camera tokens per-half
cam = outputs["camera_tokens"]  # [B, S, 3072]
cam_local = cam[..., :1536]
cam_global = cam[..., 1536:]
print(f"\nCamera tokens: local_NaN={cam_local.isnan().float().mean().item()*100:.1f}%, "
      f"global_NaN={cam_global.isnan().float().mean().item()*100:.1f}%")

# ---- Depth head diagnostic ----
import os
from einops import rearrange

depth_head_path = f"{engine_dir}/depth_head.engine"
if os.path.exists(depth_head_path):
    print(f"\n{'='*60}")
    print("DEPTH HEAD DIAGNOSTIC")
    print(f"{'='*60}")
    depth_head = TRTEngine(depth_head_path)
    print(f"Input names: {depth_head.input_names}")
    print(f"Input dtypes: {depth_head.input_dtypes}")
    print(f"Output names: {depth_head.output_names}")
    print(f"Output shapes: {depth_head.output_shapes}")
    print(f"Output dtypes: {depth_head.output_dtypes}")

    feat = outputs["features"]  # [4, B, S, N_patch, 3072]
    feat_layers = []
    for i in range(4):
        fi = feat[i]
        fi = rearrange(fi, "b s n c -> (b s) n c").contiguous()
        feat_layers.append(fi)
        print(f"  feat_layer[{i}]: shape={tuple(fi.shape)}, dtype={fi.dtype}, "
              f"NaN={fi.isnan().float().mean().item()*100:.1f}%")

    depth_out = depth_head(*feat_layers)
    for name, tensor in depth_out.items():
        nan_pct = tensor.isnan().float().mean().item() * 100
        finite = tensor[tensor.isfinite()]
        if finite.numel() > 0:
            print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                  f"NaN={nan_pct:.1f}%, range=[{finite.min():.4f}, {finite.max():.4f}], "
                  f"std={finite.std():.4f}")
        else:
            nan_count = tensor.isnan().sum().item()
            inf_count = tensor.isinf().sum().item()
            print(f"  {name}: shape={tuple(tensor.shape)}, ALL NON-FINITE "
                  f"(NaN={nan_count}, Inf={inf_count}, total={tensor.numel()})")
