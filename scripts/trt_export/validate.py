"""
End-to-end validation and benchmarking of DA3 TensorRT export.

Usage:
    # Test wrappers only (no TRT required):
    python Depth-Anything-3/scripts/trt_export/validate.py --mode wrappers \
        --model-name depth-anything/DA3-GIANT --image-dir examples/kitchen/images/

    # Test full TRT pipeline against eager model:
    python Depth-Anything-3/scripts/trt_export/validate.py --mode trt \
        --engine-dir checkpoints/da3_trt/engines_10.16.0.72_17 \
        --image-dir examples/kitchen/images/

    # Benchmark eager BF16 vs TRT FP16:
    python Depth-Anything-3/scripts/trt_export/validate.py --mode benchmark \
        --engine-dir checkpoints/da3_trt/engines_10.16.0.72_17 \
        --image-dir examples/kitchen/images/

    # Run all validation modes:
    python Depth-Anything-3/scripts/trt_export/validate.py --mode all \
        --engine-dir checkpoints/da3_trt/engines_10.16.0.72_17 \
        --image-dir examples/kitchen/images/

    # Validate nested model:
    python Depth-Anything-3/scripts/trt_export/validate.py --mode trt \
        --model-name depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
        --engine-dir checkpoints/da3_trt/engines_10.16.0.72_17
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.floor_seg_head import FloorSegHead
from depth_anything_3.trt import DA3TensorRT
from depth_anything_3.trt.config import (
    BATCH_SIZE,
    CONCAT_DIM,
    DEFAULT_NUM_FRAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_H,
    IMG_W,
    patch_dims,
)
from depth_anything_3.trt.wrappers import DA3BackboneFixed, DualDPTFixed


def load_real_images(image_dir: str, num_frames: int = DEFAULT_NUM_FRAMES) -> torch.Tensor:
    """Load images using DA3's InputProcessor for exact preprocessing match.

    Returns images in [0, 1] range (denormalized) so both eager and TRT
    pipelines can apply their own normalization consistently.
    """
    from depth_anything_3.utils.io.input_processor import InputProcessor

    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
    )
    assert image_paths, f"No images found in {image_dir}"
    image_paths = image_paths[:num_frames]
    print(f"Loading {len(image_paths)} images from {image_dir}")

    # Use the larger engine dimension as process_res — InputProcessor will
    # resize to fit while preserving aspect ratio and patch-size alignment.
    process_res = max(IMG_H, IMG_W)
    processor = InputProcessor()
    imgs_cpu, _, _ = processor(image_paths, None, None, process_res, "upper_bound_resize")
    # imgs_cpu: [N, 3, H, W] ImageNet-normalized. Denormalize to [0, 1].
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    images_01 = imgs_cpu * std + mean
    images = images_01.unsqueeze(0).contiguous()  # [1, S, 3, H, W]
    print(f"  Image tensor shape: {images.shape}")
    return images


def create_synthetic_images(num_frames: int = DEFAULT_NUM_FRAMES, device: str = "cpu") -> torch.Tensor:
    return torch.rand(1, num_frames, 3, IMG_H, IMG_W, device=device).contiguous()


def compute_metrics(name: str, eager: torch.Tensor, test: torch.Tensor) -> dict:
    eager = eager.float().cpu()
    test = test.float().cpu()
    abs_err = (eager - test).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    denom = eager.abs().clamp(min=1e-8)
    rel_err = abs_err / denom
    mean_rel = rel_err.mean().item() * 100
    cos_sim = float("nan")
    if eager.dim() >= 2:
        cos_sim = F.cosine_similarity(
            eager.flatten(0, -2), test.flatten(0, -2), dim=-1,
        ).mean().item()

    if mean_rel < 1.0:
        status = "PASS"
    elif cos_sim > 0.999 and mean_abs < 0.01:
        status = "PASS"
    elif mean_rel < 5.0:
        status = "WARN"
    else:
        status = "FAIL" if cos_sim < 0.99 else "WARN"

    print(f"  [{status}] {name}: mean_abs={mean_abs:.6f}, mean_rel={mean_rel:.4f}%, "
          f"max_abs={max_abs:.6f}, cos_sim={cos_sim:.6f}")
    return {"max_abs": max_abs, "mean_abs": mean_abs, "mean_rel%": mean_rel, "cos_sim": cos_sim}


class DA3EagerWrapper:
    """Wrapper around eager DA3 for validation comparisons."""

    def __init__(self, model_name: str = "depth-anything/DA3-GIANT", device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(device)
        # Do NOT call .to(bfloat16) — DA3Net.forward() disables autocast
        # for the depth head (da3.py:139), which requires float32 weights
        # so F.layer_norm can cast bfloat16 inputs up automatically.
        # api.py's autocast handles mixed precision for the backbone.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        inner = self.model.model
        if hasattr(inner, "da3"):
            self.backbone = inner.da3.backbone
            self.head = inner.da3.head
        else:
            self.backbone = inner.backbone
            self.head = inner.head

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run eager DA3. Images must be [0, 1] range (same as TRT pipeline input)."""
        images = images.to(self.device)
        if images.dim() == 4:
            images = images.unsqueeze(0)
        # api.forward() feeds directly into the backbone which expects
        # ImageNet-normalized input.  Normalize here so both eager and TRT
        # pipelines receive identical preprocessing.
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 1, 3, 1, 1)
        images = (images - mean) / std
        return self.model(images, export_feat_layers=[], use_ray_pose=True, ref_view_strategy="first")


# ---------------------------------------------------------------------------
# Wrapper validation
# ---------------------------------------------------------------------------

def validate_wrappers(model_name: str, image_dir: str | None, device: str) -> None:
    import copy

    print("=" * 70)
    print("WRAPPER VALIDATION")
    print("=" * 70)

    print(f"\nLoading DA3 model: {model_name} ...")
    eager = DA3EagerWrapper(model_name, device)
    backbone = eager.backbone
    head = eager.head

    images = load_real_images(image_dir).to(device) if image_dir else create_synthetic_images(device=device)
    B, S = images.shape[:2]

    # Resize to wrapper resolution (wrappers have fixed pos embeds for IMG_H x IMG_W)
    img_h, img_w = IMG_H, IMG_W
    if images.shape[-2] != img_h or images.shape[-1] != img_w:
        images = F.interpolate(
            images.reshape(B * S, *images.shape[2:]),
            size=(img_h, img_w), mode="bilinear", align_corners=False,
        ).reshape(B, S, 3, img_h, img_w)

    _, _, num_patches = patch_dims(img_h, img_w)

    # Normalize images (same as backbone expects)
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.bfloat16).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.bfloat16).view(1, 1, 3, 1, 1)
    images_bf16 = images.to(torch.bfloat16)
    images_norm = (images_bf16 - mean) / std

    # -- Eager backbone --
    # The backbone has fp32 weights; DA3's api.py uses autocast(bf16) for
    # mixed-precision inference.  We must do the same here.
    print("\n--- Running eager backbone ---")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        eager_feats, eager_aux = backbone(
            images_norm, cam_token=None, export_feat_layers=[],
            ref_view_strategy="first",
        )

    # -- Wrapper backbone --
    # deepcopy so .to(fp16) doesn't mutate the original backbone used above
    print("\n--- Testing DA3BackboneFixed ---")
    wrapper_bb = copy.deepcopy(
        DA3BackboneFixed(backbone, num_frames=S, img_h=img_h, img_w=img_w)
    ).to(device).to(torch.float16).eval()
    with torch.no_grad():
        wrapper_input = images_norm.to(torch.float16)
        w_feats_stacked, w_aux, w_cam = wrapper_bb(wrapper_input)

    # Compare features at output layers
    for i in range(4):
        eager_feat_i = eager_feats[i]
        # eager_feat_i is (patch_features, camera_tokens) — patch_features
        # already have local+global concatenated (dim=3072).
        if isinstance(eager_feat_i, (list, tuple)):
            eager_cat = eager_feat_i[0]  # [B, S, N_patches, 3072]
        else:
            eager_cat = eager_feat_i
        # Strip CLS if present
        if eager_cat.shape[-2] > num_patches:
            eager_cat = eager_cat[..., 1:, :]
        wrapper_feat_i = w_feats_stacked[i]
        compute_metrics(f"Backbone feat[{i}]", eager_cat.float(), wrapper_feat_i.float())

    # -- Eager depth head (must run BEFORE DualDPTFixed, which mutates head in-place) --
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        eager_out = head(eager_feats, img_h, img_w, patch_start_idx=0)

    # -- Wrapper depth head --
    # deepcopy: DualDPTFixed mutates head in-place (monkey-patches, wraps submodules),
    # and we need fp16 weights matching the ONNX export path.
    print("\n--- Testing DualDPTFixed ---")
    from depth_anything_3.trt.wrappers.dualdpt_wrapper import _FP32Sequential
    wrapper_dpt = copy.deepcopy(
        DualDPTFixed(head, num_frames=S, img_h=img_h, img_w=img_w)
    ).to(device).to(torch.float16).eval()
    # Restore fp32 in aux output conv (mirrors what export_depth_head does)
    for seq in wrapper_dpt.head.scratch.output_conv2_aux:
        if isinstance(seq, _FP32Sequential):
            seq.restore_fp32()

    # Use wrapper backbone features as input (what TRT would use)
    with torch.no_grad():
        feat_layers = []
        for i in range(4):
            feat_i = w_feats_stacked[i]  # [B, S, N_patch, C]
            feat_i = rearrange(feat_i, "b s n c -> (b s) n c").contiguous()
            feat_layers.append(feat_i.to(torch.float16))
        w_depth, w_depth_conf, w_ray, w_ray_conf = wrapper_dpt(*feat_layers)

    for key in ("depth", "depth_conf"):
        head_name = head.head_main if key == "depth" else f"{head.head_main}_conf"
        if head_name in eager_out:
            eager_val = eager_out[head_name].float()
            wrapper_val = (w_depth if key == "depth" else w_depth_conf).float()
            # Reshape to match
            if eager_val.dim() != wrapper_val.dim():
                eager_val = eager_val.reshape(wrapper_val.shape)
            compute_metrics(f"DualDPT {key}", eager_val, wrapper_val)

    print("\n" + "=" * 70)
    print("WRAPPER VALIDATION COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# TRT pipeline validation
# ---------------------------------------------------------------------------

_NESTED_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
_DEFAULT_MODEL = "depth-anything/DA3-GIANT"


def _match_model_to_engines(trt_pipeline: DA3TensorRT, model_name: str) -> str:
    """Auto-detect nested TRT engines and switch eager model name to match."""
    if trt_pipeline._nested and model_name == _DEFAULT_MODEL:
        print(f"  Nested TRT engines detected — switching eager model to {_NESTED_MODEL}")
        return _NESTED_MODEL
    if not trt_pipeline._nested and model_name == _NESTED_MODEL:
        print(f"  WARNING: TRT engines are non-nested but eager model is nested")
    return model_name


def validate_trt_pipeline(
    engine_dir: str, model_name: str, image_dir: str | None, device: str,
) -> None:
    print("=" * 70)
    print("TRT PIPELINE VALIDATION")
    print("=" * 70)

    trt_pipeline = DA3TensorRT(engine_dir, device=device)
    model_name = _match_model_to_engines(trt_pipeline, model_name)
    eager_pipeline = DA3EagerWrapper(model_name, device=device)

    S = trt_pipeline.num_frames or DEFAULT_NUM_FRAMES
    images = load_real_images(image_dir, S).to(device) if image_dir else create_synthetic_images(S, device=device)

    print(f"\nInput: {images.shape}")

    print("\n--- Running eager model ---")
    eager_pred = eager_pipeline(images)
    print("\n--- Running TRT pipeline ---")
    trt_pred = trt_pipeline(images)

    print("\n--- Comparing outputs ---")
    for key in ("depth", "depth_conf", "extrinsics", "intrinsics"):
        if key in eager_pred and key in trt_pred:
            compute_metrics(key, eager_pred[key].float(), trt_pred[key].float())

    print("\n" + "=" * 70)
    print("TRT PIPELINE VALIDATION COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(
    engine_dir: str | None,
    model_name: str,
    device: str,
    warmup: int = 10,
    iterations: int = 100,
    num_frames: int | None = None,
    image_dir: str | None = None,
) -> None:
    print("=" * 70)
    print("BENCHMARK")
    print("=" * 70)

    # Auto-detect S from engine dir
    if num_frames is None and engine_dir:
        from depth_anything_3.trt.pipeline import _detect_num_frames
        num_frames = _detect_num_frames(engine_dir)
    if num_frames is None:
        num_frames = DEFAULT_NUM_FRAMES
    print(f"  Using S={num_frames}")

    if image_dir:
        images = load_real_images(image_dir, num_frames).to(device).contiguous()
    else:
        images = create_synthetic_images(num_frames, device=device).contiguous()

    # Load TRT pipeline once (reused for model detection and benchmark)
    trt_pipe = None
    if engine_dir and os.path.exists(engine_dir):
        trt_pipe = DA3TensorRT(engine_dir, device=device)
        model_name = _match_model_to_engines(trt_pipe, model_name)

    # Eager benchmark
    print(f"\n--- Benchmarking Eager BF16 ---")
    eager = DA3EagerWrapper(model_name, device=device)
    for _ in range(warmup):
        eager(images)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iterations):
        eager(images)
    torch.cuda.synchronize()
    eager_ms = (time.time() - t0) / iterations * 1000
    print(f"  Eager BF16: {eager_ms:.1f} ms/inference")

    # TRT benchmark
    if trt_pipe is not None:
        print("\n--- Benchmarking TRT pipeline ---")
        trt_pipe._detailed_timing = True
        for _ in range(warmup):
            trt_pipe(images)
        # Reset timers after warmup
        for k in trt_pipe._timings:
            trt_pipe._timings[k] = 0.0 if isinstance(trt_pipe._timings[k], float) else 0
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iterations):
            trt_pipe(images)
        torch.cuda.synchronize()
        trt_ms = (time.time() - t0) / iterations * 1000
        print(f"  TRT FP16: {trt_ms:.1f} ms/inference")
        print(f"  Speedup vs Eager BF16: {eager_ms / trt_ms:.1f}x")
        trt_pipe.print_component_timing()

        # Compare outputs
        print("\n--- Comparing outputs (eager vs TRT) ---")
        eager_pred = eager(images)
        trt_pred = trt_pipe(images)
        for key in ("depth", "depth_conf", "extrinsics"):
            if key in eager_pred and key in trt_pred:
                compute_metrics(key, eager_pred[key].float(), trt_pred[key].float())

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and benchmark DA3 TRT export")
    parser.add_argument("--mode", choices=["wrappers", "trt", "benchmark", "all"], default="wrappers")
    parser.add_argument("--model-name", type=str, default="depth-anything/DA3-GIANT")
    parser.add_argument("--engine-dir", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Number of frames (S) for benchmark. Auto-detected from engine dir if not set.")
    args = parser.parse_args()

    if args.mode in ("wrappers", "all"):
        validate_wrappers(args.model_name, args.image_dir, args.device)
    if args.mode in ("trt", "all"):
        assert args.engine_dir, "--engine-dir required for TRT validation"
        validate_trt_pipeline(args.engine_dir, args.model_name, args.image_dir, args.device)
    if args.mode in ("benchmark", "all"):
        benchmark(args.engine_dir, args.model_name, args.device,
                  num_frames=args.num_frames, image_dir=args.image_dir)


if __name__ == "__main__":
    main()
