"""Shared constants and path helpers for DA3 TensorRT export pipeline."""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Fixed input resolution (both dimensions must be divisible by PATCH_SIZE)
# ---------------------------------------------------------------------------
PATCH_SIZE = 14

# Default: 518x518 (square, backward compatible).
# For non-square aspect ratios (e.g. 16:9 cameras), set IMG_H < IMG_W.
# Both must be divisible by PATCH_SIZE (14).
# Examples for 16:9: 294x518 (fast), 378x672 (balanced), 504x896 (high-res).
IMG_H = 518
IMG_W = 518

PATCH_DIM_H = IMG_H // PATCH_SIZE  # 37 for default
PATCH_DIM_W = IMG_W // PATCH_SIZE  # 37 for default
NUM_PATCHES = PATCH_DIM_H * PATCH_DIM_W  # 1369 for default

# ---------------------------------------------------------------------------
# Model architecture (DA3-Giant / vitg with cat_token=True)
# ---------------------------------------------------------------------------
EMBED_DIM = 1536
CONCAT_DIM = 2 * EMBED_DIM  # 3072 — local + global stream concatenation
NUM_HEADS = 24
HEAD_DIM = EMBED_DIM // NUM_HEADS  # 64
DEPTH = 40  # total transformer blocks

ALT_START = 13  # blocks >= ALT_START alternate local/global attention
ROPE_START = 13
QKNORM_START = 13
ROPE_FREQ = 100.0
ROPE_MAX_POS = 64

# Token layout: [CLS, patch0, patch1, ...] — vitg has 0 register tokens
NUM_REGISTER_TOKENS = 0
PATCH_START_IDX = 1  # CLS only
TOKENS_PER_FRAME = NUM_PATCHES + PATCH_START_IDX  # 1370 for default

# Intermediate layers extracted by the backbone for the DualDPT head
OUT_LAYERS = [19, 27, 33, 39]

# ---------------------------------------------------------------------------
# DualDPT head settings (Giant)
# ---------------------------------------------------------------------------
DPT_FEATURES = 256
DPT_OUT_CHANNELS = (256, 512, 1024, 1024)

# ---------------------------------------------------------------------------
# Metric backbone (DA3METRIC-LARGE / vitl, cat_token=False)
# Used by the nested variant DA3NESTED-GIANT-LARGE.
# ---------------------------------------------------------------------------
METRIC_EMBED_DIM = 1024
METRIC_NUM_HEADS = 16
METRIC_HEAD_DIM = METRIC_EMBED_DIM // METRIC_NUM_HEADS  # 64
METRIC_DEPTH = 24  # total transformer blocks for vitl
METRIC_OUT_LAYERS = [4, 11, 17, 23]
METRIC_PATCH_START_IDX = 1  # CLS only, no register tokens

# Metric DPT head settings
METRIC_DPT_DIM_IN = 1024
METRIC_DPT_FEATURES = 256
METRIC_DPT_OUT_CHANNELS = (256, 512, 1024, 1024)
METRIC_COMPONENTS = ["metric_backbone", "metric_head"]

# ---------------------------------------------------------------------------
# Floor segmentation head
# ---------------------------------------------------------------------------
SEG_LAYER_IDX = 19
SEG_DIM_IN = CONCAT_DIM  # 3072

# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------
DEFAULT_NUM_FRAMES = 4
BATCH_SIZE = 1

# ---------------------------------------------------------------------------
# Normalization (ImageNet statistics, same as DA3 backbone expects)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Export / TRT build settings
# ---------------------------------------------------------------------------
ONNX_OPSET = 17
TRT_WORKSPACE_GB = 8
TRT_FP16 = True
COMPONENTS = ["backbone", "depth_head", "floor_seg_head"]


def patch_dims(img_h: int, img_w: int) -> tuple[int, int, int]:
    """Compute patch grid dimensions for a given image resolution.

    Returns:
        (patch_dim_h, patch_dim_w, num_patches)
    """
    assert img_h % PATCH_SIZE == 0, f"IMG_H={img_h} not divisible by {PATCH_SIZE}"
    assert img_w % PATCH_SIZE == 0, f"IMG_W={img_w} not divisible by {PATCH_SIZE}"
    ph = img_h // PATCH_SIZE
    pw = img_w // PATCH_SIZE
    return ph, pw, ph * pw


def get_trt_version() -> str:
    """Return the installed TensorRT version string."""
    import tensorrt
    return tensorrt.__version__


def get_engine_dir(checkpoint_root: str, num_frames: int, trt_version: str | None = None) -> str:
    if trt_version is None:
        trt_version = get_trt_version()
    return os.path.join(checkpoint_root, "da3_trt", f"engines_{trt_version}_{num_frames}")


def get_onnx_dir(checkpoint_root: str, num_frames: int) -> str:
    return os.path.join(checkpoint_root, "da3_trt", f"onnx_{num_frames}")
