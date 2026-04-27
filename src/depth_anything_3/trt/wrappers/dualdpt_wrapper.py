"""
ONNX-exportable wrapper for the DA3 DualDPT head.

Fixes for TRT/ONNX compatibility:
  1. ``nn.quantized.FloatFunctional`` replaced with plain ``+``.
  2. ``custom_interpolate`` (INT_MAX overflow guard) replaced with ``F.interpolate``.
  3. LayerNorm wrapped to preserve fp16 dtype during tracing.
  4. ``_add_pos_embed`` patched: ``position_grid_to_embed`` hardcodes fp32
     (``make_sincos_pos_embed`` returns ``.float()``), which promotes fp16
     activations before ConvTranspose2d.  The patch casts ``pe`` back to
     input dtype.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_3.model.dualdpt import DualDPT
from depth_anything_3.trt.config import CONCAT_DIM, IMG_H, IMG_W, PATCH_SIZE


# ---------------------------------------------------------------------------
# Drop-in replacement for nn.quantized.FloatFunctional
# ---------------------------------------------------------------------------

class _PlainAdd(nn.Module):
    """``skip_add.add(a, b)`` → ``a + b``."""

    @staticmethod
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class _DtypePreservingLayerNorm(nn.Module):
    """LayerNorm wrapper that computes in fp32 and casts back to input dtype.

    Both input and weights are explicitly cast to fp32 before computing.
    This is necessary because:
      1. ONNX/TRT LayerNorm follows graph dtypes exactly (no internal upscale).
      2. Small normalized_shape (e.g. 32 in DualDPT aux head) makes variance
         computation in fp16 numerically unstable → NaN.
    """

    def __init__(self, ln: nn.LayerNorm) -> None:
        super().__init__()
        self.ln = ln

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.ln.normalized_shape,
            self.ln.weight.float() if self.ln.weight is not None else None,
            self.ln.bias.float() if self.ln.bias is not None else None,
            self.ln.eps,
        ).to(orig_dtype)


def _wrap_layernorms_dtype_preserving(module: nn.Module) -> None:
    """Recursively wrap all nn.LayerNorm with _DtypePreservingLayerNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, _DtypePreservingLayerNorm(child))
        else:
            _wrap_layernorms_dtype_preserving(child)


class _FP32Sequential(nn.Module):
    """Wrap an nn.Sequential to always compute in fp32.

    The aux head's ``output_conv2_aux`` contains Conv2d(128->32) whose
    128-channel reduction overflows fp16 max (65504).  Running the entire
    sequential in fp32 avoids the saturation.

    After the model is cast to fp16 via ``.to(fp16)``, call
    ``restore_fp32()`` to push the inner weights back to fp32 before
    ONNX tracing.
    """

    def __init__(self, seq: nn.Module) -> None:
        super().__init__()
        self.seq = seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return self.seq(x.float()).to(orig_dtype)

    def restore_fp32(self) -> None:
        """Restore inner weights to fp32 after a blanket ``.to(fp16)``."""
        self.seq.float()


def _replace_float_functional(module: nn.Module) -> None:
    """Recursively swap every ``FloatFunctional`` with ``_PlainAdd``."""
    for name, child in module.named_children():
        if isinstance(child, torch.ao.nn.quantized.FloatFunctional):
            setattr(module, name, _PlainAdd())
        else:
            _replace_float_functional(child)


# ---------------------------------------------------------------------------
# Patched custom_interpolate → plain F.interpolate
# ---------------------------------------------------------------------------

def _safe_interpolate(
    x: torch.Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        assert scale_factor is not None
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


def _add_pos_embed_fixed(self: nn.Module, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
    """Dtype-safe ``_add_pos_embed``.

    ``make_sincos_pos_embed`` hardcodes ``omega`` as fp32 while the UV grid
    inherits the model's dtype.  During fp16 ONNX tracing this creates an
    Einsum node with mixed fp16/fp32 inputs — invalid ONNX that TRT
    silently produces NaN for.  Computing the grid in fp32 avoids the
    type mismatch, then we cast back for the downstream conv layers.
    """
    from depth_anything_3.model.utils.head_utils import create_uv_grid, position_grid_to_embed

    pw, ph = x.shape[-1], x.shape[-2]
    # Grid and sincos embed must be fp32 to match make_sincos_pos_embed's
    # hardcoded fp32 omega — otherwise ONNX Einsum gets mixed types.
    pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=torch.float32, device=x.device)
    pe = position_grid_to_embed(pe, x.shape[1]) * ratio
    pe = pe.to(x.dtype)
    pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
    return x + pe


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class DualDPTFixed(nn.Module):
    """Thin wrapper that patches the eager ``DualDPT`` for ONNX export.

    Applies two surgical fixes:
      * Every ``nn.quantized.FloatFunctional`` → plain addition.
      * ``custom_interpolate`` → ``F.interpolate`` (no INT_MAX chunking).

    The module's ``forward`` delegates to the original ``_forward_impl`` with
    the interpolation function monkey-patched.

    Args:
        head: Eager ``DualDPT`` module (already loaded with weights).
        num_frames: Fixed sequence length ``S`` for this export.
        img_h: Input image height (must be divisible by 14).
        img_w: Input image width (must be divisible by 14).
    """

    def __init__(
        self,
        head: DualDPT,
        num_frames: int,
        img_h: int = IMG_H,
        img_w: int = IMG_W,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self._img_h = img_h
        self._img_w = img_w

        # Steal all sub-modules from the eager head
        self.head = head

        # Fix 1: replace FloatFunctional
        _replace_float_functional(self.head)

        # Fix 2: wrap ALL LayerNorms to preserve fp16 dtype during tracing.
        # The main norm (3072-dim) AND the LayerNorm(32) inside
        # output_conv2_aux must be wrapped — the latter promotes to fp32
        # during tracing, creating a type mismatch in the ONNX graph that
        # causes TRT to produce NaN for the aux (ray) outputs.
        _wrap_layernorms_dtype_preserving(self.head)

        # Fix 3: monkey-patch custom_interpolate at module level
        import depth_anything_3.model.dualdpt as _dualdpt_mod
        import depth_anything_3.model.dpt as _dpt_mod
        _dualdpt_mod.custom_interpolate = _safe_interpolate
        _dpt_mod.custom_interpolate = _safe_interpolate

        # Fix 4: patch _add_pos_embed to cast fp32 pos embeddings back to input dtype
        import types
        self.head._add_pos_embed = types.MethodType(_add_pos_embed_fixed, self.head)

        # Fix 5: wrap aux output conv to compute in fp32.
        # Conv2d(128→32) in output_conv2_aux overflows fp16 (128-channel
        # reduction saturates to ±65504), producing NaN in the downstream
        # LayerNorm(32).  Running the entire sequential in fp32 avoids this.
        for i, seq in enumerate(self.head.scratch.output_conv2_aux):
            self.head.scratch.output_conv2_aux[i] = _FP32Sequential(seq)

    def forward(
        self,
        feat_0: torch.Tensor,
        feat_1: torch.Tensor,
        feat_2: torch.Tensor,
        feat_3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_0 .. feat_3: ``[B*S, N_patch, concat_dim]`` — the 4 intermediate-
                layer features from the backbone (CLS already stripped).

        Returns:
            depth:      ``[B*S, H, W]``
            depth_conf: ``[B*S, H, W]``
            ray:        ``[B*S, H_aux, W_aux, 6]``
            ray_conf:   ``[B*S, H_aux, W_aux]``
        """
        feats = [feat_0, feat_1, feat_2, feat_3]
        out = self.head._forward_impl(feats, self._img_h, self._img_w, patch_start_idx=0)

        depth = out[self.head.head_main]
        depth_conf = out[f"{self.head.head_main}_conf"]
        ray = out[self.head.head_aux]
        ray_conf = out[f"{self.head.head_aux}_conf"]

        return depth, depth_conf, ray, ray_conf
