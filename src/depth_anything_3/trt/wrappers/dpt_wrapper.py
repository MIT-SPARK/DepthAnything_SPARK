"""
ONNX-exportable wrapper for the DA3 DPT head (used by the metric branch).

Same fixes as DualDPTFixed:
  1. ``nn.quantized.FloatFunctional`` replaced with plain ``+``.
  2. ``custom_interpolate`` replaced with ``F.interpolate``.

Unlike DualDPT, the metric DPT head outputs:
  - depth: [B*S, H, W] (output_dim=1, no confidence)
  - sky:   [B*S, H, W] (use_sky_head=True)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from depth_anything_3.model.dpt import DPT
from depth_anything_3.trt.config import IMG_H, IMG_W
from depth_anything_3.trt.wrappers.dualdpt_wrapper import (
    _add_pos_embed_fixed,
    _replace_float_functional,
    _safe_interpolate,
    _wrap_layernorms_dtype_preserving,
)


class DPTFixed(nn.Module):
    """Thin wrapper that patches the eager ``DPT`` for ONNX export.

    Applies the same surgical fixes as ``DualDPTFixed``:
      * Every ``nn.quantized.FloatFunctional`` -> plain addition.
      * ``custom_interpolate`` -> ``F.interpolate`` (no INT_MAX chunking).
      * LayerNorm wrapped to preserve fp16 dtype during tracing.

    Args:
        head: Eager ``DPT`` module (already loaded with weights).
        num_frames: Fixed sequence length ``S`` for this export.
        img_h: Input image height (must be divisible by 14).
        img_w: Input image width (must be divisible by 14).
    """

    def __init__(
        self,
        head: DPT,
        num_frames: int,
        img_h: int = IMG_H,
        img_w: int = IMG_W,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self._img_h = img_h
        self._img_w = img_w
        self.head = head

        _replace_float_functional(self.head)
        _wrap_layernorms_dtype_preserving(self.head)

        import types
        import depth_anything_3.model.dpt as _dpt_mod
        _dpt_mod.custom_interpolate = _safe_interpolate

        # Fix: patch _add_pos_embed to cast fp32 pos embeddings back to input dtype
        self.head._add_pos_embed = types.MethodType(_add_pos_embed_fixed, self.head)

    def forward(
        self,
        feat_0: torch.Tensor,
        feat_1: torch.Tensor,
        feat_2: torch.Tensor,
        feat_3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_0 .. feat_3: ``[B*S, N_patch, embed_dim]`` — 4 intermediate-layer
                features from the metric backbone (CLS already stripped).

        Returns:
            depth: ``[B*S, H, W]``
            sky:   ``[B*S, H, W]``
        """
        feats = [feat_0, feat_1, feat_2, feat_3]
        out = self.head._forward_impl(feats, self._img_h, self._img_w, patch_start_idx=0)

        depth = out[self.head.head_main]
        sky = out[self.head.sky_name]
        return depth, sky
