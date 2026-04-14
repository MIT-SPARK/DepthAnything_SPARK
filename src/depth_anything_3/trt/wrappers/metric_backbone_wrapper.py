"""
ONNX-exportable wrapper for the DA3-Metric ViT-Large backbone.

Much simpler than the Giant wrapper — ViT-Large has:
  - alt_start=-1: all blocks use local attention (no global cross-view attention)
  - rope_start=-1: no RoPE
  - cat_token=False: output is just x, not cat(local_x, x)
  - No camera tokens

Only needs:
  1. Precomputed positional encoding (no bicubic interpolation at runtime).
  2. Explicit block loop (static ONNX graph).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from depth_anything_3.model.dinov2.dinov2 import DinoV2
from depth_anything_3.trt.config import (
    BATCH_SIZE,
    IMG_H,
    IMG_W,
    METRIC_DEPTH,
    METRIC_EMBED_DIM,
    METRIC_OUT_LAYERS,
    METRIC_PATCH_START_IDX,
    PATCH_DIM_H,
    PATCH_DIM_W,
    PATCH_SIZE,
    patch_dims,
)


class DA3MetricBackboneFixed(nn.Module):
    """DINOv2 ViT-Large backbone with all dynamic ops replaced by static buffers.

    The metric backbone is vanilla DINOv2 — no RoPE, no alternating local/global
    attention, no camera tokens, no cat_token. Only the positional encoding needs
    to be precomputed to avoid bicubic interpolation at export time.

    Args:
        backbone: Eager ``DinoV2`` module for the metric branch (already loaded with weights).
        num_frames: Fixed sequence length ``S`` for this export.
        img_h: Input image height (must be divisible by 14).
        img_w: Input image width (must be divisible by 14).
    """

    def __init__(
        self,
        backbone: DinoV2,
        num_frames: int,
        img_h: int = IMG_H,
        img_w: int = IMG_W,
    ) -> None:
        super().__init__()
        vit = backbone.pretrained
        self.num_frames = num_frames
        ph, pw, num_patches = patch_dims(img_h, img_w)

        # Core modules (shared weights, no copy)
        self.patch_embed = vit.patch_embed
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.cls_token = vit.cls_token  # [1, 1, embed_dim]

        # Precompute positional encoding for fixed patch grid
        pos_embed = self._precompute_pos_embed(vit, ph, pw)
        self.register_buffer("fixed_pos_embed", pos_embed)

    @staticmethod
    def _precompute_pos_embed(vit: nn.Module, target_h: int, target_w: int) -> torch.Tensor:
        """Compute positional encoding for the fixed patch grid."""
        pos_embed = vit.pos_embed.float()  # [1, N+1, embed_dim]
        N = pos_embed.shape[1] - 1
        M = int(math.sqrt(N))
        assert M * M == N

        class_pos = pos_embed[:, 0]   # [1, embed_dim]
        patch_pos = pos_embed[:, 1:]  # [1, N, embed_dim]
        dim = patch_pos.shape[-1]

        if vit.interpolate_offset:
            sx = float(target_w + vit.interpolate_offset) / M
            sy = float(target_h + vit.interpolate_offset) / M
            patch_pos = F.interpolate(
                patch_pos.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                scale_factor=(sy, sx),
                mode="bicubic",
                antialias=vit.interpolate_antialias,
            )
        else:
            patch_pos = F.interpolate(
                patch_pos.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                size=(target_h, target_w),
                mode="bicubic",
                antialias=vit.interpolate_antialias,
            )

        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([class_pos.unsqueeze(0), patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images ``[B, S, 3, H, W]`` normalised to ImageNet stats.

        Returns:
            features: ``[4, B, S, N_patch, embed_dim]`` — 4 intermediate-layer outputs
                      (normalised, CLS stripped).
        """
        B, S = x.shape[:2]

        # Patch embedding + fixed positional encoding
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)                          # [B*S, N_patch, E]
        cls = self.cls_token.expand(B * S, -1, -1)       # [B*S, 1, E]
        x = torch.cat([cls, x], dim=1)                   # [B*S, N_patch+1, E]
        x = x + self.fixed_pos_embed.to(x.dtype)

        # All blocks use local attention (no rearrange needed)
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in METRIC_OUT_LAYERS:
                features.append(x)

        # Normalise and strip CLS
        strip_idx = 1  # CLS only, no register tokens
        normed_features = [self.norm(feat)[:, strip_idx:, :] for feat in features]

        # Stack: [4, B*S, N_patch, E] → [4, B, S, N_patch, E]
        stacked = torch.stack(normed_features, dim=0)
        stacked = rearrange(stacked, "L (b s) n c -> L b s n c", b=B, s=S)
        return stacked
