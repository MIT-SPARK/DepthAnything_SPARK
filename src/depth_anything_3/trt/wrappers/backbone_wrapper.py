"""
ONNX-exportable wrapper for the DA3-Giant DINOv2 backbone.

Fixes for TRT/ONNX compatibility:
  1. Precomputed positional encoding (no bicubic interpolation at runtime).
  2. Precomputed RoPE position grids as registered buffers.
  3. Reference view selection forced to "first" (no data-dependent control flow).
  4. Camera tokens always injected from the learnable parameter (no conditional).
  5. Block loop is explicit Python; each block's attention type (local / global)
     is determined at wrapper-construction time, so the ONNX graph is static.
  6. ``xformers.ops.SwiGLU`` (fused kernel) replaced with pure-PyTorch ``SwiGLUFFN``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from depth_anything_3.model.dinov2.dinov2 import DinoV2
from depth_anything_3.model.dinov2.layers.swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from depth_anything_3.trt.config import (
    ALT_START,
    BATCH_SIZE,
    EMBED_DIM,
    IMG_H,
    IMG_W,
    NUM_REGISTER_TOKENS,
    OUT_LAYERS,
    PATCH_DIM_H,
    PATCH_DIM_W,
    PATCH_SIZE,
    PATCH_START_IDX,
    ROPE_MAX_POS,
    ROPE_START,
    SEG_LAYER_IDX,
    patch_dims,
)
from depth_anything_3.trt.wrappers.rope_fixed import FixedRoPE2D, replace_rope_in_blocks


def _replace_swiglu_fused(module: nn.Module) -> None:
    """Replace ``xformers.ops.SwiGLU`` (fused kernel) with pure-PyTorch ``SwiGLUFFN``.

    Both classes use ``w12`` and ``w3`` linear layers so weights transfer directly.
    """
    for name, child in module.named_children():
        if isinstance(child, SwiGLUFFNFused) and type(child) is not SwiGLUFFN:
            pure = SwiGLUFFN(
                in_features=child.w12.in_features,
                hidden_features=child.w12.out_features // 2,
                out_features=child.w3.out_features,
                bias=child.w12.bias is not None,
            )
            pure.w12 = child.w12
            pure.w3 = child.w3
            setattr(module, name, pure)
        else:
            _replace_swiglu_fused(child)


class FP32LayerNorm(nn.Module):
    """LayerNorm that always computes in fp32 regardless of input dtype.

    Prevents fp16 overflow in variance computation for large embedding dims
    (ViT-Giant 1536-dim: sum of 1536 squared values exceeds fp16 max 65504).
    Emits explicit Cast(fp32)->LayerNorm->Cast(orig) nodes in the ONNX graph
    so TRT preserves fp32 precision for these ops even with --fp16.
    """

    def __init__(self, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.weight = norm.weight
        self.bias = norm.bias
        self.normalized_shape = norm.normalized_shape
        self.eps = norm.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(orig_dtype)


def _wrap_layernorms_fp32(module: nn.Module) -> None:
    """Recursively replace all nn.LayerNorm instances with FP32LayerNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, FP32LayerNorm(child))
        else:
            _wrap_layernorms_fp32(child)


class DA3BackboneFixed(nn.Module):
    """DINOv2 ViT-Giant backbone with all dynamic ops replaced by static buffers.

    Args:
        backbone: Eager ``DinoV2`` module (already loaded with weights).
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
        B = BATCH_SIZE
        ph, pw, num_patches = patch_dims(img_h, img_w)
        self._ph = ph
        self._pw = pw
        self._num_patches = num_patches

        # ---- core modules (shared weights, no copy) ----
        self.patch_embed = vit.patch_embed
        self.blocks = vit.blocks
        self.norm = vit.norm

        # ---- replace xformers SwiGLU with pure-PyTorch SwiGLUFFN ----
        _replace_swiglu_fused(self.blocks)

        # ---- cls token ----
        self.cls_token = vit.cls_token  # [1, 1, embed_dim]

        # ---- camera tokens (learnable ref / src) ----
        # Always inject default camera tokens at block ALT_START.
        # camera_token: [1, 2, embed_dim] — [ref, src]
        cam_tok = vit.camera_token  # nn.Parameter [1, 2, embed_dim]
        ref_token = cam_tok[:, :1].expand(B, -1, -1)               # [1, 1, E]
        src_token = cam_tok[:, 1:].expand(B, num_frames - 1, -1)   # [1, S-1, E]
        cam = torch.cat([ref_token, src_token], dim=1)              # [1, S, E]
        self.register_buffer("cam_token_expanded", cam.detach().clone())

        # ---- fixed positional encoding (replaces bicubic interpolation) ----
        pos_embed = self._precompute_pos_embed(vit, ph, pw)
        self.register_buffer("fixed_pos_embed", pos_embed)  # [1, num_patches+1, embed_dim]

        # ---- fixed RoPE ----
        feature_dim = EMBED_DIM // vit.num_heads // 2  # 32
        fixed_rope = FixedRoPE2D(
            vit.rope, feature_dim=feature_dim, max_positions=ROPE_MAX_POS,
        )
        replace_rope_in_blocks(self.blocks, fixed_rope)

        # ---- precomputed RoPE position grids ----
        local_pos, global_pos, local_pos_nodiff, global_pos_nodiff = (
            self._precompute_positions(B, num_frames, ph, pw)
        )
        self.register_buffer("local_pos", local_pos)              # [B*S, T, 2]
        self.register_buffer("global_pos", global_pos)            # [B, S*T, 2]
        self.register_buffer("local_pos_nodiff", local_pos_nodiff)
        self.register_buffer("global_pos_nodiff", global_pos_nodiff)

        # ---- wrap LayerNorms for fp32 computation ----
        # ViT-Giant 1536-dim variance computation overflows fp16.
        # Bakes explicit Cast(fp32) nodes into the ONNX graph so TRT
        # keeps LayerNorm in fp32 while using fp16 for matmuls/SDPA.
        _wrap_layernorms_fp32(self)

    # ------------------------------------------------------------------
    # Precomputation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _precompute_pos_embed(vit: nn.Module, target_h: int, target_w: int) -> torch.Tensor:
        """Compute positional encoding for the fixed patch grid."""
        pos_embed = vit.pos_embed.float()  # [1, N+1, embed_dim]
        N = pos_embed.shape[1] - 1  # number of patch positions in the stored embed
        M = int(math.sqrt(N))
        assert M * M == N

        class_pos = pos_embed[:, 0]        # [1, embed_dim]
        patch_pos = pos_embed[:, 1:]       # [1, N, embed_dim]
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

    @staticmethod
    def _precompute_positions(
        B: int, S: int, ph: int, pw: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build RoPE position grids for local and global attention."""
        num_patches = ph * pw
        # Patch positions: y,x coordinates in the patch grid
        y = torch.arange(ph)
        x = torch.arange(pw)
        patch_pos = torch.cartesian_prod(y, x)  # [num_patches, 2]

        # Expand for batch*seq
        pos = patch_pos.unsqueeze(0).expand(B * S, -1, -1).clone()  # [B*S, num_patches, 2]
        pos = pos + 1  # shift so CLS can be 0

        # CLS position = 0
        cls_pos = torch.zeros(B * S, PATCH_START_IDX, 2)  # [B*S, 1, 2]
        pos = torch.cat([cls_pos, pos], dim=1)  # [B*S, num_patches+1, 2]

        pos_nodiff = torch.ones_like(pos)
        pos_nodiff[:, :PATCH_START_IDX] = 0  # CLS stays at 0

        # Local: [B*S, T, 2]
        local_pos = pos.long()
        local_pos_nodiff = pos_nodiff.long()

        # Global: rearrange (B S) T 2 → B (S*T) 2
        global_pos = rearrange(pos, "(b s) t c -> b (s t) c", b=B, s=S).long()
        global_pos_nodiff = rearrange(
            pos_nodiff, "(b s) t c -> b (s t) c", b=B, s=S,
        ).long()

        return local_pos, global_pos, local_pos_nodiff, global_pos_nodiff

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Images ``[B, S, 3, H, W]`` normalised to ImageNet stats.

        Returns:
            features:      ``[4, B, S, N_patch, 3072]`` — 4 intermediate-layer
                           outputs (local||global concat, global-half normalised,
                           CLS stripped).
            aux_tokens:    ``[B, S, N_patch, 3072]`` — layer-19 tokens for floor seg.
            camera_tokens: ``[B, S, 3072]`` — CLS token from last output layer
                           (for camera pose estimation).
        """
        B, S = x.shape[:2]

        # -- patch embedding + fixed positional encoding --
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)                          # [B*S, N_patch, E]
        cls = self.cls_token.expand(B * S, -1, -1)       # [B*S, 1, E]
        x = torch.cat([cls, x], dim=1)                   # [B*S, N_patch+1, E]
        x = x + self.fixed_pos_embed.to(x.dtype)         # add precomputed pos embed
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)  # [B, S, N_patch+1, E]

        # -- run transformer blocks --
        features = []      # will hold 4 (cls_token, full_features) tuples
        aux_features = []  # layer-19 tokens
        local_x = x        # tracks output of last local-attention block

        for i, blk in enumerate(self.blocks):
            # Determine RoPE positions for this block
            use_rope = i >= ROPE_START

            # At alt_start: inject camera tokens
            if i == ALT_START:
                x = x.clone()
                x[:, :, 0] = self.cam_token_expanded.to(x.dtype)

            # Determine attention type
            is_global = (i >= ALT_START) and (i % 2 == 1)

            if is_global:
                # Global cross-view attention
                pos = self.global_pos_nodiff if use_rope else None
                x = rearrange(x, "b s n c -> b (s n) c")
                x = blk(x, pos=pos)
                x = rearrange(x, "b (s n) c -> b s n c", s=S)
            else:
                # Local per-view attention
                pos = self.local_pos if use_rope else None
                x = rearrange(x, "b s n c -> (b s) n c")
                x = blk(x, pos=pos)
                x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
                local_x = x

            # Collect outputs at designated layers
            if i in OUT_LAYERS:
                cat_feat = torch.cat([local_x, x], dim=-1)  # [B, S, 1370, 2E]
                features.append(cat_feat)

            if i == SEG_LAYER_IDX:
                cat_aux = torch.cat([local_x, x], dim=-1)
                aux_features.append(cat_aux)

        # -- normalise global stream (second half) and strip CLS --
        strip_idx = 1 + NUM_REGISTER_TOKENS  # 1
        normed_features = []
        for feat in features:
            local_half = feat[..., :EMBED_DIM]
            global_half = self.norm(feat[..., EMBED_DIM:])
            normed = torch.cat([local_half, global_half], dim=-1)
            normed_features.append(normed[..., strip_idx:, :])  # strip CLS → [B,S,1369,3072]

        # Camera tokens: CLS from last output layer (before stripping)
        last_feat = features[-1]
        local_cls = last_feat[..., :EMBED_DIM]
        global_cls = self.norm(last_feat[..., EMBED_DIM:])
        camera_tokens = torch.cat([local_cls, global_cls], dim=-1)[:, :, 0]  # [B, S, 3072]

        # Aux tokens: normalise and strip CLS
        normed_aux = []
        for aux in aux_features:
            local_half = aux[..., :EMBED_DIM]
            global_half = self.norm(aux[..., EMBED_DIM:])
            normed_aux.append(
                torch.cat([local_half, global_half], dim=-1)[..., strip_idx:, :]
            )

        # Stack features: [4, B, S, 1369, 3072]
        features_stacked = torch.stack(normed_features, dim=0)
        aux_out = normed_aux[0] if normed_aux else normed_features[0]

        return features_stacked, aux_out, camera_tokens
