"""
Fixed RoPE2D with precomputed cos/sin tables for TensorRT export.

The original RotaryPositionEmbedding2D does ``int(positions.max()) + 1`` which
extracts a Python int from a tensor at trace time, blocking torch.export / ONNX.

Fix: precompute all cos/sin tables for all positions up to *max_positions* and
store them as registered buffers.  At runtime only ``F.embedding`` (a pure
index lookup) is needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D


class FixedRoPE2D(nn.Module):
    """Drop-in replacement for ``RotaryPositionEmbedding2D`` with static tables.

    Args:
        original_rope: Eager module whose ``base_frequency`` is extracted.
        feature_dim:   Per-spatial-direction feature dim (embed_dim // num_heads // 2).
        max_positions: Largest position index to precompute for.
        dtype:         Storage dtype for the cos / sin buffers.
    """

    def __init__(
        self,
        original_rope: RotaryPositionEmbedding2D,
        feature_dim: int = 32,
        max_positions: int = 64,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        base_frequency = original_rope.base_frequency

        exponents = torch.arange(0, feature_dim, 2, dtype=torch.float32) / feature_dim
        inv_freq = 1.0 / (base_frequency ** exponents)

        positions = torch.arange(max_positions, dtype=torch.float32)
        angles = torch.einsum("i,j->ij", positions, inv_freq)  # [max_pos, feat/2]
        angles = torch.cat((angles, angles), dim=-1)            # [max_pos, feat]

        self.register_buffer("cos_cached", angles.cos().to(dtype))
        self.register_buffer("sin_cached", angles.sin().to(dtype))

    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor,
    ) -> torch.Tensor:
        cos = F.embedding(positions, self.cos_cached)[:, None, :, :]
        sin = F.embedding(positions, self.sin_cached)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = self._apply_1d_rope(vertical, positions[..., 0])
        horizontal = self._apply_1d_rope(horizontal, positions[..., 1])
        return torch.cat((vertical, horizontal), dim=-1)


def replace_rope_in_blocks(blocks: nn.ModuleList, fixed_rope: FixedRoPE2D) -> None:
    """Walk *blocks* and replace every ``RotaryPositionEmbedding2D`` with *fixed_rope*."""
    for blk in blocks:
        for name, child in blk.named_modules():
            if hasattr(child, "rope") and isinstance(child.rope, RotaryPositionEmbedding2D):
                child.rope = fixed_rope
