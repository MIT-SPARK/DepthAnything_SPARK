"""
Thin re-export of the model-agnostic TRTEngine from the VGGT package.

If VGGT is not installed, a minimal fallback is provided.
"""
from __future__ import annotations

from vggt.trt.engine import TRTEngine

__all__ = ["TRTEngine"]
