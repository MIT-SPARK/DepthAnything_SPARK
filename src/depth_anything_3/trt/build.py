"""
Convert ONNX models to TensorRT .engine files.

Usage as library:
    from depth_anything_3.trt.build import build_all
    build_all(onnx_dir, engine_dir)
"""
from __future__ import annotations

import os
import subprocess
import time

from depth_anything_3.trt.config import COMPONENTS, TRT_FP16, TRT_WORKSPACE_GB

# Components that need bf16 instead of fp16 due to activation outliers
# (ViT-Giant 1536-dim residual stream exceeds fp16 max in later blocks).
_BF16_COMPONENTS = {"backbone"}

# Components that must stay fp32 — aux head Conv2d(128→32) overflows fp16
# and TRT's mixed-precision constraints can't isolate it without breaking
# graph fusion.  The depth head is small (~92 MB), so fp32 is acceptable.
_FP32_COMPONENTS = {"depth_head"}


def _force_fp32_layers(network: "trt.INetworkDefinition", config: "trt.IBuilderConfig") -> None:
    """Force all normalization layers to fp32 to prevent fp16 variance overflow."""
    import tensorrt as trt

    norm_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type == trt.LayerType.NORMALIZATION:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            norm_count += 1
    if norm_count > 0:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        print(f"  Forced {norm_count} normalization layers to fp32")


def build_engine_trt_api(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    bf16: bool = False,
    workspace_gb: int = 8,
) -> None:
    """Build a TensorRT engine from an ONNX model using the Python API."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"  Parsing ONNX: {onnx_path}")
    if hasattr(parser, "parse_from_file"):
        ok = parser.parse_from_file(onnx_path)
    else:
        with open(onnx_path, "rb") as f:
            ok = parser.parse(f.read())

    if not ok:
        for i in range(parser.num_errors):
            print(f"    Parse error: {parser.get_error(i)}")
        raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    print(f"  Network: {network.num_inputs} inputs, {network.num_outputs} outputs")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    Input  {i}: {inp.name} shape={inp.shape} dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    Output {i}: {out.name} shape={out.shape} dtype={out.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if bf16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.BF16)
        _force_fp32_layers(network, config)
        print("  BF16 enabled (fp32 range, avoids outlier overflow)")
    elif fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        _force_fp32_layers(network, config)
        print("  FP16 enabled")
    elif fp16 or bf16:
        print("  WARNING: reduced precision not supported on this platform, using FP32")

    print("  Building engine (this may take several minutes)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    assert serialized is not None, f"Engine build failed for {onnx_path}"

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"  Saved: {engine_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")


def build_engine_trtexec(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    bf16: bool = False,
    workspace_gb: int = 8,
) -> None:
    """Build a TensorRT engine using the trtexec CLI (fallback)."""
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_gb * 1024}",
    ]
    if bf16:
        cmd.append("--bf16")
        cmd.append("--layerNormPrecision=fp32")
    elif fp16:
        cmd.append("--fp16")
        cmd.append("--layerNormPrecision=fp32")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"trtexec failed for {onnx_path}")

    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"  Saved: {engine_path} ({size_mb:.1f} MB)")


def build_all(
    onnx_dir: str,
    engine_dir: str,
    fp16: bool = TRT_FP16,
    workspace_gb: int = TRT_WORKSPACE_GB,
    components: list[str] | None = None,
    use_trtexec: bool = False,
) -> None:
    """Build TRT engines for all (or selected) DA3 components.

    Args:
        onnx_dir: Directory containing ONNX models.
        engine_dir: Output directory for ``.engine`` files.
        fp16: Enable FP16.
        workspace_gb: TRT workspace size in GB.
        components: Subset of COMPONENTS. None = all.
        use_trtexec: Use trtexec CLI instead of Python API.
    """
    if components is None:
        components = list(COMPONENTS)
    os.makedirs(engine_dir, exist_ok=True)

    build_fn = build_engine_trtexec if use_trtexec else build_engine_trt_api

    if not use_trtexec:
        try:
            import tensorrt
            print(f"TensorRT version: {tensorrt.__version__}")
        except ImportError:
            print("TensorRT Python package not found. Falling back to trtexec CLI.")
            build_fn = build_engine_trtexec

    for name in components:
        # Backbone may use external data subdirectory (>2 GB)
        subdir_path = os.path.join(onnx_dir, f"{name}_data", f"{name}.onnx")
        flat_path = os.path.join(onnx_dir, f"{name}.onnx")
        onnx_path = subdir_path if os.path.exists(subdir_path) else flat_path

        if not os.path.exists(onnx_path):
            print(f"\nSkipping {name}: {onnx_path} not found")
            continue

        engine_path = os.path.join(engine_dir, f"{name}.engine")
        # Per-component precision:
        # - backbone: bf16 (outlier activations overflow fp16 in later blocks)
        # - depth_head: fp32 (aux Conv2d(128→32) overflows fp16)
        # - others: fp16
        force_fp32 = name in _FP32_COMPONENTS
        use_bf16 = fp16 and name in _BF16_COMPONENTS
        use_fp16 = fp16 and not use_bf16 and not force_fp32
        # fp32 components need more workspace (128→32 conv at BS=32 needs >8 GB)
        ws = max(workspace_gb, 16) if force_fp32 else workspace_gb
        print(f"\n=== Building engine: {name} ===")
        build_fn(onnx_path, engine_path, fp16=use_fp16, bf16=use_bf16, workspace_gb=ws)

    print(f"\n=== Engine build complete → {engine_dir} ===")
