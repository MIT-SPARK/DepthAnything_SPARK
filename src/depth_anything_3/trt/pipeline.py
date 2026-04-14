"""
TensorRT inference pipeline for DA3-Giant: backbone → depth_head → floor_seg_head.

Provides a callable that takes normalized images and returns depth, camera poses,
and optional floor segmentation — matching the output format consumed by the SLAM
pipeline (solver.py:run_predictions_da).

Usage:
    from depth_anything_3.trt import DA3TensorRT

    pipeline = DA3TensorRT("checkpoints/da3_trt/engines_10.16.0.72_4/")
    result = pipeline(images)  # images: [B, S, 3, H, W] or [S, 3, H, W] in [0, 1]
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from depth_anything_3.trt.config import (
    BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from depth_anything_3.trt.engine import TRTEngine
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray


def _detect_num_frames(engine_dir: str) -> int | None:
    """Detect num_frames (S) from engine directory name like engines_10.16.0.72_4."""
    dirname = os.path.basename(engine_dir.rstrip("/"))
    parts = dirname.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        warnings.warn(f"Could not detect num_frames from engine dir name: {dirname}")
        return None


class DA3TensorRT:
    """Full DA3-Giant TensorRT pipeline — drop-in replacement for DA3 model.

    Chains TRT engines (backbone, depth_head, optionally floor_seg_head and
    metric branch) for fast inference. Post-processes ray/ray_conf →
    extrinsics/intrinsics in Python (SVD, not a bottleneck).

    Engine resolution is auto-detected from the backbone engine's input shape,
    so the pipeline works with any resolution the engines were exported at.

    All engine outputs are normalized to fp32 at a single point (the engine call
    itself via ``output_dtype``), eliminating scattered dtype casts in downstream
    math (SVD, alignment, quantile).

    Args:
        engine_dir: Directory containing the .engine files.
        device: CUDA device string.
    """

    def __init__(self, engine_dir: str, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.num_frames = _detect_num_frames(engine_dir)

        print(f"Loading DA3 TRT engines from {engine_dir} (S={self.num_frames})...")
        self.backbone = TRTEngine(os.path.join(engine_dir, "backbone.engine"))
        self.depth_head = TRTEngine(os.path.join(engine_dir, "depth_head.engine"))

        floor_seg_path = os.path.join(engine_dir, "floor_seg_head.engine")
        self.floor_seg_head = TRTEngine(floor_seg_path) if os.path.exists(floor_seg_path) else None

        # Metric branch engines (nested model support)
        metric_bb_path = os.path.join(engine_dir, "metric_backbone.engine")
        metric_hd_path = os.path.join(engine_dir, "metric_head.engine")
        has_metric = os.path.exists(metric_bb_path) and os.path.exists(metric_hd_path)
        self.metric_backbone = TRTEngine(metric_bb_path) if has_metric else None
        self.metric_head = TRTEngine(metric_hd_path) if has_metric else None
        self._nested = has_metric
        if has_metric:
            print("  Detected metric branch engines — nested mode enabled.")

        # Auto-detect engine resolution from backbone input shape [B, S, 3, H, W]
        bb_input_shape = self.backbone.input_shapes[self.backbone.input_names[0]]
        self._engine_h = bb_input_shape[3]
        self._engine_w = bb_input_shape[4]
        print(f"  Engine resolution: {self._engine_h}x{self._engine_w}")

        # Auto-detect I/O dtype from backbone engine (for input casting)
        self._io_dtype = self.backbone.input_dtypes[self.backbone.input_names[0]]

        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=self._io_dtype).view(1, 1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device, dtype=self._io_dtype).view(1, 1, 3, 1, 1)

        # Share a single CUDA stream across all engines
        self._stream = torch.cuda.Stream()
        engines = [self.backbone, self.depth_head]
        if self.floor_seg_head is not None:
            engines.append(self.floor_seg_head)
        if self.metric_backbone is not None:
            engines.append(self.metric_backbone)
        if self.metric_head is not None:
            engines.append(self.metric_head)
        for engine in engines:
            engine._stream = self._stream

        # Per-component timing accumulators
        self._detailed_timing = False
        self._timings = {
            "prepare": 0.0, "engines": 0.0, "backbone": 0.0,
            "depth_head": 0.0, "floor_seg": 0.0,
            "ransac": 0.0, "nested_alignment": 0.0,
            "resize": 0.0, "calls": 0,
        }

        print("All DA3 TRT engines loaded.")

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        compute_floor_seg: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run DA3 inference.

        Args:
            images: [B, S, 3, H, W] or [S, 3, H, W] in range [0, 1].
            compute_floor_seg: Run floor segmentation engine.

        Returns:
            Dict with depth, depth_conf, extrinsics, intrinsics, images,
            and optionally floor_logits. All tensors are fp32.
        """
        S = images.shape[-4] if images.dim() == 5 else images.shape[0]
        if self.num_frames is not None and S > self.num_frames:
            raise RuntimeError(
                f"DA3 TRT engines built for S={self.num_frames}, got S={S}. "
                "Build engines for the correct S."
            )
        return self._trt_forward(images, compute_floor_seg)

    def print_component_timing(self) -> None:
        """Print per-component timing breakdown."""
        t = self._timings
        n = t["calls"] or 1
        postprocess = t["ransac"] + t["nested_alignment"]
        print(f"  DA3 TRT component breakdown ({n} forward calls):")
        print(f"    prepare (pad+resize):  {t['prepare']:.2f}s ({t['prepare']/n*1000:.1f}ms/call)")
        print(f"    TRT engines:           {t['engines']:.2f}s ({t['engines']/n*1000:.1f}ms/call)")
        if self._detailed_timing:
            print(f"      backbone:            {t['backbone']:.2f}s ({t['backbone']/n*1000:.1f}ms/call)")
            print(f"      depth_head:          {t['depth_head']:.2f}s ({t['depth_head']/n*1000:.1f}ms/call)")
            print(f"      floor_seg:           {t['floor_seg']:.2f}s ({t['floor_seg']/n*1000:.1f}ms/call)")
        print(f"    postprocess:           {postprocess:.2f}s ({postprocess/n*1000:.1f}ms/call)")
        print(f"      ransac (ray->pose):  {t['ransac']:.2f}s ({t['ransac']/n*1000:.1f}ms/call)")
        print(f"      nested alignment:    {t['nested_alignment']:.2f}s ({t['nested_alignment']/n*1000:.1f}ms/call)")
        print(f"    resize outputs back:   {t['resize']:.2f}s ({t['resize']/n*1000:.1f}ms/call)")

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------

    def _prepare_images(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """Normalize batch dim, pad S if needed, resize to engine resolution."""
        if images.dim() == 4:
            images = images.unsqueeze(0)

        B, S, C, H, W = images.shape
        images = images.to(self.device, dtype=self._io_dtype).contiguous()
        orig_images = images
        actual_S = S

        # Pad to expected num_frames by repeating last frame
        expected_S = self.num_frames
        if expected_S is not None and S < expected_S:
            pad_count = expected_S - S
            padding = images[:, -1:].expand(B, pad_count, C, H, W).contiguous()
            images = torch.cat([images, padding], dim=1).contiguous()
            S = expected_S

        # Resize to engine resolution if needed
        eng_h, eng_w = self._engine_h, self._engine_w
        if H != eng_h or W != eng_w:
            images = F.interpolate(
                images.view(B * S, C, H, W), size=(eng_h, eng_w),
                mode="bilinear", align_corners=False,
            ).view(B, S, C, eng_h, eng_w).contiguous()

        return images, orig_images, B, S, actual_S

    # ------------------------------------------------------------------
    # Output resizing
    # ------------------------------------------------------------------

    def _resize_outputs(self, predictions: dict[str, torch.Tensor], orig_H: int, orig_W: int) -> None:
        """Resize spatial outputs from engine resolution back to original input resolution."""
        eng_h, eng_w = self._engine_h, self._engine_w
        if orig_H == eng_h and orig_W == eng_w:
            return
        size = (orig_H, orig_W)
        for key in ("depth", "depth_conf", "floor_logits"):
            if key not in predictions:
                continue
            t = predictions[key]
            B, S, tH, tW = t.shape
            t = t.reshape(B * S, 1, tH, tW)
            t = F.interpolate(t, size=size, mode="bilinear", align_corners=True)
            predictions[key] = t.reshape(B, S, orig_H, orig_W).contiguous()

    def _slice_to_actual_S(self, predictions: dict[str, torch.Tensor], actual_S: int, padded_S: int) -> None:
        """Slice predictions from padded S back to actual number of frames."""
        if actual_S == padded_S:
            return
        for key, t in predictions.items():
            if not isinstance(t, torch.Tensor) or t.dim() < 2:
                continue
            if t.shape[1] == padded_S:
                predictions[key] = t[:, :actual_S].contiguous()

    # ------------------------------------------------------------------
    # Post-processing: ray/ray_conf → extrinsics/intrinsics
    # ------------------------------------------------------------------

    # Maximum ray spatial dimension before RANSAC downsampling.
    # DualDPT aux output is 8x the patch grid (e.g. 168x296 for 294x518 input).
    # RANSAC only uses 8 points per hypothesis and samples 30% of total, so high
    # spatial resolution is wasted. 37 matches the patch-grid for 518px input;
    # this reduces N from ~50K to ~1K, cutting RANSAC intermediates by ~64x.
    _MAX_RAY_SIDE = 37

    def _ray_to_poses(
        self,
        ray: torch.Tensor,
        ray_conf: torch.Tensor,
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert ray/ray_conf tensors to extrinsics and intrinsics.

        Follows da3.py:_process_ray_pose_estimation.
        All inputs are already fp32 (normalized at engine call site).

        Args:
            ray:      [B, S, H_ray, W_ray, 6]
            ray_conf: [B, S, H_ray, W_ray]
            H, W: spatial dims of depth output (for intrinsic denormalization).

        Returns:
            extrinsics: [B, S, 3, 4] c2w
            intrinsics: [B, S, 3, 3]
        """
        B, S, H_ray, W_ray = ray.shape[:4]

        # Downsample ray to patch-grid resolution for faster RANSAC.
        # Ray directions and confidence are spatially smooth — bilinear is safe.
        if H_ray > self._MAX_RAY_SIDE or W_ray > self._MAX_RAY_SIDE:
            scale = self._MAX_RAY_SIDE / max(H_ray, W_ray)
            new_h = max(1, round(H_ray * scale))
            new_w = max(1, round(W_ray * scale))
            ray_flat = ray.reshape(B * S, H_ray, W_ray, 6).permute(0, 3, 1, 2)
            ray_flat = F.interpolate(ray_flat, size=(new_h, new_w), mode="bilinear", align_corners=True)
            ray = ray_flat.permute(0, 2, 3, 1).reshape(B, S, new_h, new_w, 6)

            conf_flat = ray_conf.reshape(B * S, 1, H_ray, W_ray)
            conf_flat = F.interpolate(conf_flat, size=(new_h, new_w), mode="bilinear", align_corners=True)
            ray_conf = conf_flat.reshape(B, S, new_h, new_w)

        # get_extrinsic_from_camray expects [B, S, H, W, 6] and [B, S, H, W]
        pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
            ray, ray_conf, ray.shape[-3], ray.shape[-2],
        )

        # w2c → c2w
        pred_extrinsic = affine_inverse(pred_extrinsic)[:, :, :3, :]  # [B, S, 3, 4]

        # Build intrinsic matrix
        pred_intrinsic = torch.eye(3, device=ray.device, dtype=ray.dtype).unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1).clone()
        pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * W
        pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * H
        pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * W * 0.5
        pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * H * 0.5

        return pred_extrinsic, pred_intrinsic

    # ------------------------------------------------------------------
    # Nested alignment (ports NestedDepthAnything3Net post-processing)
    # ------------------------------------------------------------------

    def _apply_nested_alignment(
        self,
        predictions: dict[str, torch.Tensor],
        metric_depth: torch.Tensor,
        sky: torch.Tensor,
    ) -> None:
        """Apply metric scaling, depth alignment, and sky handling.

        Mirrors ``NestedDepthAnything3Net.forward()`` post-processing steps:
          1. Scale metric depth by Giant-predicted intrinsics
          2. Least-squares align Giant depth to metric depth
          3. Set sky regions to 99th-percentile non-sky depth (capped at 200m)

        All inputs are already fp32 (normalized at engine call site).
        """
        # 1. Scale metric depth by Giant's predicted intrinsics
        metric_depth = apply_metric_scaling(metric_depth, predictions["intrinsics"])

        # 2. Depth alignment via least squares
        non_sky_mask = compute_sky_mask(sky, threshold=0.3)
        depth_conf_ns = predictions["depth_conf"][non_sky_mask]
        depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
        median_conf = torch.quantile(depth_conf_sampled, 0.5)

        align_mask = compute_alignment_mask(
            predictions["depth_conf"], non_sky_mask,
            predictions["depth"], metric_depth, median_conf,
        )
        scale_factor = least_squares_scale_scalar(
            metric_depth[align_mask], predictions["depth"][align_mask],
        )

        predictions["depth"] = predictions["depth"] * scale_factor
        predictions["extrinsics"][:, :, :3, 3] = (
            predictions["extrinsics"][:, :, :3, 3] * scale_factor
        )
        predictions["is_metric"] = True

        # 3. Sky regions → 99th-percentile non-sky depth (capped at 200m)
        non_sky_mask = compute_sky_mask(sky, threshold=0.3)
        non_sky_depth = predictions["depth"][non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = min(torch.quantile(sampled_depth, 0.99).item(), 200.0)

        predictions["depth"], predictions["depth_conf"] = set_sky_regions_to_max_depth(
            predictions["depth"], predictions["depth_conf"],
            non_sky_mask, max_depth=non_sky_max,
        )

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def _trt_forward(self, images: torch.Tensor, compute_floor_seg: bool) -> dict[str, torch.Tensor]:
        """Run through backbone → depth_head → post-process → optional floor_seg → optional metric.

        All engine outputs are cast to fp32 via ``output_dtype=torch.float32``
        at the call site, so downstream code never encounters mixed bf16/fp16/fp32.
        """
        t0 = time.perf_counter()
        images, orig_images, B, S, actual_S = self._prepare_images(images)
        orig_H, orig_W = orig_images.shape[-2], orig_images.shape[-1]
        torch.cuda.synchronize()
        t_prepare = time.perf_counter()

        predictions: dict[str, torch.Tensor] = {}
        predictions["images"] = orig_images
        detailed = self._detailed_timing
        fp32 = torch.float32

        with torch.cuda.stream(self._stream):
            # -- 1. Normalize images --
            images_norm = (images - self._mean) / self._std

            # -- 2. Giant backbone --
            backbone_out = self.backbone(images_norm, sync=False, output_dtype=fp32)
            features = backbone_out["features"]          # [4, B, S, N_patch, 3072]
            aux_tokens = backbone_out["aux_tokens"]      # [B, S, N_patch, 3072]
            camera_tokens = backbone_out["camera_tokens"]  # [B, S, 3072]
            if detailed:
                self._stream.synchronize()
                t_backbone = time.perf_counter()

            # -- 3. Giant depth head --
            feat_layers = []
            for i in range(4):
                feat_i = features[i]  # [B, S, N_patch, C]
                feat_i = rearrange(feat_i, "b s n c -> (b s) n c").contiguous()
                feat_layers.append(feat_i)

            depth_out = self.depth_head(*feat_layers, sync=False, output_dtype=fp32)
            depth = depth_out["depth"]          # [B*S, H, W]
            depth_conf = depth_out["depth_conf"]  # [B*S, H, W]
            ray = depth_out["ray"]              # [B*S, H_ray, W_ray, 6]
            ray_conf = depth_out["ray_conf"]    # [B*S, H_ray, W_ray]
            if detailed:
                self._stream.synchronize()
                t_depth = time.perf_counter()

            # -- 4. Floor segmentation (optional) --
            if compute_floor_seg and self.floor_seg_head is not None:
                floor_out = self.floor_seg_head(aux_tokens, images_norm, sync=False, output_dtype=fp32)
                floor_logits = floor_out["floor_logits"]  # [B, S, H, W]
                predictions["floor_logits"] = floor_logits

            # -- 5. Metric branch (nested model) --
            metric_depth_bs = None
            sky_bs = None
            if self._nested:
                metric_bb_out = self.metric_backbone(images_norm, sync=False, output_dtype=fp32)
                metric_features = metric_bb_out["features"]  # [4, B, S, N_patch, 1024]

                metric_feat_layers = []
                for i in range(4):
                    mf = metric_features[i]
                    mf = rearrange(mf, "b s n c -> (b s) n c").contiguous()
                    metric_feat_layers.append(mf)

                metric_out = self.metric_head(*metric_feat_layers, sync=False, output_dtype=fp32)
                metric_depth_bs = metric_out["depth"]  # [B*S, H, W]
                sky_bs = metric_out["sky"]              # [B*S, H, W]

            if detailed:
                self._stream.synchronize()
                t_floor = time.perf_counter()

        # Sync all engine work
        self._stream.synchronize()
        t_engines_done = time.perf_counter()

        # -- 6. Reshape depth outputs: [B*S, ...] → [B, S, ...] --
        H_out, W_out = depth.shape[-2], depth.shape[-1]
        predictions["depth"] = depth.reshape(B, S, H_out, W_out)
        predictions["depth_conf"] = depth_conf.reshape(B, S, H_out, W_out)

        # -- 7. Post-process ray → extrinsics/intrinsics (Python, uses SVD) --
        H_ray, W_ray = ray.shape[-3], ray.shape[-2]
        ray = ray.reshape(B, S, H_ray, W_ray, 6)
        ray_conf = ray_conf.reshape(B, S, H_ray, W_ray)
        predictions["extrinsics"], predictions["intrinsics"] = self._ray_to_poses(
            ray, ray_conf, H_out, W_out,
        )
        torch.cuda.synchronize()
        t_ransac = time.perf_counter()

        # -- 8. Nested alignment (metric scaling + depth alignment + sky) --
        if self._nested and metric_depth_bs is not None:
            metric_depth = metric_depth_bs.reshape(B, S, H_out, W_out)
            sky = sky_bs.reshape(B, S, H_out, W_out)
            self._apply_nested_alignment(predictions, metric_depth, sky)

        predictions["camera_tokens"] = camera_tokens
        torch.cuda.synchronize()
        t_alignment = time.perf_counter()

        # -- 9. Slice padded frames, resize to original resolution --
        self._slice_to_actual_S(predictions, actual_S, S)
        self._resize_outputs(predictions, orig_H, orig_W)
        torch.cuda.synchronize()
        t_resize = time.perf_counter()

        # Accumulate timing
        self._timings["prepare"] += t_prepare - t0
        self._timings["engines"] += t_engines_done - t_prepare
        if detailed:
            self._timings["backbone"] += t_backbone - t_prepare
            self._timings["depth_head"] += t_depth - t_backbone
            self._timings["floor_seg"] += t_floor - t_depth
        self._timings["ransac"] += t_ransac - t_engines_done
        self._timings["nested_alignment"] += t_alignment - t_ransac
        self._timings["resize"] += t_resize - t_alignment
        self._timings["calls"] += 1

        return predictions

    # ------------------------------------------------------------------
    # High-level inference (drop-in for DepthAnything3.inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference(
        self,
        image: list[str],
        export_feat_layers: list[int] | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        **kwargs,
    ) -> Prediction:
        """Drop-in replacement for ``DepthAnything3.inference()``.

        Loads images from file paths, runs TRT inference, and returns a
        ``Prediction`` dataclass with numpy arrays — same interface that
        ``solver.run_predictions_da()`` expects.

        Args:
            image: List of image file paths.
            export_feat_layers: If non-empty and floor_seg engine is loaded,
                run floor segmentation and return logits in ``prediction.aux``.
            process_res: Processing resolution passed to InputProcessor.
            process_res_method: Resize method (``"upper_bound_resize"``).

        Returns:
            Prediction with depth, conf, extrinsics, intrinsics, processed_images,
            and optionally floor_logits in aux.
        """
        # 1. Load and preprocess images via DA3's InputProcessor
        processor = InputProcessor()
        imgs_cpu, _, _ = processor(image, None, None, process_res, process_res_method)
        # imgs_cpu: [N, 3, H, W] ImageNet-normalized

        # 2. Denormalize to [0, 1] — the TRT pipeline applies its own normalization
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        images_01 = (imgs_cpu * std + mean).unsqueeze(0)  # [1, S, 3, H, W]

        # 3. Run TRT forward
        compute_floor = bool(
            export_feat_layers and self.floor_seg_head is not None
        )
        trt_out = self(images_01, compute_floor_seg=compute_floor)

        # 4. Build Prediction from TRT outputs
        depth = trt_out["depth"].squeeze(0).cpu().numpy()  # [S, H, W]
        depth_conf = trt_out.get("depth_conf", None)
        if depth_conf is not None:
            depth_conf = depth_conf.squeeze(0).cpu().numpy()  # [S, H, W]

        # Extrinsics: [B, S, 3, 4] c2w → pad to [S, 4, 4]
        ext_34 = trt_out["extrinsics"].squeeze(0).cpu()  # [S, 3, 4]
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(ext_34.shape[0], 1, 4)
        extrinsics = torch.cat([ext_34, bottom], dim=1).numpy()  # [S, 4, 4]

        intrinsics = trt_out["intrinsics"].squeeze(0).cpu().numpy()  # [S, 3, 3]

        # 5. Processed images: denormalize imgs_cpu to uint8 [N, H, W, 3]
        processed = imgs_cpu.permute(0, 2, 3, 1).cpu().numpy()  # [N, H, W, 3]
        processed = processed * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        processed = np.clip(processed, 0, 1)
        processed_images = (processed * 255).astype(np.uint8)

        # 6. Build aux dict
        aux = {}
        if compute_floor and "floor_logits" in trt_out:
            aux["floor_logits"] = trt_out["floor_logits"].cpu()

        is_metric = 1 if trt_out.get("is_metric", False) else 0

        return Prediction(
            depth=depth,
            is_metric=is_metric,
            conf=depth_conf,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            processed_images=processed_images,
            aux=aux if aux else None,
        )
