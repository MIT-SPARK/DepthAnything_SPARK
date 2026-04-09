"""
Ablation study: train one FloorSegHead per ViT layer of the DA3 backbone and
report/plot validation IoU vs layer index to identify the best layer.

All layer tokens are extracted in a SINGLE backbone forward pass per batch via
get_intermediate_layers(n=all_layers).  Each head is then updated sequentially
and the token tensor for that layer is freed immediately afterwards, keeping
peak GPU memory bounded.

Automatically uses all available GPUs via DDP (torchrun re-launch).

Usage:
    python3 train_floor_seg_ablation.py \\
        --dataset_root /home/dominic/Documents/floor_label/floor_dataset \\
        --da_model depth-anything/DA3-GIANT \\
        --epochs 15 --layer_stride 2

    # Only test a specific set of layers:
    python3 train_floor_seg_ablation.py --layers 10 15 20 25 30 35
"""

import argparse
import datetime
import gc
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, "/home/dominic/vggt")  # for training.data.datasets.floor_dataset

import torch
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
    print("TensorBoard is available. Training logs and figures will be saved to the specified --tb_dir.")
except ImportError:
    SummaryWriter = None
    _TB_AVAILABLE = False

import io

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.floor_seg_head import FloorSegHead
from training.data.datasets.floor_dataset import YOLOFloorDataset

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    std  = _IMAGENET_STD.to(x.device,  dtype=x.dtype).view(1, 1, 3, 1, 1)
    return (x - mean) / std


def get_backbone(model: DepthAnything3):
    inner = model.model
    if hasattr(inner, "da3"):
        return inner.da3.backbone
    return inner.backbone


def bce_dice_loss(logits, targets, bce_weight=0.5, pos_weight=3.0):
    pw = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    probs = torch.sigmoid(logits)
    smooth = 1e-6
    intersection = (probs * targets).sum()
    dice = 1 - (2 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return bce_weight * bce + (1 - bce_weight) * dice


def compute_iou(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def is_main(rank): return rank == 0

def maybe_print(rank, *args, **kwargs):
    if is_main(rank): print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Single backbone pass → all layer tokens
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_layer_tokens(backbone, images_norm: torch.Tensor, layer_indices: list):
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs, _ = backbone.pretrained.get_intermediate_layers(
            images_norm, n=layer_indices
        )
    return [out[0] for out in outputs]  # list of [B, S, N_patch, dim]


# ---------------------------------------------------------------------------
# Validation — one backbone pass per batch, all heads evaluated, DDP-reduced
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_all_heads(backbone, heads, val_loader, layer_indices, device, world_size):
    for h in heads.values():
        h.eval()

    # flat accumulator: [iou_l0, iou_l1, ..., n_batches]
    stats = torch.zeros(len(layer_indices) + 1, device=device)

    for batch in val_loader:
        images = batch["images"].to(device, dtype=torch.bfloat16)
        masks  = batch["masks"].to(device)
        images_norm = normalize_imagenet(images)

        all_tokens = extract_all_layer_tokens(backbone, images_norm, layer_indices)

        for i, layer_idx in enumerate(layer_indices):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = heads[layer_idx](all_tokens[i], images)
            stats[i] += compute_iou(logits.float(), masks)
            all_tokens[i] = None

        del all_tokens
        stats[-1] += 1.0

    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    n_batches = max(1.0, stats[-1].item())
    for h in heads.values():
        h.train()

    return {l: stats[i].item() / n_batches for i, l in enumerate(layer_indices)}


# ---------------------------------------------------------------------------
# Training worker (one process per GPU)
# ---------------------------------------------------------------------------

def train_worker(rank: int, world_size: int, args):
    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=world_size, rank=rank,
            timeout=datetime.timedelta(seconds=300),
        )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    maybe_print(rank, f"Using {world_size} GPU(s)  |  rank {rank} → {device}")

    out_dir = Path(args.out_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = None
    if is_main(rank) and _TB_AVAILABLE:
        tb_writer = SummaryWriter(log_dir=args.tb_dir)
        print(f"TensorBoard: tensorboard --logdir {args.tb_dir}")

    # ---- Load backbone (staggered to avoid RAM contention) ----
    if use_ddp and rank > 0:
        time.sleep(rank * 15)
    maybe_print(rank, f"Loading DA3 model: {args.da_model} ...")
    da3_wrapper = DepthAnything3.from_pretrained(args.da_model)
    da3_wrapper = da3_wrapper.to(device).to(torch.bfloat16)
    da3_wrapper.eval()
    for p in da3_wrapper.parameters():
        p.requires_grad_(False)

    backbone = get_backbone(da3_wrapper)

    inner = da3_wrapper.model
    if hasattr(inner, "da3"):
        for attr in ("head", "cam_dec", "cam_enc", "gs_head", "gs_adapter"):
            setattr(inner.da3, attr, None)
        inner.da3_metric = None
    else:
        for attr in ("head", "cam_dec", "cam_enc", "gs_head", "gs_adapter"):
            setattr(inner, attr, None)
    gc.collect()
    torch.cuda.empty_cache()

    if use_ddp:
        dist.barrier()

    n_blocks  = backbone.pretrained.n_blocks
    embed_dim = backbone.pretrained.embed_dim
    dim_in    = 2 * embed_dim
    maybe_print(rank, f"Backbone: n_blocks={n_blocks}, embed_dim={embed_dim}, dim_in={dim_in}")

    # ---- Determine layers to ablate ----
    if args.layers is not None:
        layer_indices = sorted(set(args.layers))
        invalid = [l for l in layer_indices if not (0 <= l < n_blocks)]
        if invalid:
            raise ValueError(f"Layer indices out of range [0, {n_blocks-1}]: {invalid}")
    else:
        layer_indices = list(range(0, n_blocks, args.layer_stride))

    maybe_print(rank, f"Ablating {len(layer_indices)} layers: {layer_indices}")

    if is_main(rank):
        H = W = args.img_size
        N_patch = (H // 14) * (W // 14)
        token_bytes = args.batch_size * args.seq_len * N_patch * dim_in * len(layer_indices) * 2
        print(f"Peak GPU memory for all-layer tokens per batch: ~{token_bytes / 1e9:.2f} GB")

    # ---- Datasets ----
    train_dataset = YOLOFloorDataset(
        dataset_root=args.dataset_root, split="train",
        seq_len=args.seq_len, img_size=args.img_size,
    )
    val_dataset = None
    val_split_path = Path(args.dataset_root) / args.val_split
    if val_split_path.exists():
        val_dataset = YOLOFloorDataset(
            dataset_root=args.dataset_root, split=args.val_split,
            seq_len=args.seq_len, img_size=args.img_size,
        )
    else:
        maybe_print(rank, f"Warning: val split not found at {val_split_path}, skipping validation.")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False) if (use_ddp and val_dataset) else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    ) if val_dataset else None

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = steps_per_epoch * args.warmup_epochs

    # ---- One seg head + optimizer + scheduler per layer ----
    heads      = {}
    heads_ddp  = {}
    optimizers = {}
    schedulers = {}
    for l in layer_indices:
        h = FloorSegHead(dim_in=dim_in, patch_size=14).to(device)
        opt = torch.optim.AdamW(h.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = build_scheduler(opt, total_steps, warmup_steps)
        heads[l]      = h
        heads_ddp[l]  = DDP(h, device_ids=[rank], output_device=rank) if use_ddp else h
        optimizers[l] = opt
        schedulers[l] = sch

    best_val_iou = {l: 0.0 for l in layer_indices}
    history      = {l: []  for l in layer_indices}
    epoch_durations = []  # for ETA

    maybe_print(rank, f"\nTraining {len(layer_indices)} heads for {args.epochs} epochs "
                      f"({steps_per_epoch} steps/epoch)\n")

    # ---- Training loop ----
    for epoch in range(args.epochs):
        for h in heads.values():
            h.train()
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = {l: 0.0 for l in layer_indices}
        epoch_start = time.time()
        step_start  = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device, dtype=torch.bfloat16)
            masks  = batch["masks"].to(device)
            images_norm = normalize_imagenet(images)

            # Single backbone forward — all layers at once
            all_tokens = extract_all_layer_tokens(backbone, images_norm, layer_indices)

            # Update each head sequentially; free token tensor immediately after
            for i, l in enumerate(layer_indices):
                tokens = all_tokens[i]
                all_tokens[i] = None

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = heads_ddp[l](tokens, images)

                loss = bce_dice_loss(logits.float(), masks, args.bce_weight, args.pos_weight)
                optimizers[l].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads[l].parameters(), 1.0)
                optimizers[l].step()
                schedulers[l].step()

                epoch_loss[l] += loss.item()
                del tokens

            del all_tokens

            global_step = epoch * steps_per_epoch + step + 1
            if is_main(rank) and tb_writer is not None and (step + 1) % 50 == 0:
                for l in layer_indices:
                    tb_writer.add_scalar(f"train_loss/layer_{l:02d}",
                                         epoch_loss[l] / (step + 1), global_step)

            if is_main(rank) and (step + 1) % 50 == 0:
                avg = {l: epoch_loss[l] / (step + 1) for l in layer_indices}
                best_l  = min(avg, key=avg.get)
                worst_l = max(avg, key=avg.get)
                # Step-level ETA: extrapolate remaining steps + epochs
                elapsed      = time.time() - epoch_start
                secs_per_step = elapsed / (step + 1)
                steps_left   = (steps_per_epoch - step - 1) + steps_per_epoch * (args.epochs - epoch - 1)
                eta_str      = _fmt_duration(secs_per_step * steps_left)
                print(f"  Epoch {epoch+1}/{args.epochs}  Step {step+1}/{steps_per_epoch}  "
                      f"Best loss: L{best_l}={avg[best_l]:.4f}  "
                      f"Worst: L{worst_l}={avg[worst_l]:.4f}  "
                      f"ETA {eta_str}")

        epoch_durations.append(time.time() - epoch_start)

        # ---- Validation ----
        if val_loader is not None:
            val_ious = validate_all_heads(
                backbone, heads, val_loader, layer_indices, device, world_size
            )
            for l in layer_indices:
                history[l].append(val_ious[l])
                if val_ious[l] > best_val_iou[l]:
                    best_val_iou[l] = val_ious[l]
                    if is_main(rank):
                        torch.save({
                            "layer_idx":           l,
                            "dim_in":              dim_in,
                            "best_val_iou":        best_val_iou[l],
                            "seg_head_state_dict": heads[l].state_dict(),
                        }, out_dir / f"best_layer_{l:02d}.pt")

            if is_main(rank):
                avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
                remaining_epochs = args.epochs - epoch - 1
                eta_str = _fmt_duration(avg_epoch_time * remaining_epochs)

                best_l  = max(val_ious, key=val_ious.get)
                worst_l = min(val_ious, key=val_ious.get)
                print(f"Epoch {epoch+1}/{args.epochs}  "
                      f"val IoU best: L{best_l}={val_ious[best_l]:.4f}  "
                      f"worst: L{worst_l}={val_ious[worst_l]:.4f}  "
                      f"ETA {eta_str}")

                if tb_writer is not None:
                    for l in layer_indices:
                        tb_writer.add_scalar(f"val_iou/layer_{l:02d}", val_ious[l], epoch + 1)
                    tb_writer.add_scalar("train/eta_seconds", avg_epoch_time * remaining_epochs, epoch + 1)

                    # Heatmap: rows = layers, cols = epochs completed so far
                    fig = _make_iou_heatmap(layer_indices, history, epoch)
                    tb_writer.add_image("val/iou_heatmap", _fig_to_image(fig),
                                        epoch + 1, dataformats="HWC")
                    plt.close(fig)

                    # Bar chart: best IoU per layer so far
                    fig = _make_iou_bar(layer_indices, best_val_iou, val_ious)
                    tb_writer.add_image("val/iou_bar", _fig_to_image(fig),
                                        epoch + 1, dataformats="HWC")
                    plt.close(fig)

                    tb_writer.flush()
        else:
            for l in layer_indices:
                history[l].append(0.0)

    # ---- Save results (rank 0 only) ----
    if is_main(rank):
        best_layer = max(best_val_iou, key=best_val_iou.get)
        print(f"\nBest layer: {best_layer}  (val IoU={best_val_iou[best_layer]:.4f})")
        print(f"Per-layer checkpoints saved to {out_dir}/best_layer_XX.pt")

        results = {
            "da_model":      args.da_model,
            "layer_indices": layer_indices,
            "best_val_iou":  {str(l): v for l, v in best_val_iou.items()},
            "history":       {str(l): v for l, v in history.items()},
        }
        json_path = out_dir / "ablation_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_path}")

        _plot_results(layer_indices, best_val_iou, history, args.epochs, out_dir)

        if tb_writer is not None:
            tb_writer.close()

    if use_ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# ETA + TensorBoard figures (rank 0 only)
# ---------------------------------------------------------------------------

def _fig_to_image(fig) -> np.ndarray:
    """Render a matplotlib figure to an HxWx3 uint8 numpy array via a PNG buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = plt.imread(buf)          # HxWx4 float32 RGBA in [0,1]
    return (img[:, :, :3] * 255).astype(np.uint8)   # HxWx3 uint8


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _make_iou_heatmap(layer_indices, history, epoch):
    """
    Heatmap: rows = layer index, cols = epoch (1..epoch+1), colour = val IoU.
    Annotates each cell with its IoU value.
    """
    n_epochs = epoch + 1
    data = np.array([[history[l][e] for e in range(n_epochs)] for l in layer_indices])

    fig_h = max(4, len(layer_indices) * 0.35 + 1.5)
    fig_w = max(6, n_epochs * 0.55 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, origin="upper")
    fig.colorbar(im, ax=ax, label="Val IoU", fraction=0.03, pad=0.04)

    ax.set_xticks(range(n_epochs))
    ax.set_xticklabels([str(e + 1) for e in range(n_epochs)], fontsize=8)
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels([str(l) for l in layer_indices], fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer Index")
    ax.set_title(f"Val IoU — all layers × all epochs (epoch {epoch + 1})")

    # Annotate cells if the grid isn't too large
    if len(layer_indices) * n_epochs <= 400:
        for row, l in enumerate(layer_indices):
            for col in range(n_epochs):
                val = data[row, col]
                color = "white" if val < 0.5 else "black"
                ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(5, 8 - n_epochs // 10), color=color)

    fig.tight_layout()
    return fig


def _make_iou_bar(layer_indices, best_val_iou, current_val_ious):
    """
    Grouped bar chart: best-so-far vs current-epoch IoU for every layer.
    Best layer (by best-so-far) is highlighted in a different colour.
    """
    best_layer = max(best_val_iou, key=best_val_iou.get)
    n = len(layer_indices)
    x = np.arange(n)
    width = 0.4

    best_vals    = [best_val_iou[l]       for l in layer_indices]
    current_vals = [current_val_ious.get(l, 0.0) for l in layer_indices]

    fig_w = max(8, n * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    colors = ["tomato" if l == best_layer else "steelblue" for l in layer_indices]
    ax.bar(x - width / 2, best_vals,    width, label="Best so far", color=colors, alpha=0.9)
    ax.bar(x + width / 2, current_vals, width, label="This epoch",  color="grey", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layer_indices],
                       rotation=45 if n > 20 else 0, fontsize=8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Val IoU")
    ax.set_title("Val IoU per Layer  (best-so-far vs current epoch, best layer in red)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Final plot saved to disk (rank 0 only)
# ---------------------------------------------------------------------------

def _plot_results(layer_indices, best_val_iou, history, epochs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iou_values = [best_val_iou[l] for l in layer_indices]
    best_l = layer_indices[int(np.argmax(iou_values))]

    # Bar chart
    ax = axes[0]
    colors = ["tomato" if l == best_l else "steelblue" for l in layer_indices]
    bar_w  = max(1, layer_indices[1] - layer_indices[0]) * 0.8 if len(layer_indices) > 1 else 1
    ax.bar(layer_indices, iou_values, color=colors, width=bar_w)
    ax.set_xlabel("ViT Layer Index")
    ax.set_ylabel("Best Validation IoU")
    ax.set_title("Floor Seg: Best Val IoU vs Layer (best in red)")
    ax.set_xticks(layer_indices[::max(1, len(layer_indices) // 20)])
    ax.grid(axis="y", alpha=0.4)

    # Learning curves
    ax = axes[1]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))
    for i, l in enumerate(layer_indices):
        ax.plot(range(1, epochs + 1), history[l], color=cmap[i],
                label=f"L{l}" if l == best_l else None,
                linewidth=2.0 if l == best_l else 0.8,
                alpha=1.0  if l == best_l else 0.45)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation IoU")
    ax.set_title("IoU Curves per Layer (best highlighted)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=layer_indices[0],
                                                   vmax=layer_indices[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1], label="Layer Index", fraction=0.03, pad=0.04)

    fig.suptitle("FloorSegHead Layer Ablation", fontsize=13)
    fig.tight_layout()

    plot_path = out_dir / "ablation_iou_vs_layer.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Eval-only worker
# ---------------------------------------------------------------------------

def eval_worker(rank: int, world_size: int, args):
    """Load saved per-layer checkpoints from out_dir and evaluate on test_split."""
    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=world_size, rank=rank,
            timeout=datetime.timedelta(seconds=300),
        )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    maybe_print(rank, f"Eval mode  |  {world_size} GPU(s)  |  rank {rank} → {device}")

    out_dir = Path(args.out_dir)

    # ---- Load backbone (staggered) ----
    if use_ddp and rank > 0:
        time.sleep(rank * 15)
    maybe_print(rank, f"Loading DA3 model: {args.da_model} ...")
    da3_wrapper = DepthAnything3.from_pretrained(args.da_model)
    da3_wrapper = da3_wrapper.to(device).to(torch.bfloat16)
    da3_wrapper.eval()
    for p in da3_wrapper.parameters():
        p.requires_grad_(False)

    backbone = get_backbone(da3_wrapper)

    inner = da3_wrapper.model
    if hasattr(inner, "da3"):
        for attr in ("head", "cam_dec", "cam_enc", "gs_head", "gs_adapter"):
            setattr(inner.da3, attr, None)
        inner.da3_metric = None
    else:
        for attr in ("head", "cam_dec", "cam_enc", "gs_head", "gs_adapter"):
            setattr(inner, attr, None)
    gc.collect()
    torch.cuda.empty_cache()

    if use_ddp:
        dist.barrier()

    n_blocks  = backbone.pretrained.n_blocks
    embed_dim = backbone.pretrained.embed_dim
    dim_in    = 2 * embed_dim

    # ---- Determine layer indices ----
    if args.layers is not None:
        layer_indices = sorted(set(args.layers))
        invalid = [l for l in layer_indices if not (0 <= l < n_blocks)]
        if invalid:
            raise ValueError(f"Layer indices out of range [0, {n_blocks-1}]: {invalid}")
    else:
        layer_indices = list(range(0, n_blocks, args.layer_stride))

    maybe_print(rank, f"Evaluating {len(layer_indices)} layers: {layer_indices}")

    # ---- Load checkpoints ----
    heads = {}
    missing = []
    for l in layer_indices:
        ckpt_path = out_dir / f"best_layer_{l:02d}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            ckpt_dim = ckpt.get("dim_in", dim_in)
            h = FloorSegHead(dim_in=ckpt_dim, patch_size=14).to(device)
            h.load_state_dict(ckpt["seg_head_state_dict"])
            h.eval()
            heads[l] = h
            maybe_print(rank, f"  Loaded layer {l:02d}: {ckpt_path.name}  "
                              f"(best_val_iou={ckpt.get('best_val_iou', float('nan')):.4f})")
        else:
            missing.append(l)

    if missing:
        maybe_print(rank, f"Warning: no checkpoint found for layers {missing}, skipping them.")

    if not heads:
        maybe_print(rank, "No checkpoints found. Nothing to evaluate.")
        if use_ddp:
            dist.destroy_process_group()
        return

    eval_layer_indices = sorted(heads.keys())

    # ---- Test dataset ----
    test_split_path = Path(args.dataset_root) / args.test_split
    if not test_split_path.exists():
        maybe_print(rank, f"Error: test split not found at {test_split_path}")
        if use_ddp:
            dist.destroy_process_group()
        return

    test_dataset = YOLOFloorDataset(
        dataset_root=args.dataset_root, split=args.test_split,
        seq_len=args.seq_len, img_size=args.img_size,
    )
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler=test_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    maybe_print(rank, f"Test split: {test_split_path}  ({len(test_dataset)} samples)")

    # ---- Run evaluation ----
    test_ious = validate_all_heads(
        backbone, heads, test_loader, eval_layer_indices, device, world_size
    )

    if is_main(rank):
        print("\n=== Eval-only results ===")
        for l in eval_layer_indices:
            print(f"  Layer {l:02d}:  test IoU = {test_ious[l]:.4f}")
        best_l = max(test_ious, key=test_ious.get)
        print(f"\nBest layer: {best_l}  (test IoU={test_ious[best_l]:.4f})")

        # Save results JSON
        results = {
            "da_model":          args.da_model,
            "test_split":        args.test_split,
            "layer_indices":     eval_layer_indices,
            "test_iou":          {str(l): v for l, v in test_ious.items()},
        }
        json_path = out_dir / "eval_only_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_path}")

        # Plot bar chart
        iou_values = [test_ious[l] for l in eval_layer_indices]
        colors = ["tomato" if l == best_l else "steelblue" for l in eval_layer_indices]
        bar_w = max(1, eval_layer_indices[1] - eval_layer_indices[0]) * 0.8 if len(eval_layer_indices) > 1 else 1
        fig, ax = plt.subplots(figsize=(max(8, len(eval_layer_indices) * 0.45 + 2), 4))
        ax.bar(eval_layer_indices, iou_values, color=colors, width=bar_w)
        ax.set_xlabel("ViT Layer Index")
        ax.set_ylabel("Test IoU")
        ax.set_title(f"Floor Seg: Test IoU vs Layer  [{args.test_split} split]  (best in red)")
        ax.set_xticks(eval_layer_indices[::max(1, len(eval_layer_indices) // 20)])
        ax.grid(axis="y", alpha=0.4)
        fig.tight_layout()
        plot_path = out_dir / "eval_only_iou_vs_layer.png"
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {plot_path}")

    if use_ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation: train FloorSegHead per DA3 ViT layer, plot IoU vs layer"
    )
    parser.add_argument("--dataset_root", type=str,
                        default="/home/dominic/Documents/floor_label/floor_dataset")
    parser.add_argument("--da_model", type=str, default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--layer_stride", type=int, default=1,
                        help="Test every Nth layer. Use 2 to halve runtime.")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Explicit layer indices to test, overrides --layer_stride")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-GPU batch size. Keep small — all layer tokens are in GPU "
                             "memory simultaneously before being freed sequentially.")
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--pos_weight", type=float, default=3.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test",
                        help="Dataset split used for --eval_only evaluation.")
    parser.add_argument("--out_dir", type=str, default="./ablation_floor_seg")
    parser.add_argument("--tb_dir", type=str, default="./runs/floor_seg_ablation")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; load best_layer_XX.pt from --out_dir and "
                             "evaluate each layer on the --test_split dataset.")
    return parser.parse_args()


def main():
    args = parse_args()

    worker_fn = eval_worker if args.eval_only else train_worker

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        worker_fn(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args)
        return

    n_gpus = max(1, torch.cuda.device_count())
    if n_gpus == 1:
        worker_fn(0, 1, args)
    else:
        port = find_free_port()
        mode = "eval" if args.eval_only else "ablation"
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={n_gpus}",
            "--master_addr=localhost",
            f"--master_port={port}",
            os.path.abspath(__file__),
        ] + sys.argv[1:]
        print(f"Launching {n_gpus}-GPU {mode} via torch.distributed.run (port {port})")
        sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
