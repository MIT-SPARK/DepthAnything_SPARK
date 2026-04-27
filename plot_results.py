"""
Plot test IoU vs VGGT layer index from an eval-only ablation results JSON.

Usage:
    python3 plot_results.py
    python3 plot_results.py --results ablation_floor_seg/eval_only_results.json
    python3 plot_results.py --results results.json --output my_plot.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

mpl.rcParams["font.family"] = "Times New Roman"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=str,
        default="./ablation_floor_seg/eval_only_results.json",
        help="Path to eval_only_results.json produced by train_floor_seg_ablation.py --eval_only",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: same dir as results JSON)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = Path(args.results)

    with open(results_path) as f:
        data = json.load(f)

    if "test_iou" in data:
        raw_iou = data["test_iou"]
    else:
        raise KeyError(f"No 'test_iou' key found. Keys: {list(data.keys())}")

    test_iou_map = {int(k): v for k, v in raw_iou.items()}
    layers = sorted(test_iou_map.keys())
    display_layers = [l + 1 for l in layers]  # shift to 1-based for display
    iou_values = [test_iou_map[l] for l in layers]

    best_layer = max(test_iou_map, key=test_iou_map.get)
    best_display = best_layer + 1

    FONT_SIZE = 22
    TICK_SIZE = 20

    bar_w = max(1, display_layers[1] - display_layers[0]) * 0.8 if len(display_layers) > 1 else 1
    colors = ["tomato" if l == best_layer else "steelblue" for l in layers]

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.45 + 2), 5))
    ax.bar(display_layers, iou_values, color=colors, width=bar_w)

    ax.legend(
        handles=[
            Patch(color="tomato", label=f"Best layer (layer {best_display})"),        ],
        fontsize=FONT_SIZE - 2,
        loc="lower right",
    )

    ax.set_ylim(0.3, 1.0)

    ax.set_xlabel("ViT Layer Index", fontsize=FONT_SIZE)
    ax.set_ylabel("Test IoU", fontsize=FONT_SIZE)
    ax.set_xticks(display_layers[:: max(1, len(display_layers) // 20)])
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(axis="y", alpha=0.95)

    plt.tight_layout()

    output_path = args.output or str(results_path.parent / "test_iou_vs_layer.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
