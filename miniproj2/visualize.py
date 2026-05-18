import argparse
import math
import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(MODULE_DIR, ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(MODULE_DIR, ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model_clean

# Example usage:
# python3 visualize.py 512 --metric bytes_dram
# python3 visualize.py 512 --metric pred_ms
# python3 visualize.py 512 --metric intensity_l2

"""
If you don't specify a metric, it will plot it for all metrics in METRIC_ORDER
python3 visualize.py 512

If you want the plots for (4,4), add it to the TILE_PAIRS list below.
"""

# (4,4) only available for 512, 2048, 8192, 65536
# add (4,4) to list below

TILE_PAIRS = [(8, 8), (16, 16)]
METRIC_ORDER = [
    "intensity_dram",
    "intensity_l2",
    "intensity_l1",
    "intensity_shared_mem",
    "tflops/s",
    "flops",
]
ALL_METRICS = [
    "pred_ms",
    "flops",
    "tflops/s",
    "bytes_dram",
    "bytes_dram_read",
    "bytes_dram_write",
    "bytes_l2",
    "bytes_l2_read",
    "bytes_l2_write",
    "bytes_l1",
    "bytes_l1_read",
    "bytes_l1_write",
    "intensity_dram",
    "intensity_l2",
    "intensity_l1",
    "intensity_shared_mem",
]
ALL_METRIC_CHOICES = ALL_METRICS + [
    "bound",
]
PLOTTABLE_METRICS = [
    "pred_ms",
    "flops",
    "tflops/s",
    "bytes_dram",
    "bytes_dram_read",
    "bytes_dram_write",
    "bytes_l2",
    "bytes_l2_read",
    "bytes_l2_write",
    "bytes_l1",
    "bytes_l1_read",
    "bytes_l1_write",
    "intensity_dram",
    "intensity_l2",
    "intensity_l1",
    "intensity_shared_mem",
]


def _plot_metric(ax, metric, tile_labels):
    values = metric["values"]
    predicted = [v["predicted"] if v["predicted"] is not None else math.nan for v in values]
    actual = [v["actual"] if v["actual"] is not None else math.nan for v in values]
    errors = [v["error_pct"] for v in values]

    x = list(range(len(values)))
    width = 0.34
    pred_x = [i - width / 2 for i in x]
    actual_x = [i + width / 2 for i in x]

    ax.bar(pred_x, predicted, width, label="predicted", color="#4C78A8")
    ax.bar(actual_x, actual, width, label="actual", color="#F58518")

    max_val = max([v for v in predicted + actual if not math.isnan(v)], default=0)
    label_pad = max_val * 0.035 if max_val else 0.05
    ax.set_ylim(0, max_val * 1.18 if max_val else 1)

    for i, error in enumerate(errors):
        pair_top = max(
            predicted[i] if not math.isnan(predicted[i]) else 0,
            actual[i] if not math.isnan(actual[i]) else 0,
        )
        label = f"{error:.1f}%" if error is not None else "n/a"
        ax.text(i, pair_top + label_pad, label, ha="center", va="bottom", fontsize=9)

    ax.set_title(metric["label"])
    ax.set_ylabel(metric["unit"])
    ax.set_xticks(x)
    ax.set_xticklabels(tile_labels)
    ax.grid(axis="y", alpha=0.25)


def plot_for_s(S, metric_name=None, output_dir="plots"):
    if metric_name == "bound":
        raise ValueError("bound is returned by the model but is not numeric, so it cannot be plotted.")
    if metric_name and metric_name not in PLOTTABLE_METRICS:
        raise ValueError(f"Unknown metric {metric_name!r}.")

    metric_order = [metric_name] if metric_name else METRIC_ORDER
    data = model_clean.get_visualization_data(S, tile_pairs=TILE_PAIRS)
    tile_labels = [f"({Br}, {Bc})" for Br, Bc in data["tile_pairs"]]

    if metric_name:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(3, 2, figsize=(13, 12))
        axes = axes.flatten()

    for ax, key in zip(axes, metric_order):
        _plot_metric(ax, data["metrics"][key], tile_labels)

    for ax in axes[len(metric_order):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    title = data["metrics"][metric_order[0]]["label"] if metric_name else "Metrics"
    fig.suptitle(f"Predicted vs Actual {title} for S={S}", y=0.995)
    legend_y = 0.92 if metric_name else 0.965
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, legend_y),
               ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86 if metric_name else 0.92))

    os.makedirs(output_dir, exist_ok=True)
    suffix = metric_name.replace("/", "_") if metric_name else "metrics"
    path = os.path.join(output_dir, f"S{S}_{suffix}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("S", type=int)
    parser.add_argument("--metric", choices=ALL_METRIC_CHOICES)
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    path = plot_for_s(args.S, metric_name=args.metric, output_dir=args.output_dir)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
