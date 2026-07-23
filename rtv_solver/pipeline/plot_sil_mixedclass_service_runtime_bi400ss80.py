"""
2026-07-20: simple Service Rate + Runtime comparison for the mixed-class
SIL experiment (bi400/ss80, t=0.5, 39 training files incl. 16 class-2
instances, lc204 excluded). No orig-6/new-6 breakdown - combined 12-instance
validation set only. Same pattern as plot_sil_mixedclass_service_runtime.py
(bi400/ss200 version).
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

ALL_12 = ["lc108", "lc109", "lr111", "lr112", "lrc107", "lrc108",
          "lc207", "lc208", "lr210", "lr211", "lrc207", "lrc208"]
VARIANTS = ["baseline", "request", "pair", "both"]
VARIANT_LABELS = {"baseline": "SIL (mixed-class,\nno pruner)", "request": "Request Pruner",
                   "pair": "Pair Pruner", "both": "Both"}
VARIANT_COLORS = {"baseline": "#9AA5A9", "request": "#276575", "pair": "#A6493A", "both": "#3E8F6C"}
OUT_DIR = Path("outputs/pruner_comparison_summary")


def find_run_dir(variant):
    base = Path(f"outputs/outputs/sil_training_bi400_ss80_mixedclass_{variant}_t0.5")
    return sorted(base.glob("*/mc2_*"))[-1]


def service_rate(variant):
    run_dir = find_run_dir(variant)
    total_serviced = total_requests = 0
    for inst in ALL_12:
        with open(run_dir / "val" / "epoch_4" / inst / "results.json") as f:
            stats = json.load(f)["stats"]
        total_serviced += stats["serviced"]
        total_requests += stats["total_requests"]
    return 100.0 * total_serviced / total_requests


def runtime_minutes(variant):
    run_dir = find_run_dir(variant)
    timestamps = []
    with open(run_dir / "assignment_data.jsonl") as f:
        for line in f:
            d = json.loads(line)
            timestamps.append(d["time"])
    first = datetime.fromisoformat(timestamps[0])
    last = datetime.fromisoformat(timestamps[-1])
    return (last - first).total_seconds() / 60.0


def make_bar_figure(data, ylabel, title, out_path, value_fmt="{:.1f}", ylim=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"SIL bi400/ss80, t=0.5 — Mixed-Class Training\n{title}", fontsize=12.5)
    xs = range(len(VARIANTS))
    vals = [data[v] for v in VARIANTS]
    colors = [VARIANT_COLORS[v] for v in VARIANTS]
    bars = ax.bar(xs, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(value_fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=9.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    service = {}
    runtime = {}
    for variant in VARIANTS:
        service[variant] = service_rate(variant)
        runtime[variant] = runtime_minutes(variant)
        print(f"{variant:10s}  service={service[variant]:5.1f}%   runtime={runtime[variant]:6.1f}min")

    make_bar_figure(service, "Service Rate (%)", "Service Rate (combined 12-instance val set)",
                     OUT_DIR / "fig_sil_mixedclass_bi400ss80_service_rate.png", ylim=(0, 100))
    make_bar_figure(runtime, "Runtime (minutes, full 5-epoch training run)", "Runtime",
                     OUT_DIR / "fig_sil_mixedclass_bi400ss80_runtime.png")
