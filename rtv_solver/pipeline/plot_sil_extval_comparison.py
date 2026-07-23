"""
2026-07-19: figures for the extended-validation-set experiment (bi400/ss200,
t=0.5) - tests whether the request-pruner damage mechanism, established all
session on the standard 6 LC1/LR1/LRC1 validation instances, also holds on
the 6 LC2/LR2/LRC2 test instances (part of the pruners' own held-out test
set, but never covered by any SIL evaluation before this run). Uses the new
--extra_validation_files flag (rtv_solver/main.py) which adds instances to
training_loop.VALIDATION_FILES without touching TRAINING_FILES.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ORIGINAL_6 = ["lc108", "lc109", "lr111", "lr112", "lrc107", "lrc108"]
NEW_6 = ["lc207", "lc208", "lr210", "lr211", "lrc207", "lrc208"]
INSTANCE_SETS = [("orig-6 (LC1/LR1/LRC1)", ORIGINAL_6), ("new-6 (LC2/LR2/LRC2)", NEW_6),
                  ("combined-12", ORIGINAL_6 + NEW_6)]
SET_COLORS = {"orig-6 (LC1/LR1/LRC1)": "#276575", "new-6 (LC2/LR2/LRC2)": "#A6493A",
              "combined-12": "#3E8F6C"}
VARIANTS = ["baseline", "request", "pair", "both"]
VARIANT_LABELS = {"baseline": "SIL (no pruner)", "request": "Request Pruner",
                   "pair": "Pair Pruner", "both": "Both"}
OUT_DIR = Path("outputs/pruner_comparison_summary")


def find_run_dir(variant):
    base = Path(f"outputs/outputs/sil_training_bi400_ss200_extval_{variant}_t0.5")
    return sorted(base.glob("*/mc2_*"))[-1]


def service_rate(variant, instances):
    run_dir = find_run_dir(variant)
    total_serviced = total_requests = 0
    for inst in instances:
        with open(run_dir / "val" / "epoch_4" / inst / "results.json") as f:
            stats = json.load(f)["stats"]
        total_serviced += stats["serviced"]
        total_requests += stats["total_requests"]
    return 100.0 * total_serviced / total_requests


def ever_served_set(p):
    served = set()
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            if d.get("message") != "Status":
                continue
            served.update(str(k) for k in d["extra"]["status"].get("assigned_requests", {}).keys())
    return served


def direct_miss_rate(variant, instances):
    run_dir = find_run_dir(variant)
    total_baseline_served = total_missed = 0
    for inst in instances:
        baseline_log = Path(f"outputs/eval_rh_no_learning_bi400_ss200/{inst}/rh_no_learning/assignment_data.jsonl")
        baseline_served = ever_served_set(baseline_log)
        with open(run_dir / "val" / "epoch_4" / inst / "results.json") as f:
            pruned_served = set(str(r) for r in json.load(f)["stats"]["serviced_requests"])
        total_baseline_served += len(baseline_served)
        total_missed += len(baseline_served - pruned_served)
    return 100.0 * total_missed / total_baseline_served if total_baseline_served else 0.0


def fallback_rate(variant):
    run_dir = find_run_dir(variant)
    total_fallback = total_considered = 0
    with open(run_dir / "assignment_data.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("message") != "PrunerImitationDiagnostics":
                continue
            stats = d["extra"]["stats"]
            total_fallback += stats["pruner_fallback_vehicles"]
            total_considered += stats["pruner_vehicles_considered"]
    return 100.0 * total_fallback / total_considered if total_considered else 0.0


def make_panel(ax, data, ylabel, title):
    n_groups = len(VARIANTS)
    n_bars = len(INSTANCE_SETS)
    bar_width = 0.8 / n_bars
    group_centers = range(n_groups)
    for j, (set_label, _) in enumerate(INSTANCE_SETS):
        xs = [g - 0.4 + bar_width * j + bar_width / 2 for g in group_centers]
        vals = [data[variant][set_label] for variant in VARIANTS]
        bars = ax.bar(xs, vals, width=bar_width, color=SET_COLORS[set_label], label=set_label)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7.5)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(group_centers))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=9.5)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


if __name__ == "__main__":
    service = {v: {} for v in VARIANTS}
    direct_miss = {v: {} for v in VARIANTS}
    fallback = {}

    for variant in VARIANTS:
        for set_label, instances in INSTANCE_SETS:
            service[variant][set_label] = service_rate(variant, instances)
            direct_miss[variant][set_label] = direct_miss_rate(variant, instances)
        fallback[variant] = fallback_rate(variant)
        print(f"{variant:10s} fallback (whole run, all 12 instances mixed) = {fallback[variant]:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("SIL bi400/ss200, t=0.5 — Does Damage Generalize to Unseen Instance Classes?\n"
                 "orig-6 = validated all session (LC1/LR1/LRC1) | new-6 = never tested before (LC2/LR2/LRC2)",
                 fontsize=12.5)
    make_panel(axes[0], service, "Service Rate (%)", "Service Rate by Instance Set")
    make_panel(axes[1], direct_miss, "Direct Miss Rate (%)\nof RHO-baseline-served requests",
               "Direct Pruning Damage by Instance Set")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    out_path = OUT_DIR / "fig_sil_extval_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
