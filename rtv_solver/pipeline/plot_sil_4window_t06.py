"""
2026-07-18: t=0.6 counterpart to plot_sil_4window_comparison.py (t=0.5) and
analyze_pruner_damage_mechanism.py (t=0.5). Builds the same 3 figures
(service rate, runtime, damage mechanism) for threshold 0.6 across all 4
window configs, now that the local t=0.6 sweep finished for all 4 configs.
Baseline (no pruner) is threshold-independent, reused from t=0.5 runs.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

CONFIGS = ["bi200_ss40", "bi200_ss100", "bi400_ss80", "bi400_ss200"]
CONFIG_LABELS = {"bi200_ss40": "bi200/ss40", "bi200_ss100": "bi200/ss100",
                  "bi400_ss80": "bi400/ss80", "bi400_ss200": "bi400/ss200"}
VARIANTS = ["baseline", "request", "pair", "both"]
VARIANT_LABELS = {"baseline": "SIL (no pruner)", "request": "Request Pruner",
                   "pair": "Pair Pruner", "both": "Both"}
VARIANT_COLORS = {"baseline": "#9AA5A9", "request": "#276575", "pair": "#A6493A", "both": "#3E8F6C"}
VAL_INSTANCES = ["lc108", "lc109", "lr111", "lr112", "lrc107", "lrc108"]
OUT_DIR = Path("outputs/pruner_comparison_summary")

# Baseline (no pruner) values, threshold-independent, reused from t=0.5 runs.
BASELINE_SERVICE_RATE = {"bi200_ss40": 84.27, "bi200_ss100": 75.52, "bi400_ss80": 75.18, "bi400_ss200": 54.08}
BASELINE_RUNTIME_MIN = {"bi200_ss40": 32.70, "bi200_ss100": 19.58, "bi400_ss80": 23.98, "bi400_ss200": 18.77}


def find_run_dir(cfg, variant, thr):
    base = Path(f"outputs/outputs/sil_training_{cfg}_{variant}_{thr}")
    matches = list(base.glob("*/mc2_*"))
    return matches[0] if matches else None


def service_rate(cfg, variant):
    if variant == "baseline":
        return BASELINE_SERVICE_RATE[cfg]
    run_dir = find_run_dir(cfg, variant, "t0.6")
    total_serviced = total_requests = 0
    for inst in VAL_INSTANCES:
        with open(run_dir / "val" / "epoch_4" / inst / "results.json") as f:
            stats = json.load(f)["stats"]
        total_serviced += stats["serviced"]
        total_requests += stats["total_requests"]
    return 100.0 * total_serviced / total_requests


def runtime_minutes(cfg, variant):
    if variant == "baseline":
        return BASELINE_RUNTIME_MIN[cfg]
    log_path = Path(f"sil_training_{cfg}_{variant}_t0.6.log")
    timestamps = re.findall(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})", log_path.read_text(), re.MULTILINE)
    from datetime import datetime
    first = datetime.strptime(timestamps[0][0], "%Y-%m-%d %H:%M:%S")
    last = datetime.strptime(timestamps[-1][0], "%Y-%m-%d %H:%M:%S")
    return (last - first).total_seconds() / 60.0


def ever_served_set(assignment_log_path):
    served = set()
    with open(assignment_log_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("message") != "Status":
                continue
            served.update(str(k) for k in d["extra"]["status"].get("assigned_requests", {}).keys())
    return served


def direct_miss_rate(cfg, variant):
    use_thr = "t0.5" if variant == "baseline" else "t0.6"
    run_dir = find_run_dir(cfg, variant, use_thr)
    total_baseline_served = total_missed = 0
    for inst in VAL_INSTANCES:
        baseline_log = Path(f"outputs/eval_rh_no_learning_{cfg}/{inst}/rh_no_learning/assignment_data.jsonl")
        baseline_served = ever_served_set(baseline_log)
        with open(run_dir / "val" / "epoch_4" / inst / "results.json") as f:
            pruned_served = set(str(r) for r in json.load(f)["stats"]["serviced_requests"])
        total_baseline_served += len(baseline_served)
        total_missed += len(baseline_served - pruned_served)
    return 100.0 * total_missed / total_baseline_served if total_baseline_served else 0.0


def fallback_rate(cfg, variant):
    use_thr = "t0.5" if variant == "baseline" else "t0.6"
    run_dir = find_run_dir(cfg, variant, use_thr)
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


def make_panel(ax, data, ylabel, title, value_fmt="{:.1f}"):
    n_groups = len(CONFIGS)
    n_bars = len(VARIANTS)
    bar_width = 0.8 / n_bars
    group_centers = range(n_groups)
    for j, variant in enumerate(VARIANTS):
        xs = [g - 0.4 + bar_width * j + bar_width / 2 for g in group_centers]
        vals = [data[cfg][variant] for cfg in CONFIGS]
        bars = ax.bar(xs, vals, width=bar_width, color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant])
        for b, v in zip(bars, vals):
            ax.annotate(value_fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(group_centers))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS], fontsize=9.5)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def make_single_figure(data, ylabel, title, out_path, ylim=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(f"RHO with Structured Imitation Learning (SIL) — {title}\n"
                 "All 4 window configs tested, threshold 0.6, seed 42", fontsize=12.5)
    make_panel(ax, data, ylabel, "")
    ax.set_xlabel("Window config (batch_interval / step_size)")
    if ylim:
        ax.set_ylim(*ylim)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def make_damage_figure(direct_miss, fallback, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("SIL Request-Pruner Damage Mechanism — All 4 Window Configs, Threshold 0.6, Seed 42",
                 fontsize=13)
    make_panel(axes[0], direct_miss, "Direct miss rate (%)\nof RHO-baseline-served requests",
               "(A) Direct Pruning Damage\n(vs. RHO no-learning, in-vivo live SIL rollout)")
    make_panel(axes[1], fallback, "Fallback rate (%)\nof vehicle decisions",
               "(B) Training-Time Cascade\n(y*-matching reject fallback)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    service = {cfg: {} for cfg in CONFIGS}
    runtime = {cfg: {} for cfg in CONFIGS}
    direct_miss = {cfg: {} for cfg in CONFIGS}
    fallback = {cfg: {} for cfg in CONFIGS}

    for cfg in CONFIGS:
        for variant in VARIANTS:
            service[cfg][variant] = service_rate(cfg, variant)
            runtime[cfg][variant] = runtime_minutes(cfg, variant)
            direct_miss[cfg][variant] = direct_miss_rate(cfg, variant)
            fallback[cfg][variant] = fallback_rate(cfg, variant)
            print(f"{cfg:14s} {variant:10s}  service={service[cfg][variant]:5.1f}%  "
                  f"runtime={runtime[cfg][variant]:5.1f}min  direct_miss={direct_miss[cfg][variant]:5.1f}%  "
                  f"fallback={fallback[cfg][variant]:5.1f}%")

    make_single_figure(service, "Service Rate (%)", "Service Rate by Window Config",
                        OUT_DIR / "fig_sil_4window_service_rate_t06.png", ylim=(0, 100))
    make_single_figure(runtime, "Runtime (minutes, full 5-epoch training run)", "Runtime by Window Config",
                        OUT_DIR / "fig_sil_4window_runtime_t06.png")
    make_damage_figure(direct_miss, fallback, OUT_DIR / "fig_sil_damage_mechanism_t06.png")
