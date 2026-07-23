"""
2026-07-18: Ebene-2 decomposition of the request-pruner's SIL service-rate
damage into two distinct, directly-logged mechanisms (no new runs needed,
everything below is already on disk from the bi{200,400}/ss{40,80,100,200}
t=0.5 seed=42 SIL sweeps):

1. Direct pruning damage: of the requests the UNPRUNED RHO baseline actually
   serves (outputs/eval_rh_no_learning_bi{cfg}/{inst}/rh_no_learning/
   assignment_data.jsonl, union of "assigned_requests" keys across all
   windows = ever-served set), how many does the pruned SIL run's epoch_4
   validation (val/epoch_4/{inst}/results.json -> "serviced_requests") fail
   to serve? This is an in-vivo counterpart to the offline classifier recall
   in sweep_summary_5seed.csv - measured on the actual pruned rollout
   instead of a static held-out test split.

2. Training-time cascade (fallback) rate: PrunerImitationDiagnostics records
   in the top-level assignment_data.jsonl of each training run log
   pruner_fallback_vehicles / pruner_vehicles_considered per window - this
   is how often ALL candidate trips for a vehicle scored <=0 under y*-
   matching, forcing torch.min to pick the reject action (see
   imitation_handler.py's score_combinations_against_solution /
   build_y_star_per_vehicle_from_imit_scores). A vehicle hitting fallback
   contributes a corrupted training signal for that window, independent of
   whether any of ITS requests were ever pruned - this is the mechanism
   behind the "untouched requests also get worse" collateral damage
   discussed in the meeting, not something service rate or offline recall
   can show on their own.
"""
import json
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


def ever_served_set(assignment_log_path):
    served = set()
    with open(assignment_log_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("message") != "Status":
                continue
            served.update(str(k) for k in d["extra"]["status"].get("assigned_requests", {}).keys())
    return served


def find_run_dir(cfg, variant):
    base = Path(f"outputs/outputs/sil_training_{cfg}_{variant}_t0.5")
    matches = list(base.glob("*/mc2_*"))
    if not matches:
        raise FileNotFoundError(f"no run dir under {base}")
    return matches[0]


def direct_miss_rate(cfg, variant):
    run_dir = find_run_dir(cfg, variant)
    total_baseline_served = 0
    total_missed = 0
    for inst in VAL_INSTANCES:
        baseline_log = Path(f"outputs/eval_rh_no_learning_{cfg}/{inst}/rh_no_learning/assignment_data.jsonl")
        baseline_served = ever_served_set(baseline_log)

        results_path = run_dir / "val" / "epoch_4" / inst / "results.json"
        with open(results_path) as f:
            pruned_served = set(str(r) for r in json.load(f)["stats"]["serviced_requests"])

        missed = baseline_served - pruned_served
        total_baseline_served += len(baseline_served)
        total_missed += len(missed)

    return 100.0 * total_missed / total_baseline_served if total_baseline_served else 0.0


def fallback_rate(cfg, variant):
    run_dir = find_run_dir(cfg, variant)
    log_path = run_dir / "assignment_data.jsonl"
    total_fallback = 0
    total_considered = 0
    with open(log_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("message") != "PrunerImitationDiagnostics":
                continue
            stats = d["extra"]["stats"]
            total_fallback += stats["pruner_fallback_vehicles"]
            total_considered += stats["pruner_vehicles_considered"]
    return 100.0 * total_fallback / total_considered if total_considered else 0.0


def collect():
    direct_miss = {cfg: {} for cfg in CONFIGS}
    fallback = {cfg: {} for cfg in CONFIGS}
    for cfg in CONFIGS:
        for variant in VARIANTS:
            dm = direct_miss_rate(cfg, variant)
            fr = fallback_rate(cfg, variant)
            direct_miss[cfg][variant] = dm
            fallback[cfg][variant] = fr
            print(f"{cfg:14s} {variant:10s}  direct_miss={dm:5.1f}%   fallback_rate={fr:5.1f}%")
    return direct_miss, fallback


def make_panel(ax, data, ylabel, title):
    n_groups = len(CONFIGS)
    n_bars = len(VARIANTS)
    bar_width = 0.8 / n_bars
    group_centers = range(n_groups)

    for j, variant in enumerate(VARIANTS):
        xs = [g - 0.4 + bar_width * j + bar_width / 2 for g in group_centers]
        vals = [data[cfg][variant] for cfg in CONFIGS]
        bars = ax.bar(xs, vals, width=bar_width, color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7.5)

    ax.set_ylabel(ylabel)
    ax.set_xticks(list(group_centers))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS], fontsize=9.5)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def make_figure(direct_miss, fallback, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("SIL Request-Pruner Damage Mechanism — All 4 Window Configs, Threshold 0.5, Seed 42",
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
    direct_miss, fallback = collect()
    make_figure(direct_miss, fallback, OUT_DIR / "fig_sil_damage_mechanism.png")
