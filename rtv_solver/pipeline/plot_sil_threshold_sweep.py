"""
2026-07-17: SIL (mode="train" + mode="eval", real trained model) threshold
sweep 0.4/0.5/0.6, bi400/ss200, on the 6 hardcoded VALIDATION_FILES
(lc108, lc109, lr111, lr112, lrc107, lrc108 - LC1/LR1/LRC1 classes only,
see chat: LC2/LR2/LRC2 are excluded from training_loop.py's VALIDATION_FILES
entirely, so this sweep can't speak to those classes at all).

Unlike OPT (avg delta -3 to -71pp depending on class, not monotonic in
threshold) and RHO ohne Learning (avg delta -9.6..+5.9pp, flat across
thresholds), SIL shows a clean, monotonic trend: damage grows steadily with
threshold (t=0.4: -3.2pp, t=0.5: -5.0pp, t=0.6: -10.1pp for request pruner).
Pair pruner: exactly 0pp at every threshold, consistent with RHO and OPT.
"""
import csv
import matplotlib.pyplot as plt

CSV_PATH = "outputs/pruner_comparison_summary/sil_mode_train_eval_bi400_ss200_all_thresholds.csv"

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

baseline = {r["instance"]: float(r["service_rate_pct"]) for r in rows if r["variant"] == "baseline"}
by_key = {}
for r in rows:
    if r["variant"] == "baseline":
        continue
    by_key.setdefault((r["variant"], r["threshold"]), []).append(float(r["service_rate_pct"]))

VARIANTS = ["request", "pair", "both"]
LABELS = {"request": "Request Pruner", "pair": "Pair Pruner", "both": "Both"}
THRESHOLDS = ["0.4", "0.5", "0.6"]
THRESHOLD_COLORS = {"0.4": "#a9cce3", "0.5": "#2e86ab", "0.6": "#0b3d5c"}

avg_baseline = sum(baseline.values()) / len(baseline)
avg_service_rate = {(v, th): sum(by_key[(v, th)]) / len(by_key[(v, th)]) for v in VARIANTS for th in THRESHOLDS}
avg_delta = {k: v - avg_baseline for k, v in avg_service_rate.items()}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("SIL (trainiertes Modell) — Threshold-Vergleich 0.4 / 0.5 / 0.6, bi400/ss200 (Ø über 6 Validierungsinstanzen: LC1/LR1/LRC1)", fontsize=11.5)

n_groups = len(VARIANTS)
n_bars = len(THRESHOLDS)
bar_width = 0.8 / n_bars
group_centers = range(n_groups)

ax = axes[0]
for j, th in enumerate(THRESHOLDS):
    xs = [g - 0.4 + bar_width * j + bar_width / 2 for g in group_centers]
    vals = [avg_delta[(v, th)] for v in VARIANTS]
    bars = ax.bar(xs, vals, width=bar_width, color=THRESHOLD_COLORS[th], label=f"t={th}")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 2 if v >= 0 else -14), ha="center", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Δ Service Rate ggü. Baseline (pp)")
ax.set_ylabel("Δ Service Rate (Prozentpunkte)")
ax.set_xticks(list(group_centers)); ax.set_xticklabels([LABELS[v] for v in VARIANTS])
ax.legend(fontsize=9)

ax = axes[1]
for j, th in enumerate(THRESHOLDS):
    xs = [g - 0.4 + bar_width * j + bar_width / 2 for g in group_centers]
    vals = [avg_service_rate[(v, th)] for v in VARIANTS]
    bars = ax.bar(xs, vals, width=bar_width, color=THRESHOLD_COLORS[th], label=f"t={th}")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8)
ax.axhline(avg_baseline, color="gray", linewidth=1, linestyle="--", label=f"Baseline ({avg_baseline:.1f}%)")
ax.set_title("Service Rate (%)")
ax.set_ylabel("Service Rate (%)")
ax.set_ylim(0, 100)
ax.set_xticks(list(group_centers)); ax.set_xticklabels([LABELS[v] for v in VARIANTS])
ax.legend(fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.92])
out_path = "outputs/pruner_comparison_summary/fig_sil_threshold_sweep.png"
fig.savefig(out_path, dpi=150)
print(f"saved {out_path}")
for k in sorted(avg_service_rate):
    print(k, "service_rate=", round(avg_service_rate[k],1), "delta=", round(avg_delta[k],2))
