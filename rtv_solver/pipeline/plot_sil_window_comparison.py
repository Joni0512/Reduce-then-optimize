"""
2026-07-17: RHO mit Structured Imitation Learning (SIL) - service rate,
threshold 0.5, seed 42, comparing the two window configs tested so far:
bi200/ss100 (more frequent re-optimization) vs bi400/ss200 (less frequent).
Counter-intuitive finding: bi200/ss100 has a much better baseline but a
FAR larger request-pruner penalty (-31.6pp avg vs -5.0pp at bi400/ss200) -
smaller windows mean fewer candidate trips to begin with, so pruning hits
the already-thin candidate pool harder under y*-matching's exact-prefix-
match requirement.
"""
import matplotlib.pyplot as plt

VARIANTS = ["Baseline", "Request Pruner", "Pair Pruner", "Both"]
CONFIG_COLORS = {"bi200/ss100": "#2e86ab", "bi400/ss200": "#c17a3d"}

# service rate (%), avg over the 6 validation instances, threshold 0.5, seed 42
SERVICE_RATE = {
    "bi200/ss100": {"Baseline": 75.52, "Request Pruner": 43.97, "Pair Pruner": 75.52, "Both": 43.97},
    "bi400/ss200": {"Baseline": 54.08, "Request Pruner": 49.1, "Pair Pruner": 54.1, "Both": 48.4},
}
# per-instance delta (request pruner vs baseline), for the annotated second panel
INSTANCES = ["lc108", "lc109", "lr111", "lr112", "lrc107", "lrc108"]
DELTA_BY_INSTANCE = {
    "bi200/ss100": {"lc108": 0.0, "lc109": 0.0, "lr111": -33.4, "lr112": -33.9, "lrc107": -60.4, "lrc108": -61.6},
    "bi400/ss200": {"lc108": -5.6, "lc109": -17.0, "lr111": -1.9, "lr112": -1.8, "lrc107": -1.9, "lrc108": -1.9},
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("RHO mit Structured Imitation Learning (SIL) — Fenster-Vergleich bei Threshold 0.5, Seed 42\n"
             "häufigere Reoptimierung (bi200/ss100) schützt NICHT vor Pruner-Schaden — im Gegenteil", fontsize=12)

# Panel 1: average service rate by variant, both configs
ax = axes[0]
x = range(len(VARIANTS))
bar_width = 0.35
for i, cfg in enumerate(["bi200/ss100", "bi400/ss200"]):
    vals = [SERVICE_RATE[cfg][v] for v in VARIANTS]
    xs = [xi + (i - 0.5) * bar_width for xi in x]
    bars = ax.bar(xs, vals, width=bar_width, color=CONFIG_COLORS[cfg], label=cfg)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8)
ax.set_title("Ø Service Rate (%)")
ax.set_ylabel("Service Rate (%)")
ax.set_ylim(0, 100)
ax.set_xticks(list(x)); ax.set_xticklabels(VARIANTS, rotation=15, ha="right", fontsize=9)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)

# Panel 2: per-instance delta (request pruner vs baseline), both configs
ax = axes[1]
xi = range(len(INSTANCES))
for i, cfg in enumerate(["bi200/ss100", "bi400/ss200"]):
    vals = [DELTA_BY_INSTANCE[cfg][inst] for inst in INSTANCES]
    xs = [x_ + (i - 0.5) * bar_width for x_ in xi]
    bars = ax.bar(xs, vals, width=bar_width, color=CONFIG_COLORS[cfg], label=cfg)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 2 if v >= 0 else -12), ha="center", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Δ Service Rate: Request Pruner vs. Baseline (pp)")
ax.set_ylabel("Δ Service Rate (Prozentpunkte)")
ax.set_xticks(list(xi)); ax.set_xticklabels(INSTANCES, rotation=15, ha="right", fontsize=9)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)

fig.tight_layout(rect=[0, 0, 1, 0.88])
out_path = "outputs/pruner_comparison_summary/fig_sil_window_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"saved {out_path}")
