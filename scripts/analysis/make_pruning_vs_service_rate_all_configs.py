"""
2026-09-24: Combined pruning-rate-vs-service-rate scatter across all 3 window configs
(mixed-class, expert-prefix-label Request Pruner) - refines the single-config finding
from make_pruning_vs_service_rate_scatter.py (bi200/ss100 only). Across all 3 configs,
a simple "more pruning -> less service" correlation does NOT hold uniformly (e.g.
bi400/ss200's lr111/lr112/lrc107/lrc108 are barely pruned at all, 0-4%, yet still only
~32-36% served) - the more robust, cross-config pattern is an INTERACTION: lc instances
(clustered) tolerate even high pruning rates (23-51%) while staying 86-95% served,
lr/lrc instances (random/mixed) collapse at the SAME or even lower pruning rates
(0-59%) to 0-36% served. Same pruning hurts lr/lrc far more than lc.

Pruning rates: seed-42-only (parsed from each config's own .log, eval-mode epoch-1
block). Service rates: 5-seed means from summarize_per_instance_mixedclass.py's
corrected (epoch-index-fixed) output, 2026-09-24. Mixing a single-seed x with a
5-seed-mean y is an approximation (seed 42 is one of the 5 averaged) - good enough to
see the pattern, not seed-matched to the last decimal.

Run: python3 -m scripts.analysis.make_pruning_vs_service_rate_all_configs
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# (pruned_pct_seed42, service_pct_5seed_mean)
DATA = {
    "bi200/ss100": {
        "lc108": (3.8, 93.2), "lc109": (4.1, 90.2), "lc207": (5.2, 94.1), "lc208": (13.5, 94.1),
        "lr111": (30.6, 10.7), "lr112": (35.6, 9.8), "lrc107": (32.7, 2.6), "lrc108": (34.3, 3.1),
        "lr210": (53.1, 0.0), "lr211": (56.2, 1.2), "lrc207": (51.4, 0.0), "lrc208": (56.1, 0.0),
    },
    "bi400/ss80": {
        "lc108": (23.1, 93.2), "lc109": (37.4, 87.5), "lc207": (45.3, 91.4), "lc208": (43.2, 94.5),
        "lr111": (26.4, 20.4), "lr112": (24.6, 16.2), "lrc107": (30.8, 10.6), "lrc108": (24.8, 8.1),
        "lr210": (56.2, 0.8), "lr211": (58.8, 0.0), "lrc207": (56.3, 0.0), "lrc208": (58.6, 0.0),
    },
    "bi400/ss200": {
        "lc108": (31.6, 91.3), "lc109": (42.6, 89.8), "lc207": (47.8, 85.5), "lc208": (51.4, 91.4),
        "lr111": (1.8, 32.2), "lr112": (0.0, 32.1), "lrc107": (3.6, 32.8), "lrc108": (0.0, 32.7),
        "lr210": (46.0, 34.1), "lr211": (50.3, 20.8), "lrc207": (45.1, 17.6), "lrc208": (50.6, 23.9),
    },
}
MARKERS = {"bi200/ss100": "o", "bi400/ss80": "s", "bi400/ss200": "^"}

fig, ax = plt.subplots(figsize=(8.5, 6.5))
for cfg, insts in DATA.items():
    for inst, (pruned, served) in insts.items():
        is_lc = inst.startswith("lc")
        color = "#276575" if is_lc else "#A6493A"
        ax.scatter(pruned, served, color=color, marker=MARKERS[cfg], s=80, zorder=3,
                   edgecolor="white", linewidth=0.7, alpha=0.9)

# legend: color = instance type, marker = config
h1 = ax.scatter([], [], color="#276575", s=80, label="lc (clustered)")
h2 = ax.scatter([], [], color="#A6493A", s=80, label="lr / lrc (random / mixed)")
h3 = ax.scatter([], [], color="gray", marker="o", s=80, label="bi200/ss100")
h4 = ax.scatter([], [], color="gray", marker="s", s=80, label="bi400/ss80")
h5 = ax.scatter([], [], color="gray", marker="^", s=80, label="bi400/ss200")
leg1 = ax.legend(handles=[h1, h2], loc="upper right", fontsize=10, frameon=False, title="instance type")
ax.add_artist(leg1)
ax.legend(handles=[h3, h4, h5], loc="center right", fontsize=9.5, frameon=False, title="config (marker shape)")

ax.set_xlabel("Request Pruner: share pruned per instance, seed 42 (%)", fontsize=11)
ax.set_ylabel("Service rate, 5-seed mean (%)", fontsize=11)
ax.set_xlim(-3, 62)
ax.set_ylim(-5, 100)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
ax.set_title("Same pruning rate hurts lr/lrc far more than lc\nall 3 configs, mixed-class, expert-prefix-label pruner",
             fontsize=12.5)
ax.grid(True, linestyle=":", alpha=0.4, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()

out = "figures_export/Request Pruner SIL/pruning_vs_service_rate_all_configs_scatter"
fig.savefig(f"{out}.png", dpi=200)
fig.savefig(f"{out}.pdf")
print(f"saved {out}.png / .pdf")
