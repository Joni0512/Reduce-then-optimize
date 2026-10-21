"""
2026-09-24: Scatter of per-instance pruning rate vs. service rate for the bi200/ss100
mixed-class expert-prefix-label Request Pruner run (seed 42, correct best-val epoch = 0
after the 2026-09-24 epoch-index fix), across all 12 val instances. Shows the pruning
rate essentially explains the lc-vs-lr/lrc service-rate gap: lc instances get pruned
lightly (4-14%) and stay near-fully served, lr/lrc instances get pruned much harder
(31-56%) and collapse toward 0% service.

Source data (from summarize_per_instance_mixedclass.py's results.json reads + a manual
per-instance pruning-rate parse of the run's own .log, both 2026-09-24, see chat).

Run: python3 -m scripts.analysis.make_pruning_vs_service_rate_scatter
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA = {
    "lc108": (3.8, 94.3), "lc109": (4.1, 92.5), "lc207": (5.2, 89.0), "lc208": (13.5, 91.0),
    "lr111": (30.6, 25.9), "lrc107": (32.7, 7.5), "lrc108": (34.3, 5.8), "lr112": (35.6, 15.1),
    "lr210": (53.1, 0.0), "lrc207": (51.4, 0.4), "lr211": (56.2, 1.2), "lrc208": (56.1, 0.0),
}

fig, ax = plt.subplots(figsize=(8, 6.2))
for inst, (pruned, served) in DATA.items():
    is_lc = inst.startswith("lc")
    color = "#276575" if is_lc else "#A6493A"
    ax.scatter(pruned, served, color=color, s=90, zorder=3, edgecolor="white", linewidth=0.8)
    ax.annotate(inst, (pruned, served), textcoords="offset points", xytext=(6, 4), fontsize=9)

ax.scatter([], [], color="#276575", s=90, label="lc (clustered)")
ax.scatter([], [], color="#A6493A", s=90, label="lr / lrc (random / mixed)")
ax.legend(loc="upper right", fontsize=10, frameon=False)

ax.set_xlabel("Request Pruner: share pruned per instance (%)", fontsize=11)
ax.set_ylabel("Service rate (%)", fontsize=11)
ax.set_xlim(0, 62)
ax.set_ylim(-5, 100)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
ax.set_title("Pruning rate vs. service rate, per val instance\nbi200/ss100 mixed-class, expert-prefix-label pruner, seed 42, best-val epoch",
             fontsize=12)
ax.grid(True, linestyle=":", alpha=0.4, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()

out = "figures_export/Request Pruner SIL/pruning_vs_service_rate_scatter"
fig.savefig(f"{out}.png", dpi=200)
fig.savefig(f"{out}.pdf")
print(f"saved {out}.png / .pdf")
