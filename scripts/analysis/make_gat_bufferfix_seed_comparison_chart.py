"""
2026-09-11: bar chart of the GAT + TD-bootstrap (gamma=0.99, tau=0.005)
replay-buffer bufferfix results, seeds 43-46, all 12 balanced test
instances - service rate at the final (episode 19) checkpoint, pulled
from srl_training_curves.csv on the cluster (see chat).
"""
import matplotlib.pyplot as plt

INSTANCES = ["lc107", "lc108", "lc207", "lc208", "lr111", "lr112", "lr203", "lr210",
             "lrc107", "lrc108", "lrc201", "lrc207"]
SEEDS = ["43", "44", "45", "46"]
SEED_COLORS = {"43": "#276575", "44": "#4f9d9d", "45": "#c17a3d", "46": "#8c5a9e"}

# service rate (%) at episode 19, srl_training_curves.csv, GAT + TD-bootstrap + bufferfix
SERVICE_RATE = {
    "lc107":  {"43": 96.23, "44": 96.23, "45": 84.91, "46": 79.25},
    "lc108":  {"43": 88.68, "44": 94.34, "45": 86.79, "46": 84.91},
    "lc207":  {"43": 76.47, "44": 80.39, "45": 80.39, "46": 74.51},
    "lc208":  {"43": 74.51, "44": 76.47, "45": 78.43, "46": 78.43},
    "lr111":  {"43": 59.26, "44": 61.11, "45": 64.81, "46": 61.11},
    "lr112":  {"43": 54.72, "44": 54.72, "45": 60.38, "46": 64.15},
    "lr203":  {"43": 54.90, "44": 60.78, "45": 58.82, "46": 64.71},
    "lr210":  {"43": 54.90, "44": 52.94, "45": 52.94, "46": 54.90},
    "lrc107": {"43": 69.81, "44": 64.15, "45": 66.04, "46": 66.04},
    "lrc108": {"43": 65.38, "44": 67.31, "45": 65.38, "46": 63.46},
    "lrc201": {"43": 68.63, "44": 60.78, "45": 60.78, "46": 62.75},
    "lrc207": {"43": 60.78, "44": 49.02, "45": 52.94, "46": 56.86},
}

fig, ax = plt.subplots(figsize=(14, 6))
x = range(len(INSTANCES))
bar_width = 0.2

for i, seed in enumerate(SEEDS):
    vals = [SERVICE_RATE[inst][seed] for inst in INSTANCES]
    xs = [xi + (i - 1.5) * bar_width for xi in x]
    bars = ax.bar(xs, vals, width=bar_width, color=SEED_COLORS[seed], label=f"Seed {seed}")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)

seed_avgs = [sum(SERVICE_RATE[inst].values()) / 4 for inst in INSTANCES]
overall_avg = sum(seed_avgs) / len(seed_avgs)
ax.axhline(overall_avg, color="gray", linestyle="--", linewidth=1)
ax.annotate(f"Overall mean: {overall_avg:.1f}%", (len(INSTANCES) - 1, overall_avg),
            textcoords="offset points", xytext=(0, 6), ha="right", fontsize=9, color="gray")

ax.set_xticks(list(x))
ax.set_xticklabels(INSTANCES, fontsize=10)
ax.set_ylabel("Service Rate (%)")
ax.set_ylim(0, 105)
ax.set_title("GAT Critic + TD-Bootstrap ($\\gamma$=0.99, $\\tau$=0.005) After Replay-Buffer Fix\nService Rate at Episode 19, Seeds 43-46, 12 Balanced Test Instances")
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.legend(fontsize=9, loc="upper right", ncol=4)

fig.tight_layout()
fig.savefig("figures_export/gat_bufferfix_seed_comparison_chart.png", dpi=150)
fig.savefig("figures_export/gat_bufferfix_seed_comparison_chart.pdf")
print("saved figures_export/gat_bufferfix_seed_comparison_chart.png")
print("saved figures_export/gat_bufferfix_seed_comparison_chart.pdf")
