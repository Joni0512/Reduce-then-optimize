"""
2026-07-16: quick matplotlib figure matching the reference chart style the
user pasted (two side-by-side bar panels: runtime reduction % vs baseline,
and service rate %) - but for THIS project's RHO-ohne-Learning pruner
comparison, averaged across all 12 test instances instead of per-instance
(explicit ask: "du musst jetzt nicht jede einzelne Instanz machen, der
Durchschnitt"). Source numbers are the same ones already saved in
outputs/pruner_comparison_summary/rho_no_learning_mode_offline_bi200_ss100_t0.5.csv.
"""
import matplotlib.pyplot as plt

RHO_DATA = {
    "lc108":  {"baseline": {"s": 51, "t": 53, "rt": 1.726}, "request": {"s": 50, "t": 53, "rt": 1.233}, "pair": {"s": 50, "t": 53, "rt": 1.324}, "both": {"s": 50, "t": 53, "rt": 1.301}},
    "lc109":  {"baseline": {"s": 48, "t": 53, "rt": 1.304}, "request": {"s": 49, "t": 53, "rt": 1.324}, "pair": {"s": 48, "t": 53, "rt": 1.164}, "both": {"s": 48, "t": 53, "rt": 1.169}},
    "lc207":  {"baseline": {"s": 46, "t": 51, "rt": 104.721}, "request": {"s": 49, "t": 51, "rt": 13.064}, "pair": {"s": 43, "t": 51, "rt": 34.257}, "both": {"s": 42, "t": 51, "rt": 7.546}},
    "lc208":  {"baseline": {"s": 49, "t": 51, "rt": 9.888}, "request": {"s": 48, "t": 51, "rt": 8.367}, "pair": {"s": 48, "t": 51, "rt": 5.983}, "both": {"s": 49, "t": 51, "rt": 6.874}},
    "lr111":  {"baseline": {"s": 36, "t": 54, "rt": 1.417}, "request": {"s": 36, "t": 54, "rt": 0.602}, "pair": {"s": 36, "t": 54, "rt": 0.908}, "both": {"s": 36, "t": 54, "rt": 0.507}},
    "lr112":  {"baseline": {"s": 35, "t": 53, "rt": 1.538}, "request": {"s": 32, "t": 53, "rt": 0.616}, "pair": {"s": 35, "t": 53, "rt": 1.147}, "both": {"s": 32, "t": 53, "rt": 0.483}},
    "lr210":  {"baseline": {"s": 29, "t": 51, "rt": 2.804}, "request": {"s": 27, "t": 51, "rt": 1.184}, "pair": {"s": 29, "t": 51, "rt": 3.264}, "both": {"s": 27, "t": 51, "rt": 1.021}},
    "lr211":  {"baseline": {"s": 26, "t": 50, "rt": 2.224}, "request": {"s": 24, "t": 50, "rt": 0.987}, "pair": {"s": 26, "t": 50, "rt": 2.034}, "both": {"s": 24, "t": 50, "rt": 1.080}},
    "lrc107": {"baseline": {"s": 39, "t": 53, "rt": 1.145}, "request": {"s": 36, "t": 53, "rt": 0.561}, "pair": {"s": 39, "t": 53, "rt": 0.877}, "both": {"s": 36, "t": 53, "rt": 0.417}},
    "lrc108": {"baseline": {"s": 38, "t": 52, "rt": 1.306}, "request": {"s": 33, "t": 52, "rt": 0.559}, "pair": {"s": 38, "t": 52, "rt": 1.090}, "both": {"s": 33, "t": 52, "rt": 0.445}},
    "lrc207": {"baseline": {"s": 27, "t": 51, "rt": 1.614}, "request": {"s": 26, "t": 51, "rt": 0.858}, "pair": {"s": 27, "t": 51, "rt": 1.487}, "both": {"s": 26, "t": 51, "rt": 0.763}},
    "lrc208": {"baseline": {"s": 27, "t": 51, "rt": 2.286}, "request": {"s": 30, "t": 51, "rt": 1.047}, "pair": {"s": 27, "t": 51, "rt": 1.911}, "both": {"s": 31, "t": 51, "rt": 1.204}},
}

VARIANTS = ["baseline", "request", "pair", "both"]
LABELS = {"baseline": "Baseline", "request": "Request Pruner", "pair": "Pair Pruner", "both": "Both"}
COLORS = {"baseline": "#9AA5A9", "request": "#276575", "pair": "#A6493A", "both": "#3E8F6C"}

instances = list(RHO_DATA.keys())

# service rate (%) per variant, averaged (simple mean of per-instance percentages)
avg_service_rate = {
    v: sum(RHO_DATA[i][v]["s"] / RHO_DATA[i][v]["t"] * 100 for i in instances) / len(instances)
    for v in VARIANTS
}

# runtime reduction vs. baseline (%) per instance, then averaged - matches the
# reference chart's left-panel metric, and avoids the lc207 baseline outlier
# (104.7s) dominating a plain average-of-seconds the way it would otherwise.
avg_runtime_reduction = {"baseline": 0.0}
for v in ["request", "pair", "both"]:
    reductions = [
        (RHO_DATA[i]["baseline"]["rt"] - RHO_DATA[i][v]["rt"]) / RHO_DATA[i]["baseline"]["rt"] * 100
        for i in instances
    ]
    avg_runtime_reduction[v] = sum(reductions) / len(reductions)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("RHO ohne Learning — Pruner-Vergleich, Ø über 12 Testinstanzen (bi200/ss100, t=0.5)", fontsize=13)

x = range(len(VARIANTS))
labels = [LABELS[v] for v in VARIANTS]
colors = [COLORS[v] for v in VARIANTS]

ax = axes[0]
vals = [avg_runtime_reduction[v] for v in VARIANTS]
bars = ax.bar(x, vals, color=colors)
ax.set_title("Laufzeitreduktion ggü. Baseline (%)")
ax.set_ylabel("Laufzeitreduktion ggü. Baseline (%)")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.axhline(0, color="black", linewidth=0.8)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points", xytext=(0, 3 if v >= 0 else -12), ha="center", fontsize=9)

ax = axes[1]
vals = [avg_service_rate[v] for v in VARIANTS]
bars = ax.bar(x, vals, color=colors)
ax.set_title("Service Rate (%)")
ax.set_ylabel("Service Rate (%)")
ax.set_ylim(0, 100)
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out_path = "outputs/pruner_comparison_summary/fig_rho_avg_runtime_service_rate.png"
fig.savefig(out_path, dpi=150)
print(f"saved {out_path}")
print("avg_service_rate:", {k: round(v, 2) for k, v in avg_service_rate.items()})
print("avg_runtime_reduction:", {k: round(v, 2) for k, v in avg_runtime_reduction.items()})
