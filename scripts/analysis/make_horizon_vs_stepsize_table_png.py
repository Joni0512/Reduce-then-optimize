"""Renders the horizon-vs-step_size grouping table as a PNG."""
import matplotlib.pyplot as plt

NAVY = "#1E2761"
GROUP_BG = ["#F3F6FC", "#FFFFFF", "#FBECE7"]
BORDER = "#D8D8D8"

groups = [
    ("8 min", [
        ("0.5 min", "81.40%", "865 s"),
        ("1 min", "82.60%", "406 s"),
        ("2 min (baseline)", "79.70%", "209 s"),
        ("4 min", "78.90%", "107 s"),
    ]),
    ("16 min", [
        ("4 min", "95.90%", "176 s"),
        ("8 min", "95.20%", "72 s"),
    ]),
    ("240 min", [
        ("60 min", "38.40%", "11,258 s"),
        ("120 min", "26.60%", "5,601 s"),
    ]),
]

row_h = 0.42
header_h = 0.5
group_gap = 0.12
col0_w = 1.6
col1_w = 2.6
col2_w = 2.3
col3_w = 2.3
total_w = col0_w + col1_w + col2_w + col3_w

total_h = header_h
for _, rows in groups:
    total_h += len(rows) * row_h + group_gap

fig, ax = plt.subplots(figsize=(9.2, total_h * 0.78 + 1.3))
ax.set_xlim(0, total_w)
ax.set_ylim(0, total_h + 0.9)
ax.invert_yaxis()
ax.axis("off")

y = 0.15
# header
ax.add_patch(plt.Rectangle((0, y), total_w, header_h, facecolor=NAVY, edgecolor=BORDER))
headers = ["Horizon", "step_size", "Service Rate", "Runtime"]
xpos = [0, col0_w, col0_w + col1_w, col0_w + col1_w + col2_w]
for hx, htext in zip(xpos, headers):
    ax.text(hx + 0.12, y + header_h / 2, htext, va="center", ha="left",
            fontsize=11.5, fontweight="bold", color="white")
y += header_h

for gi, (horizon, rows) in enumerate(groups):
    group_h = len(rows) * row_h
    bg = GROUP_BG[gi % len(GROUP_BG)]
    ax.add_patch(plt.Rectangle((0, y), total_w, group_h, facecolor=bg, edgecolor=BORDER, linewidth=0.6))
    ax.text(col0_w / 2, y + group_h / 2, horizon, va="center", ha="center",
            fontsize=12, fontweight="bold", color=NAVY)
    ry = y
    for step, rate, rt in rows:
        bold = rate == "26.60%"
        ax.text(col0_w + 0.15, ry + row_h / 2, step, va="center", ha="left", fontsize=10.5, color="#222")
        ax.text(col0_w + col1_w + col2_w / 2, ry + row_h / 2, rate, va="center", ha="center",
                fontsize=10.5, color="#B8221A" if bold else "#222", fontweight="bold" if bold else "normal")
        ax.text(col0_w + col1_w + col2_w + col3_w / 2, ry + row_h / 2, rt, va="center", ha="center",
                fontsize=10.5, color="#222")
        ry += row_h
    for gx in [0, col0_w, col0_w + col1_w, col0_w + col1_w + col2_w, total_w]:
        ax.plot([gx, gx], [y, y + group_h], color=BORDER, linewidth=0.6)
    y += group_h + group_gap

ax.text(0, y + 0.05,
         "8 min group: 3.7 pp service-rate spread vs. 8.1$\\times$ runtime spread — step_size mostly affects runtime.\n"
         "240 min group: 11.8 pp service-rate spread — step_size now drives quality (fleet-capacity cap binds; cardinality 2).",
         va="top", ha="left", fontsize=9.5, color="#52514e", style="italic")

ax.text(total_w / 2, -0.35, "RHO — Service Rate & Runtime Grouped by Horizon, Varying step_size",
         ha="center", va="bottom", fontsize=14.5, fontweight="bold", color=NAVY)

plt.tight_layout()
out_path = "figures_export/nyc_rho_horizon_vs_stepsize_table.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print("saved:", out_path)
