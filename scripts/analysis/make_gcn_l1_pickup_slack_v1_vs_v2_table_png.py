"""Render the GCN L1 pickup_slack ablation table (v1 vs v2+competition, with/without) as PNG."""
import matplotlib.pyplot as plt

TEXT_PRIMARY = "#0b0b0b"
GRID = "#e3e2dd"
BEST_BG = "#eaf1fb"

rows = [
    ("1", "77.99%", "78.62%", "75.16%", "76.73%"),
    ("2", "77.36%", "78.62%", "77.04%", "76.10%"),
    ("3", "76.42%", "79.25%", "74.21%", "77.36%"),
    ("4", "77.04%", "78.93%", "75.47%", "76.73%"),
    ("5", "77.36%", "79.25%", "73.90%", "76.42%"),
    ("Mean ± Std", "77.23% ± 0.57%", "78.93% ± 0.32%", "75.16% ± 1.24%", "76.67% ± 0.47%"),
]
bold_cells = {(6, 2)}  # overall best mean

fig, ax = plt.subplots(figsize=(9.6, 3.6))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.axis("off")

col_labels = ["Seed", "v1", "v1 +\npickup_slack", "v2+comp", "v2+comp +\npickup_slack"]
cell_text = [[r[0], r[1], r[2], r[3], r[4]] for r in rows]
table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.7)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(GRID)
    if row == 0:
        cell.set_text_props(weight="bold", color=TEXT_PRIMARY)
        cell.set_facecolor("#f2f1ec")
        continue
    is_mean_row = row == len(rows)
    cell.set_facecolor(BEST_BG if is_mean_row and col == 2 else "#fcfcfb")
    weight = "bold" if (row, col) in bold_cells else "normal"
    cell.set_text_props(color=TEXT_PRIMARY, weight=weight)

ax.set_title(
    "GCN L1 — cr_pickup_slack on v1 vs. v2+Competition\nSIL bi200/ss100 class1 legacy, lr=0.001, hidden_dim=128",
    fontsize=11.5, color=TEXT_PRIMARY, pad=14,
)
plt.tight_layout()
out_path = "figures_export/gcn_l1_pickup_slack_v1_vs_v2_table.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_path)
