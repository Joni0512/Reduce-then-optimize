"""Render the GCN L1 pickup_slack ablation table (v2 with/without) as PNG."""
import matplotlib.pyplot as plt

TEXT_PRIMARY = "#0b0b0b"
GRID = "#e3e2dd"
BEST_BG = "#eaf1fb"

rows = [
    ("1", "75.16%", "76.73%"),
    ("2", "77.04%", "76.10%"),
    ("3", "74.21%", "77.36%"),
    ("4", "75.47%", "76.73%"),
    ("5", "73.90%", "76.42%"),
    ("Mean ± Std", "75.16% ± 1.24%", "76.67% ± 0.47%"),
]
bold_cells = {(2, 1), (3, 2), (6, 2)}  # (table_row incl header=0, col) per-cell winners + mean winner

fig, ax = plt.subplots(figsize=(7.4, 3.5))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.axis("off")

col_labels = ["Seed", "v2 without\npickup_slack (94-dim)", "v2 with\npickup_slack (95-dim)"]
cell_text = [[r[0], r[1], r[2]] for r in rows]
table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10.5)
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
    "GCN L1 — cr_pickup_slack Ablation on Feature Builder v2\nSIL bi200/ss100 class1 legacy, lr=0.001, hidden_dim=128",
    fontsize=11.5, color=TEXT_PRIMARY, pad=14,
)
plt.tight_layout()
out_path = "figures_export/gcn_l1_pickup_slack_ablation_table.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_path)
