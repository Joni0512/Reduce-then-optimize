"""Render the GCN L1 grid-best vs. Bayes-best 5-seed comparison table as PNG."""
import matplotlib.pyplot as plt

TEXT_PRIMARY = "#0b0b0b"
GRID = "#e3e2dd"
BEST_BG = "#eaf1fb"

rows = [
    ("1", "77.99%", "78.30%"),
    ("2", "77.36%", "77.04%"),
    ("3", "76.42%", "76.42%"),
    ("4", "77.04%", "76.42%"),
    ("5", "77.36%", "76.42%"),
    ("Mean ± Std", "77.23% ± 0.57%", "76.92% ± 0.82%"),
]
best_row = {0: True, 5: True}  # grid-best column wins seed 1? no -> per-cell below

fig, ax = plt.subplots(figsize=(7.2, 3.4))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.axis("off")

col_labels = ["Seed", "Grid-best\n(lr=0.001, h=128)", "Bayes-best\n(lr=0.000643, h=256)"]
cell_text = [[r[0], r[1], r[2]] for r in rows]
table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 1.7)

# bold winners: seed1 -> bayes col; mean row -> grid col
bold_cells = {(1, 2), (6, 1)}  # (row_in_table incl header=0, col)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(GRID)
    if row == 0:
        cell.set_text_props(weight="bold", color=TEXT_PRIMARY)
        cell.set_facecolor("#f2f1ec")
        continue
    is_mean_row = row == len(rows)
    cell.set_facecolor(BEST_BG if is_mean_row and col == 1 else "#fcfcfb")
    weight = "bold" if (row, col) in bold_cells else "normal"
    cell.set_text_props(color=TEXT_PRIMARY, weight=weight)

ax.set_title(
    "GCN L1 — Grid-best vs. Bayes-best, 5 Seeds\nSIL bi200/ss100 class1 legacy, dropout=0.0",
    fontsize=11.5, color=TEXT_PRIMARY, pad=14,
)
plt.tight_layout()
out_path = "figures_export/gcn_l1_grid_vs_bayes_5seed_comparison_table.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_path)
