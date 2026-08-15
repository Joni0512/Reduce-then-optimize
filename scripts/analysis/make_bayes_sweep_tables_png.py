"""Render the GCN L1 and Pool L2 Bayes-sweep tables as PNGs (no LaTeX
compiler available locally, so this mirrors the .tex tables directly)."""
import matplotlib.pyplot as plt

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"
BEST_BG = "#eaf1fb"

GCN_L1_ROWS = [
    ("0.000643", "256", "78.30%", True),
    ("0.000500", "128", "77.67%", False),
    ("0.000891", "256", "77.67%", False),
    ("0.001229", "128", "77.67%", False),
    ("0.004555", "64", "77.36%", False),
    ("0.001068", "128", "77.36%", False),
    ("0.002945", "128", "77.04%", False),
    ("0.003686", "128", "76.42%", False),
    ("0.001452", "64", "75.47%", False),
    ("0.000371", "64", "75.47%", False),
    ("0.000306", "64", "74.84%", False),
]

POOL_L2_ROWS = [
    ("0.001869", "128", "76.42%", True),
    ("0.000943", "128", "76.10%", False),
    ("0.001105", "128", "75.47%", False),
    ("0.000555", "128", "75.47%", False),
    ("0.004709", "64", "75.16%", False),
    ("0.000420", "64", "73.58%", False),
]


def render_table(rows, title, out_path):
    n_rows = len(rows) + 1
    fig_h = 0.5 + 0.4 * n_rows
    fig, ax = plt.subplots(figsize=(5.5, fig_h))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axis("off")

    col_labels = ["Learning Rate", "Hidden Dim", "Best Val Service Rate"]
    cell_text = [[lr, hd, sr] for lr, hd, sr, _ in rows]
    table = ax.table(
        cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_text_props(weight="bold", color=TEXT_PRIMARY)
            cell.set_facecolor("#f2f1ec")
        else:
            is_best = rows[row - 1][3]
            cell.set_facecolor(BEST_BG if is_best else "#fcfcfb")
            weight = "bold" if is_best else "normal"
            cell.set_text_props(color=TEXT_PRIMARY, weight=weight)

    ax.set_title(title, fontsize=11.5, color=TEXT_PRIMARY, pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print("saved:", out_path)


render_table(
    GCN_L1_ROWS,
    "GCN L1 Bayesian HP Sweep\nSIL bi200/ss100 class1 legacy, dropout=0.0, seed=1",
    "figures_export/gcn_l1_bayes_sweep_table.png",
)
render_table(
    POOL_L2_ROWS,
    "Pool L2 Bayesian HP Sweep (preliminary)\nSIL bi200/ss100 class1 legacy, dropout=0.0, seed=1",
    "figures_export/pool_l2_bayes_sweep_table.png",
)
