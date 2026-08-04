"""Renders two PNG tables: service rate comparison and runtime comparison,
for RHO 2/8, RHO 1/4, SIL 1/4, and SIL 2/8 (two learning rates each) across
Day A/B/C (1,000-request instances, 50 vehicles)."""
import matplotlib.pyplot as plt

NAVY = "#1E2761"
WHITE = "#FFFFFF"
CARD_BG = "#F4F6FC"
GOOD_BG = "#E7F3E9"
NOTE_BG = "#FCF1EC"


def render_table(rows, col_labels, title, out_path, note_rows=None, figsize=(9.6, 3.6), col_widths=None):
    note_rows = note_rows or set()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=col_labels, loc="center", cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 2.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color=WHITE, weight="bold")
        elif (row - 1) in note_rows:
            cell.set_facecolor(NOTE_BG)
        else:
            cell.set_facecolor(CARD_BG if row % 2 == 1 else WHITE)

    plt.title(title, fontsize=11, pad=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("saved:", out_path)
    pdf_path = out_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print("saved:", pdf_path)


# ---------- Service Rate table ----------
sr_header = ["Configuration", "Day A (Train)", "Day B (Val)", "Day C (Test)"]
sr_rows = [
    ["RHO 2/8", "79.7%", "84.91%", "79.83%"],
    ["RHO 1/4", "70.3%", "75.26%", "67.94%"],
    ["SIL 1/4 (LR=0.0003)", "70.30% (best, ep.5)", "76.03% (best, ep.5)", "67.84%"],
    ["SIL 1/4 (LR=0.0001)", "63.50% (best, ep.5)", "73.45% (best, ep.5)", "64.01%"],
    ["SIL 2/8 (LR=0.0003)", "26.80% (best, ep.1)", "29.42% (best, ep.1)", "27.30%"],
    ["SIL 2/8 (LR=0.0001)", "84.60% (best, ep.4)", "87.58% (best, ep.4)", "82.83%"],
]
render_table(
    sr_rows, sr_header,
    "NYC Service Rate Comparison — RHO vs. SIL, 50 vehicles\n"
    "SIL Day A = eval of the best-val checkpoint replayed on the Day A (train) instance",
    "figures_export/nyc_rho_sil_service_rate_comparison.png",
    col_widths=[0.34, 0.24, 0.24, 0.20],
    figsize=(9.6, 4.6),
)

# ---------- Runtime table ----------
rt_header = ["Configuration", "Day A (Train)", "Day B (Val)", "Day C (Test)"]
rt_rows = [
    ["RHO 2/8", "209s", "225s", "195s"],
    ["RHO 1/4", "380s", "406s", "341s"],
    ["SIL 1/4 (LR=0.0003)", "8,552s combined*", "8,552s combined*", "719s"],
    ["SIL 1/4 (LR=0.0001)", "8,035s combined*", "8,035s combined*", "667s"],
    ["SIL 2/8 (LR=0.0003)", "6,221s combined*", "6,221s combined*", "707s"],
    ["SIL 2/8 (LR=0.0001)", "5,352s combined*", "5,352s combined*", "481s"],
]
render_table(
    rt_rows, rt_header,
    "NYC Runtime Comparison — RHO vs. SIL, 50 vehicles\n"
    "*SIL Train+Val run in one continuous process (5 epochs) - no separately measured Val-only time",
    "figures_export/nyc_rho_sil_runtime_comparison.png",
    note_rows={2, 3, 4, 5},
    figsize=(9.6, 4.6),
    col_widths=[0.30, 0.24, 0.24, 0.22],
)
