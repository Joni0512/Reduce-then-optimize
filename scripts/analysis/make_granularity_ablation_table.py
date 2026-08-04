import matplotlib.pyplot as plt

NAVY = "#1E2761"
WHITE = "#FFFFFF"
GOOD_BG = "#E7F3E9"
BAD_BG = "#FCE9E5"

header = ["Configuration", "Val\nseed 42", "Val\nseed 1", "Test\nseed 42", "Test\nseed 1"]
rows = [
    ["Learning cheap / Inference cheap  (1/4)", "73.45%", "62.46%", "64.01%", "58.95%"],
    ["Learning expensive / Inference expensive  (2/8)", "87.58%", "84.34%", "82.83%", "80.04%"],
    ["Learning expensive / Inference cheap", "77.46%", "73.93%", "72.70%", "66.49%"],
    ["Labels expensive, Learning + Inference cheap", "59.50%", "43.36%", "53.26%", "46.74%"],
]
row_bg = {1: GOOD_BG, 3: BAD_BG}

fig, ax = plt.subplots(figsize=(11.4, 3.2))
ax.axis("off")
table = ax.table(
    cellText=rows, colLabels=header, loc="center", cellLoc="center",
    colWidths=[0.44, 0.14, 0.14, 0.14, 0.14],
)
table.auto_set_font_size(False)
table.set_fontsize(11.5)
table.scale(1, 2.3)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(NAVY)
        cell.set_text_props(color=WHITE, weight="bold")
    else:
        cell.set_facecolor(row_bg.get(row - 1, WHITE))
        if col == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02

plt.title(
    "Cross-Granularity Ablation — Learning vs. Inference vs. Labels\n"
    "NYC 1,000-request instances, 50 vehicles, LR = 1e-4 throughout, two seeds\n"
    "cheap = 1/4 min (step_size 60 / batch_interval 240)  ·  expensive = 2/8 min (step_size 120 / batch_interval 480)",
    fontsize=10.5, pad=14,
)
plt.tight_layout()
out_path = "figures_export/nyc_granularity_cross_experiment.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print("saved:", out_path)
pdf_path = "figures_export/nyc_granularity_cross_experiment.pdf"
plt.savefig(pdf_path, bbox_inches="tight")
print("saved:", pdf_path)
