"""2026-07-28: renders a per-epoch validation service rate comparison between
cardinality 2 and cardinality 3 (both bi200/ss100, exponential_prefix, class-1,
seed 42) as a plain PNG table, verified directly against results.json.
"""
import json
import glob

import matplotlib.pyplot as plt

RUNS = {
    "Cardinality 2": "outputs/sil_training_bi200_ss100_class1_exp_prefix",
    "Cardinality 3": "outputs/sil_training_bi200_ss100_c3_class1_exp_prefix",
}


def pooled_service_rate(base: str, epoch: int) -> float:
    files = sorted(glob.glob(f"{base}/*/*/val/epoch_{epoch}/*/results.json"))
    serviced = sum(json.load(open(f))["stats"]["serviced"] for f in files)
    total = sum(json.load(open(f))["stats"]["total_requests"] for f in files)
    return 100 * serviced / total


data = {key: [pooled_service_rate(base, e) for e in range(5)] for key, base in RUNS.items()}

col_labels = ["Epoch", "Cardinality 2", "Cardinality 3"]
cell_text = []
for e in range(5):
    row = [str(e + 1), f'{data["Cardinality 2"][e]:.2f}%', f'{data["Cardinality 3"][e]:.2f}%']
    cell_text.append(row)

fig, ax = plt.subplots(figsize=(6, 2.6))
ax.axis("off")
ax.set_title(
    "SIL Training — bi200/ss100, exponential_prefix, Class-1, Seed 42\n"
    "Pooled Validation Service Rate: Cardinality 2 vs. 3",
    fontsize=11, fontweight="bold", pad=14,
)

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.0)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#c4cbd2")
    if row == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#e9ecf0")
    elif row % 2 == 0:
        cell.set_facecolor("#f7f8f9")
    if col == 0:
        cell.set_text_props(fontweight="bold")

fig.tight_layout()
out_path = "figures_export/sil_c2_vs_c3_bi200ss100_comparison.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
pdf_path = "figures_export/sil_c2_vs_c3_bi200ss100_comparison.pdf"
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved {pdf_path}")
