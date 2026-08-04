"""2026-07-27: renders the class-1 cardinality-2 SIL per-epoch validation service
rate comparison (exponential_prefix vs legacy, 3 rolling-horizon windows) as a
plain PNG table, verified directly against results.json (not just log greps).
"""
import json
import glob

import matplotlib.pyplot as plt

RUNS = {
    ("bi200/ss100", "exp_prefix"): "outputs/sil_training_bi200_ss100_class1_exp_prefix",
    ("bi200/ss100", "legacy"): "outputs/sil_training_bi200_ss100_class1_legacy",
    ("bi200/ss50", "exp_prefix"): "outputs/sil_training_bi200_ss50_class1_exp_prefix",
    ("bi200/ss50", "legacy"): "outputs/sil_training_bi200_ss50_class1_legacy",
    ("bi100/ss50", "exp_prefix"): "outputs/sil_training_bi100_ss50_class1_exp_prefix",
    ("bi100/ss50", "legacy"): "outputs/sil_training_bi100_ss50_class1_legacy",
}
WINDOWS = ["bi200/ss100", "bi200/ss50", "bi100/ss50"]
RULES = ["exp_prefix", "legacy"]


def pooled_service_rate(base: str, epoch: int) -> float:
    files = sorted(glob.glob(f"{base}/*/*/val/epoch_{epoch}/*/results.json"))
    serviced = sum(json.load(open(f))["stats"]["serviced"] for f in files)
    total = sum(json.load(open(f))["stats"]["total_requests"] for f in files)
    return 100 * serviced / total


data = {key: [pooled_service_rate(base, e) for e in range(5)] for key, base in RUNS.items()}

col_labels = ["Epoch"]
for w in WINDOWS:
    for r in RULES:
        col_labels.append(f"{w}\n{r}")

cell_text = []
for e in range(5):
    row = [str(e + 1)]
    for w in WINDOWS:
        for r in RULES:
            row.append(f"{data[(w, r)][e]:.2f}%")
    cell_text.append(row)

fig, ax = plt.subplots(figsize=(11, 2.6))
ax.axis("off")
ax.set_title(
    "SIL Training — Class-1, Cardinality 2\nPooled Validation Service Rate by Epoch",
    fontsize=13, fontweight="bold", pad=14,
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
out_path = "figures_export/sil_c2_scoring_rule_comparison.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
pdf_path = "figures_export/sil_c2_scoring_rule_comparison.pdf"
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved {pdf_path}")
