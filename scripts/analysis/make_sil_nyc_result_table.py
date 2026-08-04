"""Renders the NYC SIL (legacy scoring) train/val/test result as a PNG table."""
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
served = [355, 340, 355, 348, 353]
val_total = 532
rates = [s / val_total for s in served]
best_epoch = 1

test_served, test_total = 325, 516

rows = []
for e, s, r in zip(epochs, served, rates):
    marker = "  <- best" if e == best_epoch else ""
    rows.append(["Val", f"{e}", f"{s} / {val_total}", f"{r*100:.2f}%{marker}"])
rows.append([
    "Test", f"(ckpt ep.{best_epoch})", f"{test_served} / {test_total}",
    f"{100*test_served/test_total:.2f}%",
])

fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.axis("off")

col_labels = ["Split", "Epoch", "Served / Total", "Service Rate"]
table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
    colWidths=[0.16, 0.24, 0.28, 0.32],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

n_val_rows = len(epochs)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1E2761")
        cell.set_text_props(color="white", weight="bold")
    elif row - 1 == best_epoch - 1:
        cell.set_facecolor("#E7F3E9")
    elif row - 1 == n_val_rows:  # test row
        cell.set_facecolor("#FCF1EC")

plt.title(
    "NYC SIL — Validation (per epoch) and Test Service Rate\n"
    "(Train: 2016-01-12, Val: 2016-01-13, Test: 2016-01-19, real single days, legacy scoring rule)",
    fontsize=10.5, pad=14,
)
plt.tight_layout()
out_path = "figures_export/nyc_sil_real500_val_service_rate.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print("saved:", out_path)
pdf_path = "figures_export/nyc_sil_real500_val_service_rate.pdf"
plt.savefig(pdf_path, bbox_inches="tight")
print("saved:", pdf_path)
