"""Renders the NYC fleet-size sweep (RHO 2/8, train instance) as a PNG table."""
import matplotlib.pyplot as plt

vehicles = [10, 20, 30, 40, 50, 60]
served = [284, 562, 712, 779, 797, 823]
total = 1000
rates = [s / total for s in served]
runtimes = ["165s", "<10s", "120s", "181s", "190s", "80s"]

rows = []
for v, s, r, rt in zip(vehicles, served, rates, runtimes):
    rows.append([f"{v}", f"{s} / {total}", f"{r*100:.1f}%", rt])

fig, ax = plt.subplots(figsize=(6.6, 2.9))
ax.axis("off")

col_labels = ["Vehicles", "Served / Total", "Service Rate", "Runtime"]
table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1E2761")
        cell.set_text_props(color="white", weight="bold")
    elif row == len(rows):  # last row (60 vehicles) highlighted
        cell.set_facecolor("#E7F3E9")

plt.title(
    "NYC Fleet-Size Sweep — RHO (step 2min / horizon 8min)\n"
    "Day A (2016-01-12, 06:00-14:00, 1,000 requests) - no learning, so no train/val/test here",
    fontsize=10.5, pad=14,
)
plt.tight_layout()
out_path = "figures_export/nyc_fleet_sweep_rho28_table.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print("saved:", out_path)
pdf_path = "figures_export/nyc_fleet_sweep_rho28_table.pdf"
plt.savefig(pdf_path, bbox_inches="tight")
print("saved:", pdf_path)
