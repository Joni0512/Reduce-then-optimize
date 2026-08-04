"""2026-07-31: consolidated overview of all legacy vs exponential_prefix SIL
results gathered so far (cardinality 2 and 3, class-1, no pruner), rendered as
two PNG tables. Reads "Best val service rate" directly from each run's main.log
rather than hardcoding remembered numbers.
"""
import glob
import re

import matplotlib.pyplot as plt

BEST_VAL_RE = re.compile(r"Best val service rate = ([\d.]+) at epoch (\d+)")


def best_val(base: str) -> str:
    logs = sorted(glob.glob(f"{base}/*/*/main.log"))
    for log in reversed(logs):  # prefer the most recent/canonical subfolder
        text = open(log).read()
        matches = BEST_VAL_RE.findall(text)
        if matches:
            rate, epoch = matches[-1]
            return f"{float(rate)*100:.2f}% (E{epoch})"
    return "pending"


# ---------------------------------------------------------------------------
# Cardinality 2
# ---------------------------------------------------------------------------
c2_rows = [
    ("bi100/ss50", "outputs/sil_training_bi100_ss50_class1_legacy",
                    "outputs/sil_training_bi100_ss50_class1_exp_prefix",
                    None, None),
    ("bi200/ss50", "outputs/sil_training_bi200_ss50_class1_legacy",
                    "outputs/sil_training_bi200_ss50_class1_exp_prefix",
                    "outputs/sil_training_bi200_ss50_class1_legacy_seed1",
                    "outputs/sil_training_bi200_ss50_class1_exp_prefix_seed1"),
    ("bi200/ss100", "outputs/sil_training_bi200_ss100_class1_legacy",
                     "outputs/sil_training_bi200_ss100_class1_exp_prefix",
                     "outputs/sil_training_bi200_ss100_class1_legacy_seed1",
                     "outputs/sil_training_bi200_ss100_class1_exp_prefix_seed1"),
]

c2_extra_seeds = {
    "seed 2": "outputs/outputs/sil_training_bi200_ss100_class1_exp_prefix_seed2",
    "seed 3": "outputs/outputs/sil_training_bi200_ss100_class1_exp_prefix_seed3",
    "seed 4": "outputs/outputs/sil_training_bi200_ss100_class1_exp_prefix_seed4",
    "seed 5": "outputs/outputs/sil_training_bi200_ss100_class1_exp_prefix_seed5",
    "seed 6": "outputs/outputs/sil_training_bi200_ss100_class1_exp_prefix_seed6",
}

col_labels_c2 = ["Window", "legacy\n(seed 42)", "exp_prefix\n(seed 42)",
                 "legacy\n(seed 1)", "exp_prefix\n(seed 1)"]
cell_text_c2 = []
for window, leg42, exp42, leg1, exp1 in c2_rows:
    row = [window, best_val(leg42), best_val(exp42)]
    row.append(best_val(leg1) if leg1 else "--")
    row.append(best_val(exp1) if exp1 else "--")
    cell_text_c2.append(row)

extra_row = ["bi200/ss100\nexp_prefix, extra seeds"]
extra_vals = [f"{name}: {best_val(base)}" for name, base in c2_extra_seeds.items()]

# ---------------------------------------------------------------------------
# Cardinality 3
# ---------------------------------------------------------------------------
# 2026-07-31: bi100/ss50-legacy, bi200/ss50-exp_prefix and bi200/ss100-legacy
# turned out to not be locally synced at all (paths from cluster terminal
# transcripts only) -- set to None here so the CLUSTER_ONLY fallback below
# actually fires, instead of best_val() silently returning "pending".
c3_rows = [
    ("bi100/ss50", None, None),
    ("bi200/ss50", None, None),
    ("bi200/ss100", "outputs/sil_training_bi200_ss100_c3_class1_exp_prefix", None),
]

# 2026-07-31: these three are NOT locally synced -- known only from cluster
# terminal transcripts (sacct/grep output pasted by the user), never verified
# against a local results.json. Marked with * rather than silently treated as
# equally verified as the locally-grep'd values.
CLUSTER_ONLY = {
    ("bi100/ss50", "legacy"): "91.19% (E4)*",
    ("bi200/ss50", "exp_prefix"): "90.25% (E3)*",
    ("bi200/ss100", "legacy"): "84.59% (E5)*",
}

col_labels_c3 = ["Window", "exp_prefix\n(seed 42)", "legacy\n(seed 42)"]
cell_text_c3 = []
for window, exp42, leg42 in c3_rows:
    row = [window]
    exp_val = best_val(exp42) if exp42 else CLUSTER_ONLY.get((window, "exp_prefix"), "pending")
    leg_val = best_val(leg42) if leg42 else CLUSTER_ONLY.get((window, "legacy"), "pending")
    row.append(exp_val)
    row.append(leg_val)
    cell_text_c3.append(row)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 7.6),
                          gridspec_kw={"height_ratios": [3.2, 1.1, 3.2]})

def style_table(ax, title, col_labels, cell_text, colw=None):
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center",
                      colWidths=colw)
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
    return table

style_table(axes[0], "Cardinality 2 -- Best Validation Service Rate (Class-1, no pruner)",
            col_labels_c2, cell_text_c2, colw=[0.18, 0.2, 0.2, 0.2, 0.2])

axes[1].axis("off")
axes[1].set_title("bi200/ss100, exp_prefix -- additional seeds (no legacy counterpart)",
                   fontsize=11, fontweight="bold", pad=6)
axes[1].text(0.5, 0.5, "   |   ".join(extra_vals), ha="center", va="center", fontsize=10)

style_table(axes[2], "Cardinality 3 -- Best Validation Service Rate (Class-1, no pruner, seed 42)",
            col_labels_c3, cell_text_c3, colw=[0.25, 0.25, 0.25])

fig.text(0.5, 0.008, "* from cluster sacct/log terminal output pasted during the session -- not re-verified against a local results.json",
          ha="center", fontsize=8.5, style="italic", color="#555555")

fig.suptitle("SIL: legacy vs. exponential_prefix -- current results overview",
             fontsize=15, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0.02, 1, 0.98])
out_path = "figures_export/sil_current_overview.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
pdf_path = "figures_export/sil_current_overview.pdf"
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved {pdf_path}")
