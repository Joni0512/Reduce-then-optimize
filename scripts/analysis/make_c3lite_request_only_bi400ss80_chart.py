"""
2026-11-01: SIL training, c3 mixedclass Lite, batch_interval=400/step_size=80: Request Pruner only (threshold 0.1-0.4) vs. no pruner vs. Pair-only.
Setup (from the run logs): epochs 5, learning_rate 0.0001, seed 42, MLP Request Pruner request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03 (min_retention_fraction 0.3),
keep_active False, max_cardinality 3, lite split (34 train / 12 val files, 623 val requests), one run per variant.
Left: best validation service rate (pooled over the 12 val files; best epoch in brackets). Right: training time (sacct Elapsed, incl. post-training check).
Sources: sil_training_bi400_ss80_c3_mixedclass_lite_reqpruner_t0.{1,2,3,4}_seed42.log (epoch lines "Epoch N: pooled val service rate" and "Best val service rate"),
sacct Elapsed of jobs 5671494-5671497 (Request-only t0.1-0.4), 5670939 (baseline), 5670940 (Pair-only t=0.5).
t0.1/t0.2 epoch values and best val are from earlier notes (logs read 2026-10-26), t0.3/t0.4 from the logs read 2026-11-01.

Run: ./venv/bin/python3 scripts/analysis/make_c3lite_request_only_bi400ss80_chart.py
Output: figures_export/Request Pruner SIL/c3lite_bi400ss80_request_only.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures_export/Request Pruner SIL")
OUT.mkdir(parents=True, exist_ok=True)
CB, CR, CP = "#1E2761", "#276575", "#E8A33D"

# name: (best val %, best epoch, hours Elapsed, colour)
rows = {
    "No pruner":           (80.26, 4, 113.40, CB),  # job 5670939, Elapsed 4-17:23:51
    "Request-only\nt=0.1":  (80.42, 5, 105.2, CR),  # job 5671494, 4-09:12:59
    "Request-only\nt=0.2":  (80.74, 4, 108.9, CR),  # job 5671495, 4-12:52:53
    "Request-only\nt=0.3":  (80.90, 5, 107.1, CR),  # job 5671496, 4-11:08:31
    "Request-only\nt=0.4":  (80.10, 4, 114.4, CR),  # job 5671497, 4-18:25:30
    "Pair-only\nt=0.5":     (78.49, 5, 26.84, CP),  # job 5670940, 1-02:50:31
}
names = list(rows)
base_h = rows["No pruner"][2]
base_v = rows["No pruner"][0]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
x = range(len(names))
a1.bar(x, [rows[n][0] for n in names], color=[rows[n][3] for n in names], width=0.62, zorder=3)
for i, n in enumerate(names):
    v, ep, _, _ = rows[n]
    a1.text(i, v + 0.15, f"{v:.2f} %\n(ep. {ep})", ha="center", fontsize=9, fontweight="bold")
a1.axhline(base_v, color="#888888", ls="--", lw=1, zorder=2)
a1.set_ylim(74, 83)
a1.set_xticks(list(x))
a1.set_xticklabels(names, fontsize=9)
a1.set_ylabel("Best validation service rate (%, pooled over 12 val files)")
a1.set_title("Accuracy (dashed line = no pruner; y-axis starts at 74 %)", fontsize=10.5)

a2.bar(x, [rows[n][2] for n in names], color=[rows[n][3] for n in names], width=0.62, zorder=3)
for i, n in enumerate(names):
    h = rows[n][2]
    a2.text(i, h + 2, f"{h:.1f} h\n{base_h / h:.2f}x", ha="center", fontsize=9, fontweight="bold")
a2.set_ylim(0, 135)
a2.set_xticks(list(x))
a2.set_xticklabels(names, fontsize=9)
a2.set_ylabel("Training time (hours, sacct Elapsed)")
a2.set_title("Training time (number = speed-up vs. no pruner)", fontsize=10.5)
for a in (a1, a2):
    a.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

fig.suptitle("SIL training, c3 mixedclass Lite, batch_interval=400/step_size=80: Request Pruner only vs. no pruner vs. Pair-only", fontsize=12)
fig.text(0.5, 0.012, "5 epochs, learning_rate 0.0001, seed 42, 1 run per variant, 34 train / 12 val files (623 val requests); Request Pruner = MLP (min_retention_fraction 0.3), max_cardinality 3. "
         "Best val is selected on the same 12 val files (optimistic; 1 request = 0.16 pp).\nSources: sil_training_bi400_ss80_c3_mixedclass_lite_reqpruner_t0.X_seed42.log; sacct Elapsed of jobs 5670939, 5670940, 5671494-5671497 (incl. post-training check).",
         ha="center", fontsize=7.6, color="#666666")
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"c3lite_bi400ss80_request_only.{ext}", dpi=170)
print({n.replace("\n", " "): (v, ep, h, round(base_h / h, 2)) for n, (v, ep, h, _) in rows.items()})
