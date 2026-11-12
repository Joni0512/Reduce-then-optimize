"""
2026-11-04: SIL training, c3 mixedclass Lite, Request Pruner only (threshold 0.1-0.4) vs. no pruner vs. Pair-only, for two windows:
batch_interval=200/step_size=100 and batch_interval=400/step_size=200 (same style as make_c3lite_request_only_bi400ss80_chart.py).
Setup (from the run logs): epochs 5, learning_rate 0.0001, seed 42, MLP Request Pruner request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03 (min_retention_fraction 0.3),
keep_active False, max_cardinality 3, lite split (34 train / 12 val files, 623 val requests), one run per variant.
Left: best validation service rate (pooled over the 12 val files; best epoch in brackets). Right: training time (sacct Elapsed, incl. post-training check).
Sources: sil_training_bi{200_ss100,400_ss200}_c3_mixedclass_lite_{baseline,pair_t0.5,reqpruner_t0.X}_seed42.log (line "Best val service rate"),
sacct Elapsed: bi200/ss100 jobs 5673889 (baseline), 5673905-5673908 (Request-only t0.1-0.4); bi400/ss200 jobs 5673909 (baseline), 5673925-5673928 (Request-only), 5673910 (Pair-only).
bi200/ss100 Pair-only: first job 5673890 FAILED after 7:45 h with "Disk quota exceeded" (no valid result); the rerun, job 5707345, COMPLETED in 10:28:23 h,
"Best val service rate = 0.6822 at epoch 4" (log sil_training_bi200_ss100_c3_mixedclass_lite_pair_t0.5_seed42.log; added 2026-11-06, values pasted by the user).
Other values read from the cluster logs/sacct output pasted by the user on 2026-11-04.

Run: ./venv/bin/python3 scripts/analysis/make_c3lite_request_only_bi200ss100_bi400ss200_chart.py
Output: figures_export/Request Pruner SIL/c3lite_bi200ss100_request_only.{png,pdf}, c3lite_bi400ss200_request_only.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures_export/Request Pruner SIL")
OUT.mkdir(parents=True, exist_ok=True)
CB, CR, CP = "#1E2761", "#276575", "#E8A33D"


def hms(d, h, m, s):
    return d * 24 + h + m / 60 + s / 3600


# name: (best val %, best epoch, hours Elapsed, colour)
WINDOWS = {
    "bi200ss100": dict(
        title="batch_interval=200/step_size=100",
        rows={
            "No pruner":          (76.73, 5, hms(1, 17, 13, 15), CB),   # job 5673889, 1-17:13:15
            "Request-only\nt=0.1": (77.05, 3, hms(1, 21, 46, 25), CR),  # job 5673905
            "Request-only\nt=0.2": (76.40, 4, hms(1, 15, 57, 5), CR),   # job 5673906
            "Request-only\nt=0.3": (76.73, 3, hms(1, 19, 57, 16), CR),  # job 5673907
            "Request-only\nt=0.4": (77.37, 5, hms(1, 20, 32, 7), CR),   # job 5673908
            "Pair-only\nt=0.5":    (68.22, 4, hms(0, 10, 28, 23), CP),  # job 5707345 (rerun; first job 5673890 died: disk quota), 10:28:23, 2026-11-06
        },
        ylim_acc=(64, 80), ylim_h=(0, 56), ymin_note="64 %",  # 2026-11-06: lower limit 74 -> 64 so the Pair-only bar (68.22 %) fits
        jobs="5673889, 5673905-5673908, 5707345",
        note="Pair-only t=0.5: rerun job 5707345 (the first job 5673890 failed after 7:45 h, disk quota exceeded).",
        file="c3lite_bi200ss100_request_only",
    ),
    "bi400ss200": dict(
        title="batch_interval=400/step_size=200",
        rows={
            "No pruner":          (64.85, 5, hms(0, 18, 6, 30), CB),    # job 5673909, 18:06:30
            "Request-only\nt=0.1": (65.49, 3, hms(0, 18, 4, 56), CR),   # job 5673925
            "Request-only\nt=0.2": (65.01, 5, hms(0, 18, 59, 50), CR),  # job 5673926
            "Request-only\nt=0.3": (65.01, 3, hms(0, 18, 25, 8), CR),   # job 5673927
            "Request-only\nt=0.4": (65.01, 3, hms(0, 18, 40, 52), CR),  # job 5673928
            "Pair-only\nt=0.5":    (60.03, 5, hms(0, 3, 54, 53), CP),   # job 5673910, 3:54:53
        },
        ylim_acc=(56, 68), ylim_h=(0, 24), ymin_note="56 %",
        jobs="5673909, 5673910, 5673925-5673928",
        note="",
        file="c3lite_bi400ss200_request_only",
    ),
}

for key, cfg in WINDOWS.items():
    rows = cfg["rows"]
    names = list(rows)
    base_v, base_h = rows["No pruner"][0], rows["No pruner"][2]
    x = list(range(len(names)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    a1.bar(x, [rows[n][0] for n in names], color=[rows[n][3] for n in names], width=0.62, zorder=3)
    lo, hi = cfg["ylim_acc"]
    pad = (hi - lo) * 0.012
    for i, n in enumerate(names):
        v, ep, _, _ = rows[n]
        a1.text(i, v + pad, f"{v:.2f} %\n(ep. {ep})", ha="center", fontsize=9, fontweight="bold")
    a1.axhline(base_v, color="#888888", ls="--", lw=1, zorder=2)
    a1.set_ylim(lo, hi)
    a1.set_xticks(x)
    a1.set_xticklabels(names, fontsize=9)
    a1.set_ylabel("Best validation service rate (%, pooled over 12 val files)")
    a1.set_title(f"Accuracy (dashed line = no pruner; y-axis starts at {cfg['ymin_note']})", fontsize=10.5)

    a2.bar(x, [rows[n][2] for n in names], color=[rows[n][3] for n in names], width=0.62, zorder=3)
    hpad = cfg["ylim_h"][1] * 0.015
    for i, n in enumerate(names):
        h = rows[n][2]
        a2.text(i, h + hpad, f"{h:.1f} h\n{base_h / h:.2f}x", ha="center", fontsize=9, fontweight="bold")
    a2.set_ylim(*cfg["ylim_h"])
    a2.set_xticks(x)
    a2.set_xticklabels(names, fontsize=9)
    a2.set_ylabel("Training time (hours, sacct Elapsed)")
    a2.set_title("Training time (number = speed-up vs. no pruner)", fontsize=10.5)
    for a in (a1, a2):
        a.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    fig.suptitle(f"SIL training, c3 mixedclass Lite, {cfg['title']}: Request Pruner only vs. no pruner"
                 + (" vs. Pair-only" if "Pair-only\nt=0.5" in rows else ""), fontsize=12)
    foot = ("5 epochs, learning_rate 0.0001, seed 42, 1 run per variant, 34 train / 12 val files (623 val requests); Request Pruner = MLP (min_retention_fraction 0.3), max_cardinality 3. "
            "Best val is selected on the same 12 val files (optimistic; 1 request = 0.16 pp).\n"
            f"Sources: sil_training_{key[:5]}_{key[5:].replace('ss', 'ss')}_c3_mixedclass_lite_*_seed42.log; sacct Elapsed of jobs {cfg['jobs']} (incl. post-training check)."
            + (("\n" + cfg["note"]) if cfg["note"] else ""))
    fig.text(0.5, 0.012, foot, ha="center", fontsize=7.6, color="#666666")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{cfg['file']}.{ext}", dpi=170)
    plt.close(fig)
    print(key, {n.replace("\n", " "): (v, ep, round(h, 2), round(base_h / h, 2)) for n, (v, ep, h, _) in rows.items()})
