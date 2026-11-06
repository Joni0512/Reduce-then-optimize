"""
2026-10-30: RHO (offline, no pruner) at batch_interval=60/step_size=30, service rate per instance for the 9 VAL_INSTANCES and 9 TEST_INSTANCES of the SRL split.
Bars = mean of 3 runs, dots = single runs (RHO is not bit-deterministic), dashed line = mean over the 9 instances of the 3-run means.
Setup: max_cardinality=2, keep_active=False, seed 42, run by scripts/analysis/rho_val_test_window.py.
Sources: rho_bi60_ss30_run1.log, rho_bi60_ss30_run2.log, rho_bi60_ss30_run3.log (repo root; lines "[rho bi=60 ss=30 ... run=N] <instance>: <rate>").

Run: ./venv/bin/python3 scripts/analysis/make_rho_bi60ss30_per_instance_chart.py
Output: figures_export/SRL/rho_bi60ss30_per_instance.{png,pdf}
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parents[2]
VAL = ["lc103", "lc202", "lr103", "lr107", "lr202", "lr208", "lrc102", "lrc202", "lrc205"]
TEST = ["lc108", "lc205", "lc208", "lr111", "lr205", "lr211", "lrc105", "lrc108", "lrc208"]
rates = {}  # instance -> [run1, run2, run3]
for run in (1, 2, 3):
    for line in open(REPO / f"rho_bi60_ss30_run{run}.log", errors="ignore"):
        m = re.match(r"\[rho bi=60 ss=30 .*run=\d\] ([a-z]+\d+): ([0-9.]+) \(", line)
        if m:
            rates.setdefault(m.group(1), []).append(float(m.group(2)))
assert all(len(rates[i]) == 3 for i in VAL + TEST), {i: len(rates.get(i, [])) for i in VAL + TEST}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, names, title, col in ((axes[0], VAL, "Validation instances", "#276575"), (axes[1], TEST, "Test instances", "#A6493A")):
    mean = np.array([np.mean(rates[i]) for i in names])
    ax.bar(range(len(names)), mean, color=col, alpha=0.8, width=0.62)
    for k, i in enumerate(names):
        ax.scatter([k] * 3, rates[i], color="black", s=11, zorder=3)
        ax.text(k, max(rates[i]) + 0.012, f"{mean[k]:.3f}", ha="center", fontsize=8.5)
    ax.axhline(mean.mean(), color="gray", ls="--", lw=1.3, label=f"mean over 9 instances {mean.mean():.4f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, fontsize=9)
    ax.set_title(title)
    ax.set_ylim(0.3, 1.04)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
axes[0].set_ylabel("Service rate (served / requests)")
fig.suptitle("RHO (no pruner), batch_interval=60 / step_size=30: service rate per instance (bar = mean of 3 runs, dots = single runs)", fontsize=11.5)
fig.text(0.5, 0.01, "Li&Lim, RHO offline, max_cardinality=2, keep_active=False, seed 42, 3 runs (RHO is not bit-deterministic); "
         "source: rho_bi60_ss30_run{1,2,3}.log via scripts/analysis/rho_val_test_window.py", ha="center", fontsize=7.8, color="#666666")
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
out = REPO / "figures_export" / "SRL"
for ext in ("png", "pdf"):
    fig.savefig(out / f"rho_bi60ss30_per_instance.{ext}", dpi=170)
for names, lab in ((VAL, "VAL"), (TEST, "TEST")):
    print(lab, {i: round(float(np.mean(rates[i])), 4) for i in names}, "mean", round(float(np.mean([np.mean(rates[i]) for i in names])), 4))
