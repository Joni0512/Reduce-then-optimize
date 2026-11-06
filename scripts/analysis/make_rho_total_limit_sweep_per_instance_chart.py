"""
2026-10-22: per-instance view of the L sweep: RHO baseline service rate of EVERY instance as a function of the total limit L of requests per vehicle
(--max_requests_per_vehicle), vehicle capacity fixed to C=2 requests on board, Li&Lim, cardinality 2, batch_interval=40, step_size=10, no pruner, one run per cell.

One small panel per instance (7 class 1, 7 class 2); title = instance, number of requests, number of vehicles (= routes of the reference solution, li_lim_parser) and how
many requests the busiest vehicle serves without limit. The dashed line is the "no limit" service rate of that instance.
RHO is not bit-deterministic: +-1 request (about 2 pp) per cell is noise.
Sources (L=16 and 20: ..._L16_20_class1.csv / ..._class2.csv, added 2026-10-22): outputs/rho_total_limit_sweep_C2_c2_bi40_ss10_class1.csv, ..._class2.csv, outputs/rho_total_limit_sweep_C2_c2_bi40_ss10.csv (lr109, lr208).

Run: ./venv/bin/python3 scripts/analysis/make_rho_total_limit_sweep_per_instance_chart.py
Output: figures_export/LiLim Dataset/rho_total_limit_sweep_per_instance_C2_c2_bi40_ss10.{png,pdf}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

import argparse as _ap

_p = _ap.ArgumentParser()
_p.add_argument("--bi", type=int, default=40)  # 2026-10-22: window configuration; default 40/10 = first series (several CSV files), others = one CSV each
_p.add_argument("--ss", type=int, default=10)
_a = _p.parse_args()
BI, SS = _a.bi, _a.ss
TAG = f"bi{BI}_ss{SS}"

OUT = Path("figures_export/LiLim Dataset")
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["2", "4", "6", "8", "10", "12", "16", "20", "none"]  # 2026-10-22: L=16 and L=20 added (files ..._L16_20_class1/2.csv)
STEM = "outputs/rho_total_limit_sweep_C2_c2_bi40_ss10"
if (BI, SS) == (40, 10):
    d = pd.concat([pd.read_csv(f"{STEM}_class1.csv"), pd.read_csv(f"{STEM}_class2.csv"), pd.read_csv(f"{STEM}.csv"),
                   pd.read_csv(f"{STEM}_L16_20_class1.csv"), pd.read_csv(f"{STEM}_L16_20_class2.csv")], ignore_index=True)
else:
    d = pd.read_csv(f"outputs/rho_total_limit_sweep_C2_c2_{TAG}.csv")
d["L"] = d["L"].astype(str)
d = d[d.status == "ok"].drop_duplicates(["instance", "L"], keep="last")
d["rate"] = 100 * d["served"] / d["requests"]
COL = {"1": "#276575", "2": "#A6493A"}


def n_vehicles(stem):
    return len(json.load(open(f"solutions/li_lim/manifests/{stem}.json"))["driver_runs"])


def main():
    insts = sorted(d.instance.unique(), key=lambda s: (s[-3] if s[-3].isdigit() else s[-3], s))
    insts = sorted(d.instance.unique(), key=lambda s: (int([c for c in s if c.isdigit()][0]), s))
    fig, axes = plt.subplots(4, 4, figsize=(15, 11.5), sharey=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, inst in zip(axes.flat, insts):
        ax.set_visible(True)
        s = d[d.instance == inst].set_index("L")
        cls = [c for c in inst if c.isdigit()][0]
        ys = [s.loc[c, "rate"] for c in ORDER]
        ax.plot(range(len(ORDER)), ys, "-o", color=COL[cls], linewidth=2, markersize=5, zorder=3)
        ax.axhline(ys[-1], color="#444444", linestyle="--", linewidth=1, zorder=1)
        for i, v in enumerate(ys):
            ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5, color=COL[cls])
        ax.set_title(f"{inst}: {int(s.requests.iloc[0])} requests, {n_vehicles(inst)} vehicles\n"
                     f"no limit: busiest vehicle serves {int(s.loc['none', 'max_served_by_one_vehicle'])}", fontsize=9)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(["2", "4", "6", "8", "10", "12", "16", "20", "no"], fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.set_ylim(0, 112)
        ax.grid(True, linestyle=":", alpha=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Total limit L per vehicle", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Service rate", fontsize=9)
    fig.suptitle(f"RHO baseline (no pruner), service rate per instance vs. total requests per vehicle L (C=2 on board, cardinality 2), batch_interval={BI}/step_size={SS}\n"
                 "blue = Class 1, red = Class 2, dashed = same instance without limit", fontsize=12)
    fig.text(0.5, 0.008, f"Li&Lim, RHO offline, max_cardinality=2, batch_interval={BI}, step_size={SS}, 1 run per cell (+-1 request noise = about 2 pp); capacity C=2 via unit-demand manifest copy,\n"
             f"L via --max_requests_per_vehicle; vehicles = routes of the reference solution; sources: outputs/rho_total_limit_sweep_C2_c2_{TAG}*.csv",
             ha="center", fontsize=7.6, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rho_total_limit_sweep_per_instance_C2_c2_{TAG}.{ext}", dpi=170)
    plt.close(fig)
    print("saved", len(insts), "panels")


if __name__ == "__main__":
    main()
