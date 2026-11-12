"""
2026-11-11: charts + table of all RHO runs on the SF LODES instances (user request, see chat).
Reads every final/results.json + config.json below the SF RHO output folders (no hard-coded numbers); the run settings
(batch_interval, step_size, time-rule filter, max_cardinality, keep_active, seed) are taken from config.json, never from
folder names. Solver time = stats.total_time of results.json.
Sources:
  * outputs/new_tests/sf_lodes_rho_grid/*            (cluster, rsync'ed; test instance seed 42)
  * outputs/new_tests/sf_lodes_smoke/*               (local; seed 42, batch_interval=900/step_size=450 without filter)
  * outputs/outputs/new_tests/sf_lodes_smoke/*       (local; seed 42, batch_interval=900/step_size=450 with filter)
  * outputs/new_tests/sf_lodes_srl_rho_ref/*         (local; seeds 1/2 = SRL train/val, batch_interval=900/step_size=450 with filter)
All runs: 1000 requests, 50 vehicles, am_capacity 2, OSMnx free-flow road matrix.
Output: figures_export/SF/sf_rho_{service_rate,solver_time,instances}.{png,pdf}, sf_rho_results_table.{csv,tex}
"""
import csv
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_export" / "SF"
GLOBS = [
    "outputs/new_tests/sf_lodes_rho_grid/*/run_*/*/final/results.json",
    "outputs/new_tests/sf_lodes_smoke/*/run_*/*/final/results.json",
    "outputs/outputs/new_tests/sf_lodes_smoke/*/run_*/*/final/results.json",
    "outputs/new_tests/sf_lodes_srl_rho_ref/*/run_*/*/final/results.json",
]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # dataviz reference palette, categorical slots 1-3 (fixed order)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs():
    runs = []
    for pattern in GLOBS:
        for f in sorted(glob.glob(str(ROOT / pattern))):
            f = Path(f)
            cfg = json.loads((f.parent.parent / "config.json").read_text())
            cfg = cfg.get("config_dict", cfg)  # main.py writes {"config_dict": {...}, "git_commit": ..., ...}
            s = json.loads(f.read_text())["stats"]
            inst = int(re.search(r"seed(\d+)_v50", f.parts[-3]).group(1))
            runs.append({
                "instance_seed": inst,
                "batch_interval": int(cfg["BATCH_INTERVAL"]), "step_size": int(cfg["STEP_SIZE"]),
                "time_rule_filter": bool(cfg["USE_REQUEST_TIME_RULE_FILTER"]),
                "max_cardinality": int(cfg["MAX_CARDINALITY"]), "keep_active": bool(cfg["KEEP_ACTIVE"]),
                "solver_seed": int(cfg["SEED"]),
                "served": int(s["serviced"]), "requests": int(s["total_requests"]),
                "service_rate": s["serviced"] / s["total_requests"],
                "wait_s": s["average_wait_time"], "detour_s": s["average_detour"],
                "vmt_over_pmt": s["vmt_over_pmt"], "solver_time_s": s["total_time"],
                "where": "cluster" if "rho_grid" in str(f) else "local",
                "source": str(f.relative_to(ROOT)),
            })
    return runs


def style(ax, ylabel):
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def grouped(runs, key, ylabel, title, fname, fmt, log=False):
    combos = sorted({(r["batch_interval"], r["step_size"]) for r in runs})
    x = np.arange(len(combos))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (flag, color, label) in enumerate(((True, BLUE, "time-rule filter on"), (False, ORANGE, "time-rule filter off"))):
        vals = []
        for c in combos:
            m = [r[key] for r in runs if (r["batch_interval"], r["step_size"]) == c and r["time_rule_filter"] == flag]
            vals.append(m[0] if m else np.nan)
        pos = x + (i - 0.5) * (w + 0.02)
        ax.bar(pos, vals, width=w, color=color, label=label, edgecolor="white", linewidth=1)
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(p, v * (1.08 if log else 1) + (0 if log else 0.01), fmt(v), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [f"batch_interval={b}\nstep_size={s}" for b, s in combos], fontsize=8.5)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1)
    style(ax, ylabel)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.01, "SF LODES test instance seed 42 - 1000 requests, 50 vehicles, capacity 2, max_cardinality=3, "
             "keep_active=False, solver seed 42, free-flow road times"
             + ("\nbatch_interval=900/step_size=450 ran locally (MacBook Air), all others on the LRZ cluster -solver times across machines are not directly comparable" if log else ""),
             fontsize=7.5, color=MUTED)  # 2026-11-11: machine note only on the runtime chart
    fig.tight_layout(rect=(0, 0.06 if log else 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=200)
    plt.close(fig)


def instances_chart(runs):
    sel = sorted([r for r in runs if (r["batch_interval"], r["step_size"]) == (900, 450) and r["time_rule_filter"]],
                 key=lambda r: r["instance_seed"])
    names = {1: "seed 1 (SRL train)", 2: "seed 2 (SRL val)", 42: "seed 42 (test)"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = [r["service_rate"] for r in sel]
    bars = ax.bar(range(len(sel)), vals, width=0.55, color=BLUE, edgecolor="white")
    for b, r in zip(bars, sel):
        ax.text(b.get_x() + b.get_width() / 2, r["service_rate"] + 0.01, f"{r['service_rate']:.1%}\n({r['served']}/{r['requests']})",
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(sel)), [names.get(r["instance_seed"], f"seed {r['instance_seed']}") for r in sel])
    ax.set_ylim(0, 1)
    style(ax, "Service rate")
    ax.set_title("RHO on three SF instances - batch_interval=900/step_size=450, time-rule filter on", fontsize=10.5, color=INK, loc="left")
    fig.text(0.01, 0.01, "max_cardinality=3, keep_active=False, solver seed 42, 50 vehicles, capacity 2", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"sf_rho_instances.{ext}", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    test = [r for r in runs if r["instance_seed"] == 42]
    grouped(test, "service_rate", "Service rate", "RHO service rate on SF (test instance seed 42)",
            "sf_rho_service_rate", lambda v: f"{v:.1%}")
    grouped(test, "solver_time_s", "Solver time [s] (log scale)", "RHO solver time on SF (test instance seed 42)",
            "sf_rho_solver_time", lambda v: f"{v:,.0f}", log=True)
    instances_chart(runs)

    runs.sort(key=lambda r: (r["instance_seed"] != 42, r["instance_seed"], not r["time_rule_filter"], r["batch_interval"], r["step_size"]))
    fields = list(runs[0].keys())
    with open(OUT / "sf_rho_results_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(runs)
    lines = [r"\begin{tabular}{rrrlrrrrr}", r"\toprule",
             r"Instance & batch\_interval & step\_size & Time-rule filter & Served & Wait [s] & Detour [s] & VMT/PMT & Solver time [s] \\", r"\midrule"]
    for r in runs:
        lines.append(f"{r['instance_seed']} & {r['batch_interval']} & {r['step_size']} & {'on' if r['time_rule_filter'] else 'off'} & "
                     f"{r['service_rate']*100:.1f}\\% & {r['wait_s']:.1f} & {r['detour_s']:.1f} & {r['vmt_over_pmt']:.3f} & {r['solver_time_s']:,.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "sf_rho_results_table.tex").write_text("\n".join(lines) + "\n")
    for r in runs:
        print(f"seed{r['instance_seed']:>2} bi={r['batch_interval']:>4} ss={r['step_size']:>4} filter={'on ' if r['time_rule_filter'] else 'off'} "
              f"mc={r['max_cardinality']} keep_active={r['keep_active']} served={r['service_rate']:.1%} wait={r['wait_s']:.1f} "
              f"detour={r['detour_s']:.1f} vmt/pmt={r['vmt_over_pmt']:.3f} time={r['solver_time_s']:.0f}s {r['where']}")


if __name__ == "__main__":
    main()
