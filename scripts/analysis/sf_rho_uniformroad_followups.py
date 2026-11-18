"""
2026-11-17: follow-ups of the SF uniform-start window grid (scripts/analysis/sf_rho_uniformroad_rhgrid.py), three
experiments, all RHO offline, 150 vehicles, time-rule filter off, max_cardinality=2, capacity 2, solver seed 42, no depot
return, rtv_timeout 1800 s, start seeds 42-46, LRZ serial_std:
  (1) step_size sweep at batch_interval=1800 (user: "bei 1800 interval verschiedene step sizes 200, 300, 450 und 600"),
      submit_sf_lodes_rho_uniformroad_bi1800_ssgrid.sbatch; step_size=900 = job 5736129.
  (2) instance seeds 1 and 2 (user: "mehr seeds"), submit_sf_lodes_rho_uniformroad_instseeds.sbatch (job 5738603);
      instance seed 42 = jobs 5734857 / 5736129.
  (3) uniform AREA starts vs uniform road-NODE starts on instance seed 42 (user idea "Flaeche"),
      submit_sf_lodes_rho_uniformarea.sbatch (job 5738604).
Per run also read from main.log: number of RTV timeouts and the maximum number of requests assigned in one step
(cap = 150 vehicles x max_cardinality 2 = 300).
Output: figures_export/SF/sf_rho_uniformroad_{ssgrid,instseeds,area}_{service,runtime}.{png,pdf} + _table.csv
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
BASE = ROOT / "outputs/new_tests/sf_lodes_nodepot"
OUTDIR = ROOT / "figures_export" / "SF"
SEEDS = (42, 43, 44, 45, 46)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#d97a2b"
COMMON = ("150 vehicles, time-rule filter off, max_cardinality=2, capacity 2, solver seed 42, no depot return, "
          "rtv_timeout 1800 s, LRZ serial_std. Bar = mean over start seeds 42-46, whisker = min-max, dots = single runs.")


def load(dirname):
    """Returns one row per start seed for outputs/new_tests/sf_lodes_nodepot/<dirname with {s}>."""
    rows = []
    for s in SEEDS:
        d = BASE / dirname.format(s=s)
        f = glob.glob(str(d / "run_*/*/final/results.json"))
        if not f:
            continue
        st = json.loads(Path(f[0]).read_text())["stats"]
        log = glob.glob(str(d / "run_*/*/main.log"))
        txt = Path(log[0]).read_text() if log else ""
        assigned = [int(m) for m in re.findall(r"unassigned / assigned: \d+ / \d+ / (\d+)", txt)]
        rows.append({"dir": dirname.format(s=s), "start_seed": s, "served": int(st["serviced"]),
                     "service_rate": st["serviced"] / st["total_requests"], "wait_s": round(st["average_wait_time"], 1),
                     "vehicle_travel_time_h": round(st["vmt"] / 3600, 1), "solver_time_s": round(st["total_time"], 1),
                     "rtv_timeouts": txt.count("RTV generation timed out"), "steps": len(assigned),
                     "max_assigned_per_step": max(assigned) if assigned else None,
                     "source": str(Path(f[0]).relative_to(ROOT))})
    return rows


def plot(name, groups, title, foot, series=None):
    """groups: list of (x label, {series name: rows}). One bar per series per group."""
    series = series or [None]
    colors = {None: BLUE, series[0]: BLUE, series[-1]: ORANGE} if len(series) > 1 else {None: BLUE}
    for kind, panels in (("service", [("service_rate", "Service rate", "{:.1%}"), ("wait_s", "Average wait time [s]", "{:.0f}"),
                                      ("vehicle_travel_time_h", "Vehicle travel time [h]", "{:.0f}")]),
                         ("runtime", [("solver_time_s", "Solver wall-clock time [s]", "{:.0f}")])):
        fig, axes = plt.subplots(1, len(panels), figsize=(max(5.6 * len(panels) + 1, 13), 5.0), squeeze=False)
        w = 0.7 / len(series)
        for ax, (key, ylab, fmt) in zip(axes[0], panels):
            allv = []
            for si, sname in enumerate(series):
                for gi, (_, by) in enumerate(groups):
                    v = np.array([r[key] for r in by.get(sname, [])])
                    if not len(v):
                        continue
                    allv.append(v)
                    x = gi + (si - (len(series) - 1) / 2) * w
                    ax.bar(x, v.mean(), width=w * 0.92, color=colors[sname], edgecolor="white", linewidth=1.5,
                           label=sname if gi == 0 else None)
                    ax.vlines(x, v.min(), v.max(), color=INK, linewidth=1)
                    ax.scatter(np.full(len(v), x), v, s=9, color=INK, zorder=3)
                    ax.text(x, v.max(), fmt.format(v.mean()) + (f"\nn={len(v)}" if len(v) < len(SEEDS) else ""),
                            ha="center", va="bottom", fontsize=7.5, color=INK)
            rot = 30 if len(groups) > 4 else 0  # 2026-11-17: >= 5 long labels overlap
            ax.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=8, rotation=rot, ha="right" if rot else "center")
            ax.set_ylabel(ylab, color=MUTED)
            if key == "service_rate":
                lo = min(v.min() for v in allv)
                ax.set_ylim(0.6 if lo > 0.65 else 0, 1.0)
            ax.grid(axis="y", color=GRID)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        if len(series) > 1:
            axes[0][0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)  # 2026-11-17: above the axes, was covering bars
        fig.suptitle(title, fontsize=11, x=0.01, ha="left", color=INK)
        fig.text(0.01, 0.01, foot + "\n" + COMMON, fontsize=6.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.1, 1, 0.92))
        for ext in ("png", "pdf"):
            fig.savefig(OUTDIR / f"{name}_{kind}.{ext}", dpi=200)
        plt.close(fig)


def table(name, tagged):
    rows = [dict(experiment_group=g, **r) for g, rs in tagged for r in rs]
    with open(OUTDIR / f"{name}_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for g, rs in tagged:
        sr = np.array([r["service_rate"] for r in rs]); rt = np.array([r["solver_time_s"] for r in rs])
        print(f"{name} | {g:32s} n={len(rs)} service {sr.mean():.2%} ({sr.min():.1%}-{sr.max():.1%}) "
              f"wait {np.mean([r['wait_s'] for r in rs]):.0f} s vtt {np.mean([r['vehicle_travel_time_h'] for r in rs]):.0f} h "
              f"solver {rt.mean():.0f} s ({rt.min():.0f}-{rt.max():.0f}) steps {rs[0]['steps']} "
              f"max assigned/step {max(r['max_assigned_per_step'] for r in rs)} rtv timeouts {sum(r['rtv_timeouts'] for r in rs)}")


def main():
    # (1) step_size sweep at batch_interval=1800
    ss = [(s, load(f"bi1800_ss{s}_mc2_v150_notimerule_noreturn_uniformroad_s{{s}}")) for s in (200, 300, 450, 600, 900)]
    plot("sf_rho_uniformroad_ssgrid", [(f"batch_interval=1800\nstep_size={s}", {None: r}) for s, r in ss],
         "RHO on SF seed 42 with uniform road-node starts: step_size at batch_interval=1800",
         "SF instance seed 42, 1000 requests 06:00-08:00. Sources: outputs/new_tests/sf_lodes_nodepot/"
         "bi1800_ss*_mc2_v150_notimerule_noreturn_uniformroad_s*/run_*/*/final/results.json")
    table("sf_rho_uniformroad_ssgrid", [(f"bi1800_ss{s}", r) for s, r in ss])

    # (2) instance seeds 42 / 1 / 2 at 900/450 and 1800/900
    inst = {}
    for bi, ss_ in ((900, 450), (1800, 900)):
        inst[(bi, ss_, 42)] = load(f"bi{bi}_ss{ss_}_mc2_v150_notimerule_noreturn_uniformroad_s{{s}}")
        for i in (1, 2):
            inst[(bi, ss_, i)] = load(f"inst{i}_bi{bi}_ss{ss_}_mc2_v150_notimerule_noreturn_uniformroad_s{{s}}")
    names = {42: "instance seed 42", 1: "instance seed 1", 2: "instance seed 2"}
    plot("sf_rho_uniformroad_instseeds",
         [(f"{names[i]}\nbatch_interval={bi}\nstep_size={ss_}", {None: r}) for (bi, ss_, i), r in inst.items()],
         "RHO on SF instance seeds 42 / 1 / 2 with uniform road-node starts",
         "SF LODES instances seed 42, 1, 2 (same generator, 1000 requests 06:00-08:00 each); start seed s gives the same "
         "150 start nodes in every instance. Sources: outputs/new_tests/sf_lodes_nodepot/[inst{1,2}_]bi*_ss*_mc2_v150_"
         "notimerule_noreturn_uniformroad_s*/run_*/*/final/results.json")
    table("sf_rho_uniformroad_instseeds", [(f"inst{i}_bi{bi}_ss{ss_}", r) for (bi, ss_, i), r in inst.items()])

    # (3) uniform road-node vs uniform area starts
    area = []
    for bi, ss_ in ((900, 450), (1800, 900)):
        area.append((f"batch_interval={bi}\nstep_size={ss_}", {
            "uniform over road nodes": load(f"bi{bi}_ss{ss_}_mc2_v150_notimerule_noreturn_uniformroad_s{{s}}"),
            "uniform over area (1 km cells)": load(f"bi{bi}_ss{ss_}_mc2_v150_notimerule_noreturn_uniformarea_s{{s}}")}))
    plot("sf_rho_uniformroad_vs_area", area, "RHO on SF seed 42: vehicle starts uniform over road nodes vs over area",
         "SF instance seed 42. Area starts: scripts/sf/make_uniform_area_starts.py (1 km cells with >= 1 road node, "
         "2100 cells, distinct cells then one node per cell). Sources: outputs/new_tests/sf_lodes_nodepot/bi*_ss*_mc2_"
         "v150_notimerule_noreturn_uniform{road,area}_s*/run_*/*/final/results.json",
         series=["uniform over road nodes", "uniform over area (1 km cells)"])
    table("sf_rho_uniformroad_vs_area", [(f"{g.split(chr(10))[0]}|{k}", r) for g, d in area for k, r in d.items()])


if __name__ == "__main__":
    main()
