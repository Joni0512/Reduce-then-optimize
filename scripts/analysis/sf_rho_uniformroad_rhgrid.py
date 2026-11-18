"""
2026-11-15: rolling-horizon window grid on SF seed 42 with demand-independent uniform road-node vehicle starts
(user request "andere time windows checken", grid "Verhaeltnis 2:1"). batch_interval/step_size = 300/150, 600/300,
900/450 (job 5734857), 1800/900, 3600/1800 (job 5736129, submit_sf_lodes_rho_uniformroad_rhgrid.sbatch).
All: 150 vehicles, uniform road-node starts (start seeds 42-46), time-rule filter off, max_cardinality=2, capacity 2,
solver seed 42, no depot return; LRZ cluster serial_std. Missing runs (still running / timeout) are skipped and listed.
Output: figures_export/SF/sf_rho_uniformroad_rhgrid_{service,runtime}.{png,pdf} + _table.csv
2026-11-16: added 3600/1800 with --rtv_timeout 21600 (job 5737975, submit_sf_lodes_rho_uniformroad_bi3600_rtv21600.sbatch,
0 RTV timeouts in main.log). The old 3600/1800 bar (RTV_TIMEOUT 1800 s hit in 2-3 steps per run) is kept and labelled
"RTV timeout 1800 s" so the confound stays visible.
"""
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_export" / "SF" / "sf_rho_uniformroad_rhgrid"
# 2026-11-16: (batch_interval, step_size, directory suffix, extra x-label line)
CONFIGS = [(300, 150, "", ""), (600, 300, "", ""), (900, 450, "", ""), (1800, 900, "", ""),
           (3600, 1800, "", "RTV limit 1800 s (hit)"), (3600, 1800, "_rtv21600", "RTV limit 21600 s")]
SEEDS = (42, 43, 44, 45, 46)
INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
FOOT = ("SF seed 42, 1000 requests 06:00-08:00, 150 vehicles starting uniformly on OSM road nodes of SF/Alameda/San Mateo/"
        "Marin (start seeds 42-46), time-rule filter off, max_cardinality=2, capacity 2, solver seed 42, no depot return; "
        "LRZ serial_std. rtv_timeout 1800 s (main.py default) unless labelled.\nBar = mean over start seeds, whisker = min-max, dots = single runs. Sources: outputs/new_tests/"
        "sf_lodes_nodepot/bi*_ss*_mc2_v150_notimerule_noreturn_uniformroad[_rtv21600]_s*/run_*/*/final/results.json")


def main():
    rows, missing = [], []
    for bi, ss, suf, _ in CONFIGS:
        for s in SEEDS:
            d = f"outputs/new_tests/sf_lodes_nodepot/bi{bi}_ss{ss}_mc2_v150_notimerule_noreturn_uniformroad{suf}_s{s}"
            f = glob.glob(str(ROOT / d / "run_*/*/final/results.json"))
            if not f:
                missing.append((bi, ss, suf, s))
                continue
            st = json.loads(Path(f[0]).read_text())["stats"]
            rows.append({"batch_interval": bi, "step_size": ss, "rtv_timeout_s": 21600 if suf else 1800, "start_seed": s, "served": int(st["serviced"]),
                         "service_rate": st["serviced"] / st["total_requests"], "wait_s": round(st["average_wait_time"], 1),
                         "detour_s": round(st["average_detour"], 1), "vehicle_travel_time_h": round(st["vmt"] / 3600, 1),
                         "solver_time_s": round(st["total_time"], 1), "source": str(Path(f[0]).relative_to(ROOT))})
    key_of = lambda r: (r["batch_interval"], r["step_size"], r["rtv_timeout_s"])
    cfgs = [c for c in CONFIGS if any(key_of(r) == (c[0], c[1], 21600 if c[2] else 1800) for r in rows)]
    labels = [f"batch_interval={bi}\nstep_size={ss}" + (f"\n{x}" if x else "") for bi, ss, _, x in cfgs]

    def per(key):
        return [np.array([r[key] for r in rows if key_of(r) == (c[0], c[1], 21600 if c[2] else 1800)]) for c in cfgs]

    for name, panels in (("service", [("service_rate", "Service rate", "{:.1%}"), ("wait_s", "Average wait time [s]", "{:.0f}"),
                                      ("vehicle_travel_time_h", "Vehicle travel time [h]", "{:.0f}")]),
                         ("runtime", [("solver_time_s", "Solver wall-clock time [s]", "{:.0f}")])):
        fig, axes = plt.subplots(1, len(panels), figsize=(max(6.2 * len(panels) + 1, 14.5), 5.2), squeeze=False)  # 2026-11-16: wider (6 configs, footnote)
        for ax, (key, ylab, fmt) in zip(axes[0], panels):
            vals = per(key)
            x = np.arange(len(cfgs))
            mean = np.array([v.mean() for v in vals])
            ax.bar(x, mean, width=0.6, color=BLUE, edgecolor="white", linewidth=2)
            ax.vlines(x, [v.min() for v in vals], [v.max() for v in vals], color=INK, linewidth=1)
            for xi, v in zip(x, vals):
                ax.scatter(np.full(len(v), xi), v, s=10, color=INK, zorder=3)
                ax.text(xi, v.max(), " " + fmt.format(v.mean()) + (f"\n n={len(v)}" if len(v) < len(SEEDS) else ""),
                        ha="center", va="bottom", fontsize=8, color=INK)
            ax.set_xticks(x, labels, fontsize=7.5, rotation=30, ha="right")  # 2026-11-16: labels overlapped
            ax.set_ylabel(ylab, color=MUTED)
            if key == "service_rate":
                ax.set_ylim(0.8 if mean.min() > 0.85 else 0, 1.0)
            ax.grid(axis="y", color=GRID)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.suptitle("RHO on SF seed 42 with uniform vehicle starts: rolling-horizon window", fontsize=11, x=0.01, ha="left", color=INK)
        fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.1, 1, 0.95))
        for ext in ("png", "pdf"):
            fig.savefig(f"{OUT}_{name}.{ext}", dpi=200)
        plt.close(fig)

    with open(f"{OUT}_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for c, sr, wt, vt, rt in zip(cfgs, per("service_rate"), per("wait_s"), per("vehicle_travel_time_h"), per("solver_time_s")):
        print(f"bi={c[0]} ss={c[1]}{c[2]} n={len(sr)}: service {sr.mean():.2%} ({sr.min():.1%}-{sr.max():.1%}), wait {wt.mean():.1f} s, "
              f"vtt {vt.mean():.1f} h, solver {rt.mean():.0f} s ({rt.min():.0f}-{rt.max():.0f})")
    print("missing:", missing)


if __name__ == "__main__":
    main()
