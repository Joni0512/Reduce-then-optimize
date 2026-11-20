"""
2026-11-19: RHO (offline, no pruner) with the no-learning vehicle cost-ratio filter (Config.USE_VEHICLE_COST_RATIO_FILTER, TripHandler._filter_vehicle_edges_by_cost_ratio) vs. own baseline.
Per request keep only vehicles whose single-trip added cost <= rho x the added cost of the best vehicle (best always kept, active requests untouched), before the shared trips are built.
Variants per instance (interleaved, REPS repetitions): base | vcr<rho> (e.g. vcr3 = rho 3.0).

Setting (same as the 75.5 % baseline): batch_interval=40, step_size=10, max_cardinality 2, C=2 on board (unit-demand copy), no total limit, no pruner, 14 test instances.
Per run: served requests, subprocess wall-clock (time.monotonic), sum over windows of the solver's own "feasible trip costs generated: X s" (RTV generation incl. filter), and the filter's
kept/total single-trip costs summed over windows (parsed from main.log). Limit per instance RHO_TIMEOUT_S (default 1800 s).

Run: RHO_TIMEOUT_S=7200 ./venv/bin/python3 scripts/analysis/rho_vehicle_cost_ratio_run.py [--rhos 3] [--reps 2] [--instances lr109] [--bi 40 --ss 10 --C 2 --card 2]
Output: outputs/rho_vehicle_cost_ratio_C<C>_c<card>_bi<bi>_ss<ss>.csv
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rho_capacity_sweep as R  # noqa: E402

INSTANCES = ["lc108", "lc109", "lr109", "lr111", "lr112", "lrc107", "lrc108", "lc207", "lc208", "lr208", "lr210", "lr211", "lrc207", "lrc208"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--rhos", nargs="+", type=float, default=[3.0])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--card", default="2")
    ap.add_argument("--bi", default="40")
    ap.add_argument("--ss", default="10")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    variants = ["base"] + [f"vcr{r:g}" for r in a.rhos]
    out_csv = R.REPO / f"outputs/rho_vehicle_cost_ratio_C{a.C}_c{a.card}_bi{a.bi}_ss{a.ss}{a.tag}.csv"
    rows = []
    for rep in range(a.reps):
        for stem in a.instances:
            f = R.make_copy(stem, a.C, "_vcr")
            n_req = len(json.load(open(R.REPO / "solutions/li_lim/manifests" / f"{stem}.json"))["requests"])
            for v in variants:
                out = R.SCRATCH / f"out_vcr_{stem}_{v}_C{a.C}_c{a.card}_bi{a.bi}_ss{a.ss}_rep{rep}"
                cmd = [sys.executable, "rtv_solver/main.py", "--mode", "offline", "--input_file", str(f), "--input_dir", str(f.parent) + "/",
                       "--imitation_solution_file", str(f), "--use_request_graph_pruner", "False", "--max_cardinality", a.card,
                       "--batch_interval", a.bi, "--step_size", a.ss, "--output_dir", str(out)]
                if v != "base":
                    cmd += ["--use_vehicle_cost_ratio_filter", "True", "--vehicle_cost_ratio", str(float(v[3:]))]
                t0 = time.monotonic()
                try:
                    subprocess.run(cmd, cwd=R.REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=R.TIMEOUT_S, env={**os.environ, "PYTHONPATH": str(R.REPO)})
                    status = "ok"
                except subprocess.TimeoutExpired:
                    status = "timeout"
                wall = time.monotonic() - t0
                res = R.summarize(out) if status == "ok" else None
                rtv = kept = total = None
                logs = glob.glob(str(out / "run_offline_*" / "*" / "main.log"))
                if logs and status == "ok":
                    txt = open(logs[0], errors="ignore").read()
                    rtv = sum(float(x) for x in re.findall(r"feasible trip costs generated: ([0-9.]+)s", txt))
                    m = re.findall(r"Vehicle cost-ratio filter \(rho=[0-9.]+\): kept (\d+)/(\d+) single-trip costs", txt)
                    if m:
                        kept, total = sum(int(x) for x, _ in m), sum(int(y) for _, y in m)
                rows.append(dict(instance=stem, variant=v, rep=rep, requests=n_req, status=status, served=(res[0] if res else None), wall_s=round(wall, 2),
                                 rtv_s=(round(rtv, 2) if rtv is not None else None), single_costs_kept=kept, single_costs_total=total))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
