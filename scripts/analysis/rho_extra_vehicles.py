"""
2026-11-11: RHO (offline, no pruner) with MORE VEHICLES than the Li&Lim reference solution: 100 vehicles in total over the 14 test instances (89 original + 11 extra),
extra vehicles only for the class-2 instances (they have only 2-3 vehicles for ~50 requests), distributed round-robin in the order lc207 lc208 lr208 lr210 lr211 lrc207 lrc208
(first 4 get +2, last 3 get +1 -> 30 instead of 19 class-2 vehicles). User decision 2026-11-11 ("insgesamt 100 autos", 14 instances, only class 2, round-robin).

How: copy of the unit-demand manifest (rho_capacity_sweep.make_copy, C on board) plus N extra driver_runs = copy of the state of vehicle 0 (depot position, start/end time, capacity)
with a new run_id and an EMPTY manifest (total_locations 0). The solver is not changed. Variants per instance: base (original vehicles) and extra (original + extra), interleaved, REPS repetitions.

Setting (same as the 75.5 % baseline): batch_interval=40, step_size=10, max_cardinality 2, C=2 on board, no total limit per vehicle, no pruner.
Per-instance limit RHO_TIMEOUT_S (rho_capacity_sweep.TIMEOUT_S, default 1800; use RHO_TIMEOUT_S=7200 for the 2 h rule).

Run: ./venv/bin/python3 scripts/analysis/rho_extra_vehicles.py [--reps 2] [--instances lr208] [--C 2] [--card 2] [--bi 40] [--ss 10]
Output: outputs/rho_extra_vehicles_C<C>_c<card>_bi<bi>_ss<ss>.csv
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rho_capacity_sweep as R  # noqa: E402

REPO, SCRATCH, TIMEOUT_S = R.REPO, R.SCRATCH, R.TIMEOUT_S
INSTANCES = ["lc108", "lc109", "lr109", "lr111", "lr112", "lrc107", "lrc108", "lc207", "lc208", "lr208", "lr210", "lr211", "lrc207", "lrc208"]
CLASS2 = ["lc207", "lc208", "lr208", "lr210", "lr211", "lrc207", "lrc208"]
EXTRA_TOTAL = 11


def extra_per_instance(instances, extra_total=EXTRA_TOTAL):
    """round-robin over the class-2 instances (in CLASS2 order) that are part of the run"""
    targets = [s for s in CLASS2 if s in instances]
    out = {s: 0 for s in instances}
    for i in range(extra_total):
        out[targets[i % len(targets)]] += 1
    return out


def make_copy_extra(stem, cap, n_extra, tag):
    p = R.make_copy(stem, cap, tag)  # unit-demand copy with am_capacity=cap, written to scratch
    if n_extra == 0:
        return p
    d = json.load(open(p))
    runs = d["driver_runs"]
    next_id = max(r["state"]["run_id"] for r in runs) + 1
    for k in range(n_extra):
        st = copy.deepcopy(runs[0]["state"])
        st["run_id"] = next_id + k
        st["locations_already_serviced"] = 0
        st["total_locations"] = 0
        runs.append({"state": st, "manifest": []})
    out = p.with_name(f"{stem}_extra{n_extra}.json")
    json.dump(d, open(out, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--card", default="2")
    ap.add_argument("--bi", default="40")
    ap.add_argument("--ss", default="10")
    ap.add_argument("--extra_total", type=int, default=EXTRA_TOTAL)
    a = ap.parse_args()
    extra = extra_per_instance(a.instances, a.extra_total)
    print("extra vehicles per instance:", extra, flush=True)
    out_csv = REPO / f"outputs/rho_extra_vehicles_C{a.C}_c{a.card}_bi{a.bi}_ss{a.ss}.csv"
    rows = []
    for rep in range(a.reps):
        for stem in a.instances:
            n_req = len(json.load(open(REPO / "solutions/li_lim/manifests" / f"{stem}.json"))["requests"])
            n_veh0 = len(json.load(open(REPO / "solutions/li_lim/manifests" / f"{stem}.json"))["driver_runs"])
            for variant in ("base", "extra"):
                n_ex = extra[stem] if variant == "extra" else 0
                f = make_copy_extra(stem, a.C, n_ex, f"_xv_C{a.C}")
                out = SCRATCH / f"out_xv_{stem}_{variant}_C{a.C}_c{a.card}_bi{a.bi}_ss{a.ss}_rep{rep}"
                cmd = [sys.executable, "rtv_solver/main.py", "--mode", "offline", "--input_file", str(f), "--input_dir", str(f.parent) + "/",
                       "--imitation_solution_file", str(f), "--use_request_graph_pruner", "False", "--max_cardinality", a.card,
                       "--batch_interval", a.bi, "--step_size", a.ss, "--output_dir", str(out)]
                t0 = time.monotonic()
                try:
                    subprocess.run(cmd, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=TIMEOUT_S, env={**os.environ, "PYTHONPATH": str(REPO)})
                    res, status = R.summarize(out), "ok"
                except subprocess.TimeoutExpired:
                    res, status = None, "timeout"
                dt = time.monotonic() - t0
                served, max_load, max_per_veh = res if res else (None, None, None)
                rows.append(dict(instance=stem, variant=variant, rep=rep, vehicles=n_veh0 + n_ex, requests=n_req, status=status, served=served,
                                 max_on_board=max_load, max_served_by_one_vehicle=max_per_veh, seconds=round(dt, 1)))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
