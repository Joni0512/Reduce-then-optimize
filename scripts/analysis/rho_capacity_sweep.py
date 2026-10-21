"""
2026-10-21: RHO baseline (offline, no pruner) as a function of the simultaneous-load limit C (vehicle capacity in REQUESTS).

How: for every C a COPY of the Li&Lim manifest is written outside the repo (scratch dir) with am=1 for every request and stop and
am_capacity=C for every vehicle, so that capacity = "at most C requests on board at the same time". The solver is not changed; the originals in
solutions/li_lim/manifests are untouched. Reference run "orig" = the original file (capacity 200/1000, real demands).
Checked before (lr109, c2, bi200/ss100): C=2 copy gives the same result as the original and the constraint holds (max 2 on board).

Measured per run: served requests (dropoffs in the final manifest), max requests on board at the same time, max requests served by one vehicle,
wall-clock seconds. Per-run limit 30 min (CLAUDE.md rule); a run that exceeds it is reported as timeout.
Note: RHO is not bit-deterministic (lr109 c2 gave 38 and 37 served in two identical runs), so one run per cell is +-1 request noise.

Run: ./venv/bin/python3 scripts/analysis/rho_capacity_sweep.py [--instances lr109 lr208] [--card 3] [--bi 400] [--ss 80]
Output: outputs/rho_capacity_sweep_c<card>_bi<bi>_ss<ss>.csv, printed table.
(2026-10-21: file name now carries card/bi/ss; the first c3 bi40/ss10 run started before this change and writes outputs/rho_capacity_sweep.csv.)
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path("/private/tmp/claude-501/-Users-joni-Desktop-Masterarbeit-Reduce-then-optimize/a8e50c2f-fe60-45b6-b4de-3217ebb7c3f1/scratchpad/cap_sweep")
CAPS = [2, 3, 4, 6, 7, 10, 12, 20, 160]
TIMEOUT_S = 1800


def make_copy(stem, cap, tag=""):
    d = json.load(open(REPO / "solutions/li_lim/manifests" / f"{stem}.json"))
    for r in d["requests"]:
        r["am"] = 1
    for dr in d["driver_runs"]:
        dr["state"]["am_capacity"] = cap
        for s in dr["manifest"]:
            s["am"] = 1
    out_dir = SCRATCH / f"manifests_C{cap}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}.json"
    json.dump(d, open(p, "w"))
    return p


def summarize(out_dir):
    f = glob.glob(str(out_dir / "run_offline_*" / "*" / "final" / "result_driver_runs.json"))
    if not f:
        return None
    d = json.load(open(f[0]))
    runs = d["driver_runs"] if "driver_runs" in d else d
    served, per_vehicle, max_load = 0, [], 0
    for dr in runs:
        load = n = 0
        for s in dr["manifest"]:
            if s["action"] == "pickup":
                load += 1
            elif s["action"] == "dropoff":
                load -= 1
                n += 1
            max_load = max(max_load, load)
        per_vehicle.append(n)
        served += n
    return served, max_load, max(per_vehicle) if per_vehicle else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["lr109", "lr208"])
    ap.add_argument("--card", default="3")
    ap.add_argument("--bi", default="400")
    ap.add_argument("--ss", default="80")
    a = ap.parse_args()

    rows = []
    for stem in a.instances:
        n_req = len(json.load(open(REPO / "solutions/li_lim/manifests" / f"{stem}.json"))["requests"])
        for cap in ["orig"] + CAPS:
            f = REPO / "solutions/li_lim/manifests" / f"{stem}.json" if cap == "orig" else make_copy(stem, cap, f"_c{a.card}_bi{a.bi}_ss{a.ss}")
            out = SCRATCH / f"out_{stem}_C{cap}_c{a.card}_bi{a.bi}_ss{a.ss}"
            cmd = [sys.executable, "rtv_solver/main.py", "--mode", "offline", "--input_file", str(f), "--input_dir", str(f.parent) + "/",
                   "--imitation_solution_file", str(f), "--use_request_graph_pruner", "False", "--max_cardinality", a.card,
                   "--batch_interval", a.bi, "--step_size", a.ss, "--output_dir", str(out)]
            t0 = time.time()
            try:
                subprocess.run(cmd, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=TIMEOUT_S, env={**__import__("os").environ, "PYTHONPATH": str(REPO)})
                res, status = summarize(out), "ok"
            except subprocess.TimeoutExpired:
                res, status = None, "timeout"
            dt = time.time() - t0
            served, max_load, max_per_veh = res if res else (None, None, None)
            rows.append(dict(instance=stem, requests=n_req, C=cap, status=status, served=served, service_rate=(100 * served / n_req if served is not None else None),
                             max_on_board=max_load, max_served_by_one_vehicle=max_per_veh, seconds=round(dt, 1)))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(REPO / f"outputs/rho_capacity_sweep_c{a.card}_bi{a.bi}_ss{a.ss}.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
