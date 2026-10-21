"""
2026-10-21: RHO baseline (offline, no pruner) with the vehicle capacity fixed to C requests ON BOARD (unit demand copy of the manifest,
see rho_capacity_sweep.make_copy) and the TOTAL number of requests per vehicle limited to L (new --max_requests_per_vehicle, Config.MAX_REQUESTS_PER_VEHICLE).
Question: how does the service rate change with L when the simultaneous capacity is fixed?
Per run: served requests, max on board, max served by one vehicle (must be <= L), seconds (30 min limit per run). RHO is not bit-deterministic (+-1 request).

Run: ./venv/bin/python3 scripts/analysis/rho_total_limit_sweep.py [--instances lr109 lr208] [--C 2] [--card 2] [--bi 40] [--ss 10] [--L 2 4 6 8 10 12 0]
(L=0 means "no limit".) Output: outputs/rho_total_limit_sweep_C<C>_c<card>_bi<bi>_ss<ss>.csv
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["lr109", "lr208"])
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--card", default="2")
    ap.add_argument("--bi", default="40")
    ap.add_argument("--ss", default="10")
    ap.add_argument("--L", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 0])
    a = ap.parse_args()
    rows = []
    for stem in a.instances:
        n_req = len(json.load(open(REPO / "solutions/li_lim/manifests" / f"{stem}.json"))["requests"])
        f = R.make_copy(stem, a.C, f"_c{a.card}_bi{a.bi}_ss{a.ss}_tl")
        for L in a.L:
            out = SCRATCH / f"out_tl_{stem}_C{a.C}_L{L}_c{a.card}_bi{a.bi}_ss{a.ss}"
            cmd = [sys.executable, "rtv_solver/main.py", "--mode", "offline", "--input_file", str(f), "--input_dir", str(f.parent) + "/",
                   "--imitation_solution_file", str(f), "--use_request_graph_pruner", "False", "--max_cardinality", a.card,
                   "--batch_interval", a.bi, "--step_size", a.ss, "--output_dir", str(out)]
            if L > 0:
                cmd += ["--max_requests_per_vehicle", str(L)]
            t0 = time.time()
            try:
                subprocess.run(cmd, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=TIMEOUT_S, env={**os.environ, "PYTHONPATH": str(REPO)})
                res, status = R.summarize(out), "ok"
            except subprocess.TimeoutExpired:
                res, status = None, "timeout"
            served, max_load, max_per_veh = res if res else (None, None, None)
            rows.append(dict(instance=stem, requests=n_req, C=a.C, L=(L if L > 0 else "none"), status=status, served=served,
                             service_rate=(100 * served / n_req if served is not None else None), max_on_board=max_load,
                             max_served_by_one_vehicle=max_per_veh, seconds=round(time.time() - t0, 1)))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(REPO / f"outputs/rho_total_limit_sweep_C{a.C}_c{a.card}_bi{a.bi}_ss{a.ss}.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
