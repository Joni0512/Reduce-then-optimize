"""
2026-11-18: run RHO (offline, no pruner) with --log_candidate_insertions True on the 14 Li&Lim test instances and collect the "CandidateInsertions" records
(one per ILP solve = per window; every feasible single-request insertion request x vehicle, chosen or not) into ONE CSV.
Purpose: analysis of an RHO heuristic ("distance request -> vehicle" and "how much longer does the trip get when the request is added"), see analyze_candidate_insertions.py.

Setting (same as the 75.5 % baseline): batch_interval=40, step_size=10, max_cardinality 2, C=2 on board (unit-demand manifest copy), no total limit, no pruner, 1 run per instance.
Window index = order of the CandidateInsertions records in assignment_data.jsonl (one per ILP solve). Limit per instance RHO_TIMEOUT_S (default 1800 s, rho_capacity_sweep).

Run: RHO_TIMEOUT_S=7200 ./venv/bin/python3 scripts/analysis/rho_candidate_insertions_run.py [--bi 40 --ss 10 --C 2 --card 2]
Output: outputs/candidate_insertions_C<C>_c<card>_bi<bi>_ss<ss>.csv (+ outputs/candidate_insertions_..._runs.csv with served per instance)
"""
import argparse
import glob
import json
import os
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
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--card", default="2")
    ap.add_argument("--bi", default="40")
    ap.add_argument("--ss", default="10")
    a = ap.parse_args()
    tag = f"C{a.C}_c{a.card}_bi{a.bi}_ss{a.ss}"
    frames, runs = [], []
    for stem in a.instances:
        f = R.make_copy(stem, a.C, "_ci")
        out = R.SCRATCH / f"out_ci_{stem}_{tag}"
        cmd = [sys.executable, "rtv_solver/main.py", "--mode", "offline", "--input_file", str(f), "--input_dir", str(f.parent) + "/",
               "--imitation_solution_file", str(f), "--use_request_graph_pruner", "False", "--max_cardinality", a.card,
               "--batch_interval", a.bi, "--step_size", a.ss, "--log_candidate_insertions", "True", "--output_dir", str(out)]
        t0 = time.monotonic()
        try:
            subprocess.run(cmd, cwd=R.REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=R.TIMEOUT_S, env={**os.environ, "PYTHONPATH": str(R.REPO)})
            status = "ok"
        except subprocess.TimeoutExpired:
            status = "timeout"
        dt = time.monotonic() - t0
        served = R.summarize(out)[0] if status == "ok" else None
        n_req = len(json.load(open(R.REPO / "solutions/li_lim/manifests" / f"{stem}.json"))["requests"])
        n_rec = 0
        logs = glob.glob(str(out / "**/assignment_data.jsonl"), recursive=True)
        if status == "ok" and logs:
            cols = None
            for line in open(logs[0]):
                r = json.loads(line)
                if r.get("message") != "CandidateInsertions":
                    continue
                cols = r["extra"]["columns"]
                df = pd.DataFrame(r["extra"]["candidates"], columns=cols)
                df.insert(0, "window", n_rec)
                df.insert(0, "instance", stem)
                frames.append(df)
                n_rec += 1
        runs.append(dict(instance=stem, requests=n_req, status=status, served=served, windows=n_rec, seconds=round(dt, 1)))
        print(runs[-1], flush=True)
        pd.DataFrame(runs).to_csv(R.REPO / f"outputs/candidate_insertions_{tag}_runs.csv", index=False)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(R.REPO / f"outputs/candidate_insertions_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
