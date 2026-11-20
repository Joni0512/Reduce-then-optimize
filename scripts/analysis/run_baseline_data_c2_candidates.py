"""
2026-11-19: RHO-no-learning baseline logs (assignment_data.jsonl + config.json) for ALL 56 Li&Lim instances at ONE window config, with C=2 on board (unit-demand manifest copy, rho_capacity_sweep.make_copy),
cardinality 2, no pruner, no total limit, seed 42, --log_candidate_insertions True (record "CandidateInsertions" per ILP solve: every feasible single-request insertion request x vehicle with added_cost,
direct_time, veh_to_pickup, ..., chosen). Same layout as run_baseline_data_window.sh so request_pruner_dataset_builder.build_dataset can read it:
outputs/eval_rh_no_learning_bi<BI>_ss<SS>_c2cand/<inst>/rh_no_learning/{assignment_data.jsonl,config.json}.
Reason: user wants to test the vehicle-related feature (extra trip time / pickup time) as an ADDITIONAL Request Pruner feature; the existing pruner datasets come from baseline logs with the ORIGINAL
capacity (not C=2, the setting of the RHO pruner tests) and contain no candidate insertions. Pruners with and without the feature are to be trained on THIS data so that only the feature differs.
Per-instance limit RHO_TIMEOUT_S (default 1800 s); an instance that exceeds it is skipped and listed in the summary CSV.

Run: RHO_TIMEOUT_S=1800 ./venv/bin/python3 scripts/analysis/run_baseline_data_c2_candidates.py [--bi 40 --ss 10] [--instances lc101 ...]
Output: outputs/eval_rh_no_learning_bi<BI>_ss<SS>_c2cand/ and outputs/eval_rh_no_learning_bi<BI>_ss<SS>_c2cand_runs.csv (served, seconds, status per instance)
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rho_capacity_sweep as R  # noqa: E402

INSTANCES = ("lc101 lc102 lc103 lc104 lc105 lc106 lc107 lc108 lc109 lc201 lc202 lc203 lc204 lc205 lc206 lc207 lc208 lr101 lr102 lr103 lr104 lr105 lr106 lr107 lr108 lr109 lr110 lr111 lr112 "
             "lr201 lr202 lr203 lr204 lr205 lr206 lr207 lr208 lr209 lr210 lr211 lrc101 lrc102 lrc103 lrc104 lrc105 lrc106 lrc107 lrc108 lrc201 lrc202 lrc203 lrc204 lrc205 lrc206 lrc207 lrc208").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", default="40")
    ap.add_argument("--ss", default="10")
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--card", default="2")
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    a = ap.parse_args()
    final_base = R.REPO / f"outputs/eval_rh_no_learning_bi{a.bi}_ss{a.ss}_c{a.C}cand"
    runs = []
    for k, stem in enumerate(a.instances, 1):
        f = R.make_copy(stem, a.C, "_ds")
        out = R.SCRATCH / f"out_ds_{stem}_C{a.C}_bi{a.bi}_ss{a.ss}"
        shutil.rmtree(out, ignore_errors=True)
        cmd = [sys.executable, "rtv_solver/main.py", "--mode", "offline", "--input_file", str(f), "--input_dir", str(f.parent) + "/",
               "--imitation_solution_file", str(f), "--use_request_graph_pruner", "False", "--use_request_pruner", "False", "--seed", "42",
               "--max_cardinality", a.card, "--batch_interval", a.bi, "--step_size", a.ss, "--log_candidate_insertions", "True", "--output_dir", str(out)]
        t0 = time.monotonic()
        try:
            subprocess.run(cmd, cwd=R.REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=R.TIMEOUT_S, env={**os.environ, "PYTHONPATH": str(R.REPO)})
            status = "ok"
        except subprocess.TimeoutExpired:
            status = "timeout"
        dt = time.monotonic() - t0
        served = None
        if status == "ok":
            runs_dirs = glob.glob(str(out / "run_offline_*" / f"{stem}_*"))
            if runs_dirs:
                rd = Path(runs_dirs[0])
                target = final_base / stem / "rh_no_learning"
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy(rd / "assignment_data.jsonl", target / "assignment_data.jsonl")
                shutil.copy(rd / "config.json", target / "config.json")
                served = R.summarize(out)[0]
            else:
                status = "no_output"
        runs.append(dict(instance=stem, status=status, served=served, seconds=round(dt, 1)))
        print(f"[{k}/{len(a.instances)}]", runs[-1], flush=True)
        pd.DataFrame(runs).to_csv(R.REPO / f"outputs/eval_rh_no_learning_bi{a.bi}_ss{a.ss}_c{a.C}cand_runs.csv", index=False)
        shutil.rmtree(out, ignore_errors=True)  # keep scratch small; the logs are copied above


if __name__ == "__main__":
    main()
