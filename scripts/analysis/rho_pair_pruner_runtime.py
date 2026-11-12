"""
2026-10-24: CONTROLLED runtime comparison RHO (no pruner) vs. RHO + Pair Pruner (GNNv1 rgnn_mixed_c2_pw2_v1_final, threshold 0.5), one run at a time (no other jobs on the laptop),
C=2 on board (unit-demand manifest copy), no total limit (L = none), cardinality 2, batch_interval=40, step_size=10, 14 instances, runs interleaved (base, pair) per instance,
REPS repetitions (default 2). Reason: the runtimes in the sweep CSVs were measured with up to 7 parallel processes and laptop sleep and cannot be compared.

Per run: wall-clock seconds of the solver subprocess (time.monotonic, not affected by sleep), served requests, and the sum over windows of the solver's own "RTV combos ... generated: X s"
time (TripHandler.run: includes Pair Pruner inference, trip generation and costs; excludes ILP and simulation), parsed from main.log, plus allowed/candidate pairs.
Limit per run 30 min (CLAUDE.md rule). Use `caffeinate -i` to keep the laptop awake.

Run: caffeinate -i ./venv/bin/python3 scripts/analysis/rho_pair_pruner_runtime.py [--reps 2]
Output: outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10.csv
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

REPO, SCRATCH, TIMEOUT_S = R.REPO, R.SCRATCH, R.TIMEOUT_S
PAIR_CKPT = REPO / "outputs/models_v2/rgnn_mixed_c2_pw2_v1_final/rgnn_mixed_c2_pw2_v1_final_best_val_f3.pt"
# 2026-11-01: window-specific Request Pruner for batch_interval=40/step_size=10 (RHO labels, sweep winner h64_l3_d0p0_pw3p0_lr0p03), see build_window_specific_request_pruners.py
REQ_CKPT = "outputs/request_pruner_mlp_bi40_ss10/request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03/request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03_best_val_f3.pt"
INSTANCES = ["lc108", "lc109", "lr109", "lr111", "lr112", "lrc107", "lrc108", "lc207", "lc208", "lr208", "lr210", "lr211", "lrc207", "lrc208"]


def parse_variant(v):
    """2026-11-01: variant name -> (pair threshold or None, request threshold or None). base | pair<t> | req<t> | comb<t_req>_<t_pair>, e.g. pair0.5, req0.2, comb0.2_0.2.
    The old names "pair" (= pair0.5) and "base" are kept so that the 2026-10-24/26 CSVs stay reproducible."""
    if v == "base":
        return None, None
    if v == "pair":
        return 0.5, None
    if v.startswith("pair"):
        return float(v[4:]), None
    if v.startswith("req"):
        return None, float(v[3:])
    if v.startswith("comb"):
        t_req, t_pair = v[4:].split("_")
        return float(t_pair), float(t_req)
    raise ValueError(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--variants", nargs="+", default=["base", "pair"], help="2026-11-01: run order per instance, e.g. base pair0.5 req0.5 comb0.5_0.5 (default = old behaviour: base pair)")
    ap.add_argument("--req_ckpt", default=REQ_CKPT, help="2026-11-01: Request Pruner checkpoint (must be trained for the SAME --bi/--ss window; default = the 40/10 one)")
    # 2026-11-06: window and total limit as arguments (were hard-wired to 40/10 and no limit) so the same comparison can run for 100/25, 120/30, 200/50 (and 40/10 at L=16).
    # Defaults (bi 40, ss 10, L 0 = no limit) keep the old behaviour and the old CSV name.
    ap.add_argument("--bi", type=int, default=40, help="2026-11-06: batch_interval")
    ap.add_argument("--ss", type=int, default=10, help="2026-11-06: step_size")
    ap.add_argument("--L", type=int, default=0, help="2026-11-06: max requests per vehicle in total (--max_requests_per_vehicle); 0 = no limit")
    ap.add_argument("--rep_start", type=int, default=0, help="2026-10-26: resume support - first repetition index to run")
    ap.add_argument("--tag", default="", help="2026-10-26: CSV name suffix so that a resumed run does not overwrite the first file")
    a = ap.parse_args()
    args_req_ckpt = str(REPO / a.req_ckpt) if not os.path.isabs(a.req_ckpt) else a.req_ckpt
    rows = []
    win = "" if (a.bi, a.ss, a.L) == (40, 10, 0) else f"_bi{a.bi}_ss{a.ss}" + (f"_L{a.L}" if a.L else "")  # 2026-11-06: old name kept for 40/10 without limit
    out_csv = REPO / f"outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10{a.tag}.csv" if not win else REPO / f"outputs/rho_pair_pruner_runtime_C2_c2{win}{a.tag}.csv"
    for rep in range(a.rep_start, a.reps):
        for stem in a.instances:
            f = R.make_copy(stem, 2, "_rt")
            for variant in a.variants:
                pair_t, req_t = parse_variant(variant)
                out = SCRATCH / f"out_rt{win}_{stem}_{variant}_rep{rep}"  # 2026-11-06: window in the scratch name so parallel jobs/windows never collide
                cmd = [sys.executable, "rtv_solver/main.py", "--mode", "offline", "--input_file", str(f), "--input_dir", str(f.parent) + "/",
                       "--imitation_solution_file", str(f), "--use_request_graph_pruner", "True" if pair_t is not None else "False",
                       "--max_cardinality", "2", "--batch_interval", str(a.bi), "--step_size", str(a.ss), "--output_dir", str(out)]
                if a.L:  # 2026-11-06: total limit of requests per vehicle (same flag as rho_total_limit_sweep.py)
                    cmd += ["--max_requests_per_vehicle", str(a.L)]
                if pair_t is not None:
                    cmd += ["--request_graph_model_path", str(PAIR_CKPT), "--request_graph_threshold", str(pair_t)]
                if req_t is not None:  # 2026-11-01: min_retention_fraction 0.3 = main.py default, same as in the SIL runs
                    cmd += ["--use_request_pruner", "True", "--request_pruner_model_path", args_req_ckpt, "--request_pruner_threshold", str(req_t),
                            "--request_pruner_min_retention_fraction", "0.3"]
                t0 = time.monotonic()
                try:
                    subprocess.run(cmd, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=TIMEOUT_S, env={**os.environ, "PYTHONPATH": str(REPO)})
                    status = "ok"
                except subprocess.TimeoutExpired:
                    status = "timeout"
                wall = time.monotonic() - t0
                res = R.summarize(out) if status == "ok" else None
                rtv = pk = pt = rk = rt_ = None
                logs = glob.glob(str(out / "run_offline_*" / "*" / "main.log"))
                if logs and status == "ok":
                    txt = open(logs[0], errors="ignore").read()
                    rtv = sum(float(x) for x in re.findall(r"feasible trip costs generated: ([0-9.]+)s", txt))
                    pairs = re.findall(r"Allowed request pairs: (\d+) / (\d+) candidate pairs", txt)
                    if pairs:
                        pk, pt = sum(int(x) for x, _ in pairs), sum(int(y) for _, y in pairs)
                    reqs = re.findall(r"Request pruner: kept (\d+)/(\d+) requests", txt)  # 2026-11-01: requests kept / requests seen, summed over windows
                    if reqs:
                        rk, rt_ = sum(int(x) for x, _ in reqs), sum(int(y) for _, y in reqs)
                rows.append(dict(instance=stem, variant=variant, rep=rep, status=status, wall_s=round(wall, 2), rtv_s=(round(rtv, 2) if rtv is not None else None),
                                 served=(res[0] if res else None), pairs_kept=pk, pairs_total=pt, req_kept=rk, req_total=rt_))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
