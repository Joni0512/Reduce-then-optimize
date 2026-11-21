"""
2026-11-20: RHO (offline) with the Pair Pruner, the Request Pruner trained WITH the vehicle-related candidate features (reqplus, 13 + 4 features) and BOTH together, vs. own baseline (no pruner).
Variants per instance (interleaved, REPS repetitions): base | pair<t> | reqplus<t> | comb<t_req>_<t_pair>, e.g. base pair0.5 reqplus0.5 comb0.5_0.5.
  - Pair Pruner: the existing GLOBAL GNN rgnn_mixed_c2_pw2_v1_final (NOT window-specific; same checkpoint as rho_pair_pruner_runtime.py), threshold <t>.
  - Request Pruner: window-specific winner of sweep_request_pruner_cand.py (plus variant, imputation copy *_deploy.pt, rho_request_pruner_cand_run.checkpoints_for), fixed threshold <t>,
    min_retention_fraction 0.3, active/urgent force-kept, pruned requests deferred.
  - Order inside TripHandler.run for comb: the Pair Pruner scores ALL requests of the window first (the request pruner with candidate features is deferred until the single-trip costs exist,
    trip_handler._apply_deferred_request_pruner); pairs that contain a request pruned later are never used because that request's trips are removed before the shared trips are built.
    So the pair-pruner inference runs on the full request set of the window, not on the request-pruned one (differs from the old comb<..> variants with a request pruner without candidate features).
Setting: C=2 on board (unit-demand copy), max_cardinality 2, no total limit, 14 test instances. Per run: served, subprocess wall-clock (time.monotonic), sum over windows of
"feasible trip costs generated: X s" (solver time.time(), can be corrupted by clock jumps), pairs kept/total ("Allowed request pairs: a / b candidate pairs"), requests kept/total ("Request pruner: kept a/b").
Limit per instance RHO_TIMEOUT_S (default 1800 s).
Run: RHO_TIMEOUT_S=1800 ./venv/bin/python3 scripts/analysis/rho_pair_request_cand_run.py --bi 100 --ss 25 --reps 2 --variants base pair0.5 reqplus0.5 comb0.5_0.5 --tag _p05r05
Output: outputs/rho_pair_request_cand_C2_c2_bi<BI>_ss<SS><tag>.csv
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
import rho_request_pruner_cand_run as C  # noqa: E402

PAIR_CKPT = R.REPO / "outputs/models_v2/rgnn_mixed_c2_pw2_v1_final/rgnn_mixed_c2_pw2_v1_final_best_val_f3.pt"


def parse_variant(v):
    """base | pair<t> | reqplus<t> | reqbase<t> | comb<t_req>_<t_pair> | combbase<t_req>_<t_pair> -> (pair threshold or None, request threshold or None, request pruner "plus" or "base")
    2026-11-21: reqbase / combbase added = the same with the Request Pruner WITHOUT the vehicle features (13 features), to compare the combination with old vs. new features."""
    if v == "base":
        return None, None, "plus"
    if v.startswith("pair"):
        return float(v[4:]), None, "plus"
    if v.startswith("reqplus"):
        return None, float(v[7:]), "plus"
    if v.startswith("reqbase"):
        return None, float(v[7:]), "base"
    if v.startswith("combbase"):
        t_req, t_pair = v[8:].split("_")
        return float(t_pair), float(t_req), "base"
    if v.startswith("comb"):
        t_req, t_pair = v[4:].split("_")
        return float(t_pair), float(t_req), "plus"
    raise ValueError(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=C.INSTANCES)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--variants", nargs="+", default=["base", "pair0.5", "reqplus0.5", "comb0.5_0.5"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--bi", default="40")
    ap.add_argument("--ss", default="10")
    a = ap.parse_args()
    CK = C.checkpoints_for(a.bi, a.ss)  # 2026-11-21: both request pruners (reqbase = 13 features, reqplus = 13 + 4)
    out_csv = R.REPO / f"outputs/rho_pair_request_cand_C2_c2_bi{a.bi}_ss{a.ss}{a.tag}.csv"
    rows = []
    for rep in range(a.reps):
        for stem in a.instances:
            f = R.make_copy(stem, 2, "_prc")
            n_req = len(json.load(open(R.REPO / "solutions/li_lim/manifests" / f"{stem}.json"))["requests"])
            for v in a.variants:
                pair_t, req_t, req_kind = parse_variant(v)
                out = R.SCRATCH / f"out_prc_bi{a.bi}_ss{a.ss}_{stem}_{v}_rep{rep}"
                cmd = [sys.executable, "rtv_solver/main.py", "--mode", "offline", "--input_file", str(f), "--input_dir", str(f.parent) + "/",
                       "--imitation_solution_file", str(f), "--use_request_graph_pruner", "True" if pair_t is not None else "False", "--max_cardinality", "2",
                       "--batch_interval", a.bi, "--step_size", a.ss, "--output_dir", str(out)]
                if pair_t is not None:
                    cmd += ["--request_graph_model_path", str(PAIR_CKPT), "--request_graph_threshold", str(pair_t)]
                if req_t is not None:
                    cmd += ["--use_request_pruner", "True", "--request_pruner_model_path", str(R.REPO / CK["reqplus" if req_kind == "plus" else "reqbase"][0]), "--request_pruner_threshold", str(req_t),
                            "--request_pruner_min_retention_fraction", "0.3"]
                t0 = time.monotonic()
                try:
                    subprocess.run(cmd, cwd=R.REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=R.TIMEOUT_S, env={**os.environ, "PYTHONPATH": str(R.REPO)})
                    status = "ok"
                except subprocess.TimeoutExpired:
                    status = "timeout"
                wall = time.monotonic() - t0
                res = R.summarize(out) if status == "ok" else None
                rtv = rk = rt_ = pk = pt = None
                logs = glob.glob(str(out / "run_offline_*" / "*" / "main.log"))
                if logs and status == "ok":
                    txt = open(logs[0], errors="ignore").read()
                    rtv = sum(float(x) for x in re.findall(r"feasible trip costs generated: ([0-9.]+)s", txt))
                    m = re.findall(r"Request pruner: kept (\d+)/(\d+) requests", txt)
                    if m:
                        rk, rt_ = sum(int(x) for x, _ in m), sum(int(y) for _, y in m)
                    p = re.findall(r"Allowed request pairs: (\d+) / (\d+) candidate pairs", txt)
                    if p:
                        pk, pt = sum(int(x) for x, _ in p), sum(int(y) for _, y in p)
                rows.append(dict(instance=stem, variant=v, rep=rep, requests=n_req, status=status, served=(res[0] if res else None), wall_s=round(wall, 2),
                                 rtv_s=(round(rtv, 2) if rtv is not None else None), req_kept=rk, req_total=rt_, pairs_kept=pk, pairs_total=pt))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
