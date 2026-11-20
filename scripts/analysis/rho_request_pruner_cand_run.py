"""
2026-11-19: RHO (offline) with the Request Pruner trained WITHOUT (base, 13 features) and WITH (plus, 13 + 4 vehicle-related candidate features) vs. own baseline (no pruner).
Both pruners: window-specific for batch_interval=40/step_size=10, trained on the SAME C=2 data and with the same protocol (sweep_request_pruner_cand.py); thresholds from
select_request_pruner_threshold_cand.py (Rule-A analogue on the 8 val instances: pruning rate >= 30 %, smallest loss of first assignments); min_retention_fraction 0.3.
The plus pruner needs the single-trip costs: TripHandler defers its filtering until after generate_trip_costs (trip_handler._apply_deferred_request_pruner); the pruned requests'
trips and costs are removed before the shared trips are built. Variants per instance (interleaved, REPS repetitions): base | reqbase | reqplus, optionally with an explicit threshold: reqbase@0.2, reqplus@0.02 (2026-11-19, fair comparison at MATCHED pruning rate).

Setting: batch_interval=40, step_size=10, max_cardinality 2, C=2 on board (unit-demand copy), no total limit, 14 test instances. Per run: served, subprocess wall-clock (time.monotonic),
sum over windows of "feasible trip costs generated: X s" (solver time.time(); can be corrupted by a system clock jump - wall-clock is the reliable one) and kept/total requests (log line "Request pruner: kept a/b").
Limit per instance RHO_TIMEOUT_S (default 1800 s).

Run: RHO_TIMEOUT_S=7200 ./venv/bin/python3 scripts/analysis/rho_request_pruner_cand_run.py [--reps 2] [--instances lr109]
Output: outputs/rho_request_pruner_cand_C2_c2_bi40_ss10.csv
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


def checkpoints_for(bi, ss):
    """2026-11-20: window-specific winner checkpoints (glob, no selection json needed) -> {name: (path, selected threshold or 1.01)}; the selected threshold (Rule-A analogue json, only present for 40/10)
    is the default for plain variant names, explicit thresholds come from 'name@threshold' (e.g. @1.01 = rank-based, keep the best 30 % of the candidates per window)."""
    sel_path = R.REPO / f"outputs/request_pruner_threshold_cand_bi{bi}_ss{ss}.json"
    sel = json.load(open(sel_path)) if sel_path.exists() else {}
    ck = {}
    for name, var in (("reqbase", "base"), ("reqplus", "plus")):
        hits = glob.glob(str(R.REPO / f"outputs/request_pruner_mlp_bi{bi}_ss{ss}_c2cand_{var}/*/*_best_val_f3.pt"))
        path = hits[0].replace(str(R.REPO) + "/", "")
        if name == "reqplus":
            path = path.replace("_best_val_f3.pt", "_best_val_f3_deploy.pt")  # plus: copy with the imputation values (make_cand_pruner_deploy_checkpoint.py)
        ck[name] = (path, sel.get(var, {}).get("threshold"))
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--variants", nargs="+", default=["base", "reqbase", "reqplus"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--bi", default="40")  # 2026-11-20: window as arguments (was hard-wired to 40/10)
    ap.add_argument("--ss", default="10")
    a = ap.parse_args()
    CK = checkpoints_for(a.bi, a.ss)
    out_csv = R.REPO / f"outputs/rho_request_pruner_cand_C2_c2_bi{a.bi}_ss{a.ss}{a.tag}.csv"
    rows = []
    for rep in range(a.reps):
        for stem in a.instances:
            f = R.make_copy(stem, 2, "_rpc")
            n_req = len(json.load(open(R.REPO / "solutions/li_lim/manifests" / f"{stem}.json"))["requests"])
            for v in a.variants:
                out = R.SCRATCH / f"out_rpc_bi{a.bi}_ss{a.ss}_{stem}_{v}_rep{rep}"
                cmd = [sys.executable, "rtv_solver/main.py", "--mode", "offline", "--input_file", str(f), "--input_dir", str(f.parent) + "/",
                       "--imitation_solution_file", str(f), "--use_request_graph_pruner", "False", "--max_cardinality", "2",
                       "--batch_interval", a.bi, "--step_size", a.ss, "--output_dir", str(out)]
                if v != "base":
                    name, _, thr_override = v.partition("@")  # 2026-11-19: "reqbase@0.2" = pruner reqbase with threshold 0.2 (matched-pruning-rate comparison); plain names use the selected threshold
                    ckpt, thr = CK[name]
                    if thr_override:
                        thr = float(thr_override)
                    if thr is None:
                        raise ValueError(f"variant {v}: no selected threshold for this window, use 'name@threshold'")
                    cmd += ["--use_request_pruner", "True", "--request_pruner_model_path", str(R.REPO / ckpt), "--request_pruner_threshold", str(thr),
                            "--request_pruner_min_retention_fraction", "0.3"]
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
                    m = re.findall(r"Request pruner: kept (\d+)/(\d+) requests", txt)
                    if m:
                        kept, total = sum(int(x) for x, _ in m), sum(int(y) for _, y in m)
                rows.append(dict(instance=stem, variant=v, rep=rep, requests=n_req, status=status, served=(res[0] if res else None), wall_s=round(wall, 2),
                                 rtv_s=(round(rtv, 2) if rtv is not None else None), req_kept=kept, req_total=total))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
