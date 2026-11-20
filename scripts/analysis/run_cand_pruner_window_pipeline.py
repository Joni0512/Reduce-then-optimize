"""
2026-11-20: data + training pipeline of the Request Pruner with vs. without the four vehicle-related candidate-insertion features for ONE window (batch_interval/step_size), the same steps that were run
by hand for 40/10 on 2026-11-19:
  1. run_baseline_data_c2_candidates.py   RHO baseline logs with CandidateInsertions for all 56 Li&Lim instances (C=2 unit-demand copy, cardinality 2, no pruner, seed 42)
  2. request_pruner_dataset_builder_cand  dataset/request_pruning_bi<BI>_ss<SS>_c2cand (15 base columns + 4 cand columns, split mixed_all)
  3. sweep_request_pruner_cand            432 configs x 5 seeds, variants base (13 features) and plus (13 + 4), outputs/request_pruner_mlp_bi<BI>_ss<SS>_c2cand_{base,plus}
  4. make_cand_pruner_deploy_checkpoint   copy of the plus winner with the imputation values (_deploy.pt)
Stops at the first failing step. The RHO timing test (rho_request_pruner_cand_run.py) is NOT part of this script: it is timing-sensitive and run one window at a time afterwards.
Run: ./venv/bin/python3 scripts/analysis/run_cand_pruner_window_pipeline.py --bi 100 --ss 25
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def step(name, cmd, env_extra=None):
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {name}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1", **(env_extra or {})})
    print(f"[{time.strftime('%H:%M:%S')}] END {name}: rc={r.returncode}, {time.time() - t0:.0f} s", flush=True)
    if r.returncode != 0:
        sys.exit(f"step failed: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", required=True)
    ap.add_argument("--ss", required=True)
    a = ap.parse_args()
    py = sys.executable
    step("baseline logs (C=2, candidate insertions)", [py, "scripts/analysis/run_baseline_data_c2_candidates.py", "--bi", a.bi, "--ss", a.ss], {"RHO_TIMEOUT_S": "1800"})
    step("dataset builder", [py, "-m", "rtv_solver.pipeline.request_pruner_dataset_builder_cand", "--bi", a.bi, "--ss", a.ss])
    step("sweep base + plus", [py, "-m", "rtv_solver.pipeline.sweep_request_pruner_cand", "--bi", a.bi, "--ss", a.ss])
    step("deploy checkpoint", [py, "scripts/analysis/make_cand_pruner_deploy_checkpoint.py", "--bi", a.bi, "--ss", a.ss])
    print("PIPELINE DONE", a.bi, a.ss, flush=True)


if __name__ == "__main__":
    main()
