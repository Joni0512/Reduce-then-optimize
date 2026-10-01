"""
2026-10-01 (see chat): repeated RHO validation runs for the horizon study, standalone so it can run on the
cluster (the local helper lived in outputs/, which is not under version control).

RHO = COAMLPipeline in mode="offline" (CO_TripCostMinimization step ILP), evaluated on the 9 VAL_INSTANCES.
The solver is non-deterministic (apply_async trip-cost order), so the same config is repeated --runs times;
every run gets its own output dir so per-instance solver logs are never overwritten. An instance whose own
RHO ILP becomes infeasible is reported as FAILED and excluded from that run's mean (it does not stop the job).

Usage:
  python -m rtv_solver.pipeline.rho_val_horizon_runs --batch_interval 40 --step_size 10 --cardinality 3 \
      --seed 42 --runs 2 3 4 5
Output lines: "[rho bi=.. ss=.. card=.. seed=.. run=..] <instance>: <service_rate> (<seconds>s)" and a MEAN line;
JSON summary per run in outputs/rho_val_runs/<config>/run<N>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import time

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.srl_train_val_test_split import VAL_INSTANCES
from rtv_solver.pipeline.srl_training_loop import REPO_ROOT, MANIFEST_DIR, _instance_service_rate
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_interval", type=int, required=True)
    ap.add_argument("--step_size", type=int, required=True)
    ap.add_argument("--cardinality", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep_active", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--runs", type=int, nargs="+", default=[1], help="run indices, e.g. 2 3 4 5")
    a = ap.parse_args()
    cfg_name = f"bi{a.batch_interval}_ss{a.step_size}_card{a.cardinality}_keepactive{a.keep_active}_seed{a.seed}"
    tag = f"[rho bi={a.batch_interval} ss={a.step_size} card={a.cardinality} seed={a.seed}"
    for run in a.runs:
        run_dir = REPO_ROOT / "outputs" / "rho_val_runs" / cfg_name / f"run{run}"
        rates, failed, times, t_run = {}, {}, {}, time.time()
        for instance in VAL_INSTANCES:
            t0 = time.time()
            input_path = MANIFEST_DIR / f"{instance}.json"
            cleared = PayloadParser.clear_vehicle_manifests(PayloadParser.load_input_data(input_path))
            out_dir = run_dir / instance
            out_dir.mkdir(parents=True, exist_ok=True)
            config = Config(OUTPUT_DIR=out_dir, MODE="coaml", BATCH_INTERVAL=a.batch_interval, STEP_SIZE=a.step_size,
                            SEED=a.seed, MAX_CARDINALITY=a.cardinality, KEEP_ACTIVE=a.keep_active)
            setup_loggers(config.OUTPUT_DIR)
            set_seed(config.SEED, config.DEBUG)
            try:
                driver_runs = COAMLPipeline(config, cleared, imitation_solution_path=input_path).solve_pdptw(cleared, mode="offline")
                rates[instance] = _instance_service_rate(config, cleared, driver_runs)
                times[instance] = time.time() - t0
                print(f"{tag} run={run}] {instance}: {rates[instance]:.4f} ({times[instance]:.1f}s)", flush=True)
            except Exception as e:  # infeasible RHO ILP etc. - skip instance, keep the job alive
                failed[instance] = f"{type(e).__name__}: {e}"
                print(f"{tag} run={run}] {instance}: FAILED {failed[instance]}", flush=True)
        mean = sum(rates.values()) / max(len(rates), 1)
        print(f"{tag} run={run}] MEAN over {len(rates)} VAL_INSTANCES = {mean:.4f} (total {time.time() - t_run:.1f}s)", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "summary.json", "w") as f:
            json.dump(dict(config=vars(a), run=run, rates=rates, failed=failed, seconds=times, mean=mean), f, indent=1)


if __name__ == "__main__":
    main()
