"""
2026-10-30: RHO (offline, no pruner) on the 9 VAL_INSTANCES and 9 TEST_INSTANCES of the SRL split for ONE window configuration, same evaluation path as
outputs/rho_val_script_ss_seed.py (COAMLPipeline mode offline, manifests cleared, _instance_service_rate, seed 42), but val and test in one run.
Reason: SRL results at batch_interval=60/step_size=30 need a RHO reference at the same window (earlier chart used RHO bi120/ss30, logs rho_bi120_ss30_{val,test}.log,
3 runs each, because RHO is not bit-deterministic). Log format mirrors those logs ("MEAN over 9 VAL_INSTANCES = ...").

Run: ./venv/bin/python3 scripts/analysis/rho_val_test_window.py <BI> <SS> <CARD> <KEEP_ACTIVE True|False> <RUN>   (PYTHONPATH=repo root)
Output dir: outputs/rho_valtest_bi<BI>_ss<SS>_keepactive<KA>_card<CARD>_run<RUN>/<split>/<instance>/
"""
import sys
import time

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.srl_train_val_test_split import TEST_INSTANCES, VAL_INSTANCES
from rtv_solver.pipeline.srl_training_loop import MANIFEST_DIR, REPO_ROOT, _instance_service_rate
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

BI, SS, CARD, KA, RUN = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "True", int(sys.argv[5])
SEED = 42
ONLY = sys.argv[6].split(",") if len(sys.argv) > 6 else None  # optional instance filter for quick checks
tag = f"rho bi={BI} ss={SS} card={CARD} keep_active={KA} seed={SEED} run={RUN}"
for split, names in (("VAL", VAL_INSTANCES), ("TEST", TEST_INSTANCES)):
    rates, T0 = {}, time.time()
    for instance in names:
        if ONLY and instance not in ONLY:
            continue
        t0 = time.time()
        input_path = MANIFEST_DIR / f"{instance}.json"
        cleared = PayloadParser.clear_vehicle_manifests(PayloadParser.load_input_data(input_path))
        out_dir = REPO_ROOT / "outputs" / f"rho_valtest_bi{BI}_ss{SS}_keepactive{KA}_card{CARD}_run{RUN}" / split.lower() / instance
        out_dir.mkdir(parents=True, exist_ok=True)
        config = Config(OUTPUT_DIR=out_dir, MODE="coaml", BATCH_INTERVAL=BI, STEP_SIZE=SS, SEED=SEED, MAX_CARDINALITY=CARD, KEEP_ACTIVE=KA)
        setup_loggers(config.OUTPUT_DIR)
        set_seed(config.SEED, config.DEBUG)
        try:
            dr = COAMLPipeline(config, cleared, imitation_solution_path=input_path).solve_pdptw(cleared, mode="offline")
            rates[instance] = _instance_service_rate(config, cleared, dr)
            print(f"[{tag}] {instance}: {rates[instance]:.4f} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[{tag}] {instance}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"[{tag}] MEAN over {len(rates)} {split}_INSTANCES = {sum(rates.values()) / max(len(rates), 1):.4f} (total {time.time() - T0:.1f}s)", flush=True)
