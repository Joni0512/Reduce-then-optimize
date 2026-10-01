"""
One-off: RHO's own service rate at bi200/ss100 (the actor's own horizon) on the 9
VAL_INSTANCES, for a fair same-horizon/same-instance-set comparison against the actor's val
service rate from the twin-critic/SRL training loops.

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.compute_rho_val_bi200
"""
from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.srl_train_val_test_split import VAL_INSTANCES
from rtv_solver.pipeline.srl_training_loop import MANIFEST_DIR, _instance_service_rate
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42

rates = {}
for instance in VAL_INSTANCES:
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    out_dir = MANIFEST_DIR.parent.parent.parent / "outputs" / "rho_val_bi200" / instance
    out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=out_dir, MODE="coaml", BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)
    pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path)
    driver_runs = pipeline.solve_pdptw(cleared_payload, mode="offline")
    rate = _instance_service_rate(config, cleared_payload, driver_runs)
    rates[instance] = rate
    print(f"  {instance}: rho_service_rate={rate:.4f}")

mean_rate = sum(rates.values()) / len(rates)
print(f"=== RHO bi200/ss100 mean over {len(rates)} VAL_INSTANCES = {mean_rate:.4f} ===")
