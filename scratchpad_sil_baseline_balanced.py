from pathlib import Path
import csv

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import StatsParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = REPO_ROOT / "solutions" / "li_lim" / "manifests"
ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

TEST_INSTANCES = ["lc107", "lc108", "lr111", "lr112", "lrc107", "lrc108", "lc207", "lc208", "lr203", "lr210", "lrc201", "lrc207"]

output_dir = REPO_ROOT / "outputs" / "true_sil_baseline_balanced"
output_dir.mkdir(parents=True, exist_ok=True)

rows = []
for instance in TEST_INSTANCES:
    input_path = MANIFEST_DIR / f"{instance}.json"
    instance_output_dir = output_dir / instance
    instance_output_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=instance_output_dir, MODE="coaml", BATCH_INTERVAL=200, STEP_SIZE=100, SEED=42)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)

    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)

    pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path)
    pipeline.load_model_weights(ACTOR_CHECKPOINT)
    final_driver_runs = pipeline.solve_pdptw(cleared_payload, mode="eval")

    full_payload_object = PayloadParser.get_payload_object(
        cleared_payload, dwell_pickup_default=config.DWELL_PICKUP, dwell_alight_default=config.DWELL_ALIGHT, online=False,
    )
    all_requests = RequestHandler(full_payload_object.requests, config=config).get_all_requests()
    stats_payload = {
        PayloadKeys.DEPOT: cleared_payload[PayloadKeys.DEPOT],
        PayloadKeys.REQUESTS: cleared_payload[PayloadKeys.REQUESTS],
        PayloadKeys.DRIVERS: final_driver_runs,
        PayloadKeys.TIME_MATRIX: cleared_payload.get(PayloadKeys.TIME_MATRIX, None),
    }
    _, episode_stats, _ = StatsParser(config, payload=stats_payload).evaluate(stats_payload)
    num_requests = len(all_requests)
    num_serviced = len(episode_stats.serviced_requests)
    service_rate = num_serviced / num_requests if num_requests > 0 else 0.0
    print(f"{instance}: true SIL baseline (balanced) service_rate={service_rate:.4f} ({num_serviced}/{num_requests})")
    rows.append({"instance": instance, "service_rate": service_rate})

csv_path = output_dir / "true_sil_baseline_balanced.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["instance", "service_rate"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved {csv_path}")
