"""
2026-09-14: Behavior Cloning (Option 1) smoke test against a LIVE RHO run
(mode="offline" at bi400/ss100), not the precomputed published-optimal
manifest - see chat. The actor trains at its OWN horizon (bi200/ss100);
different horizons are fine because ImitationHandler only extracts each
vehicle's final pickup sequence from the given manifest file
(_manifest_to_request_solution) - it doesn't need per-iteration state
alignment between the actor's and RHO's rolling-horizon schedules.

Direct comparison point for srl_behavior_cloning_smoketest.py (which
imitates the published-optimal solution instead) and
srl_rho_outcome_advantage.py (Option 2, same bi200-vs-bi400 setup).
"""
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

from pathlib import Path

import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.srl_rho_outcome_advantage import ACTOR_CHECKPOINT, MANIFEST_DIR, REPO_ROOT, _service_rate
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed, save_json
from rtv_solver.util.logger import setup_loggers

ACTOR_BATCH_INTERVAL = 200
ACTOR_STEP_SIZE = 100
RHO_BATCH_INTERVAL = 400
RHO_STEP_SIZE = 100
SEED = 42
LR = 1e-4


def _run_live_rho_and_save_manifest(instance: str, output_dir: Path) -> tuple[Path, float]:
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    rho_out_dir = output_dir / "rho_baseline"
    rho_out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=rho_out_dir, MODE="coaml", BATCH_INTERVAL=RHO_BATCH_INTERVAL, STEP_SIZE=RHO_STEP_SIZE, SEED=SEED)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)
    pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path)
    driver_runs = pipeline.solve_pdptw(cleared_payload, mode="offline")
    rho_rate = _service_rate(config, cleared_payload, driver_runs)

    # Save in the same {depot, requests, driver_runs, time_matrix} shape
    # ImitationHandler expects (it only reads PayloadKeys.DRIVERS - see
    # imitation_handler.py's _load_complete_optimal_solution).
    rho_manifest_payload = {
        PayloadKeys.DEPOT: cleared_payload[PayloadKeys.DEPOT],
        PayloadKeys.REQUESTS: cleared_payload[PayloadKeys.REQUESTS],
        PayloadKeys.DRIVERS: driver_runs,
        PayloadKeys.TIME_MATRIX: cleared_payload.get(PayloadKeys.TIME_MATRIX, None),
    }
    rho_manifest_path = rho_out_dir / "rho_manifest.json"
    save_json(rho_manifest_payload, rho_manifest_path)
    return rho_manifest_path, rho_rate


def _eval_service_rate(instance: str, model: torch.nn.Module, output_dir, epoch: int, rho_manifest_path) -> float:
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    out_dir = output_dir / "eval" / f"epoch_{epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=out_dir, MODE="coaml", BATCH_INTERVAL=ACTOR_BATCH_INTERVAL, STEP_SIZE=ACTOR_STEP_SIZE, SEED=SEED)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)
    pipeline = COAMLPipeline(config, cleared_payload, model=model, imitation_solution_path=rho_manifest_path)
    driver_runs = pipeline.solve_pdptw(cleared_payload, mode="eval")
    return _service_rate(config, cleared_payload, driver_runs)


def main(instance: str = "lc101", epochs: int = 20) -> None:
    output_dir = REPO_ROOT / "outputs" / "srl_behavior_cloning_vs_rho_smoketest" / instance

    print(f"=== Live RHO run (bi{RHO_BATCH_INTERVAL}/ss{RHO_STEP_SIZE}) on {instance} ===")
    rho_manifest_path, rho_rate = _run_live_rho_and_save_manifest(instance, output_dir)
    print(f"RHO service_rate = {rho_rate:.4f}, manifest saved to {rho_manifest_path}")

    model = None
    optimizer = None

    for epoch in range(1, epochs + 1):
        input_path = MANIFEST_DIR / f"{instance}.json"
        payload = PayloadParser.load_input_data(input_path)
        cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
        train_out_dir = output_dir / "train" / f"epoch_{epoch}"
        train_out_dir.mkdir(parents=True, exist_ok=True)
        config = Config(OUTPUT_DIR=train_out_dir, MODE="coaml", BATCH_INTERVAL=ACTOR_BATCH_INTERVAL, STEP_SIZE=ACTOR_STEP_SIZE, SEED=SEED)
        setup_loggers(config.OUTPUT_DIR)
        set_seed(config.SEED, config.DEBUG)

        # Imitation target = the LIVE RHO manifest (bi400/ss100), NOT the
        # precomputed published-optimal solution - the actor still trains
        # at its own bi200/ss100 horizon.
        pipeline = COAMLPipeline(config, cleared_payload, model=model, imitation_solution_path=rho_manifest_path)
        if model is None:
            pipeline.load_model_weights(ACTOR_CHECKPOINT)
        if optimizer is None:
            optimizer = torch.optim.Adam(pipeline.model.parameters(), lr=LR)

        pipeline.solve_pdptw(cleared_payload, mode="train", optimizer=optimizer)
        model = pipeline.model

        valid_losses = [l for l in pipeline.loss_history if l is not None]
        mean_fy_loss = sum(valid_losses) / len(valid_losses) if valid_losses else None

        eval_rate = _eval_service_rate(instance, model, output_dir, epoch, rho_manifest_path)
        print(f"epoch {epoch}: mean_fy_loss={mean_fy_loss}  eval_service_rate={eval_rate:.4f}  (rho={rho_rate:.4f})")


if __name__ == "__main__":
    main()
