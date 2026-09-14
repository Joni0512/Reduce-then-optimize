"""
2026-09-14: single-instance smoke test of Option 1 (Behavior Cloning) - see
chat/srl deck's "Training Signal Design" slide. Direct comparison point for
srl_rho_outcome_advantage.py's Option 2 smoke test, same instance/setup.

Unlike Option 2, this needs NO buffer and NO full rollout before training:
COAMLPipeline(mode="train") already IS behavior cloning - at every rolling-
horizon iteration it replays the expert (RHO-equivalent, published-optimal)
solution's y*-matching and computes a Fenchel-Young loss against it
immediately, one gradient step per iteration, per README/coaml_pipeline.py.
This script does nothing new mechanically - it just runs that existing path
for a few epochs on one instance and tracks (a) the FY loss and (b) the
actor's own eval-mode service rate after each epoch, so it's directly
comparable to the Option 2 smoke test's per-epoch service-rate curve.
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
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42
LR = 1e-4


def _eval_service_rate(instance: str, model: torch.nn.Module, output_dir: Path, epoch: int) -> float:
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    out_dir = output_dir / "eval" / f"epoch_{epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=out_dir, MODE="coaml", BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)
    pipeline = COAMLPipeline(config, cleared_payload, model=model, imitation_solution_path=input_path)
    driver_runs = pipeline.solve_pdptw(cleared_payload, mode="eval")
    return _service_rate(config, cleared_payload, driver_runs)


def main(instance: str = "lc101", epochs: int = 20) -> None:
    output_dir = REPO_ROOT / "outputs" / "srl_behavior_cloning_smoketest" / instance
    input_path = MANIFEST_DIR / f"{instance}.json"

    model = None
    optimizer = None

    for epoch in range(1, epochs + 1):
        payload = PayloadParser.load_input_data(input_path)
        cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
        train_out_dir = output_dir / "train" / f"epoch_{epoch}"
        train_out_dir.mkdir(parents=True, exist_ok=True)
        config = Config(OUTPUT_DIR=train_out_dir, MODE="coaml", BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
        setup_loggers(config.OUTPUT_DIR)
        set_seed(config.SEED, config.DEBUG)

        pipeline = COAMLPipeline(config, cleared_payload, model=model, imitation_solution_path=input_path)
        if model is None:
            pipeline.load_model_weights(ACTOR_CHECKPOINT)
        if optimizer is None:
            optimizer = torch.optim.Adam(pipeline.model.parameters(), lr=LR)

        pipeline.solve_pdptw(cleared_payload, mode="train", optimizer=optimizer)
        model = pipeline.model

        valid_losses = [l for l in pipeline.loss_history if l is not None]
        mean_fy_loss = sum(valid_losses) / len(valid_losses) if valid_losses else None

        eval_rate = _eval_service_rate(instance, model, output_dir, epoch)
        print(f"epoch {epoch}: mean_fy_loss={mean_fy_loss}  eval_service_rate={eval_rate:.4f}")


if __name__ == "__main__":
    main()
