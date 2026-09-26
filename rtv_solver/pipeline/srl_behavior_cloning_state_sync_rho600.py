"""
Same as srl_behavior_cloning_training.py (actor/RHO state sync), except RHO_BATCH_INTERVAL=600
instead of 400 - a wider RHO horizon relative to the actor's fixed bi200/ss100, to see whether
the actor/RHO gap (and whether state sync still holds up) changes at a bigger horizon mismatch.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch

from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.online_rtv_solver import ManifestConsistencyError
from rtv_solver.pipeline.co_base import InfeasibleAssignmentError
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.srl_rho_outcome_advantage import ACTOR_CHECKPOINT
from rtv_solver.pipeline.srl_train_val_test_split import (
    TRAIN_INSTANCES, VAL_INSTANCES, OVERFIT_CHECK_INSTANCES,
)
from rtv_solver.pipeline.srl_training_loop import (
    REPO_ROOT, MANIFEST_DIR, _instance_service_rate, _per_instance_service_rates,
)
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed, save_json
from rtv_solver.util.logger import setup_loggers

# Actor: shorter horizon (bi200/ss100), unchanged. RHO: wider horizon (bi600/ss100) than the
# bi400 default - larger visibility gap for state sync to bridge.
ACTOR_BATCH_INTERVAL = 200
ACTOR_STEP_SIZE = 100
RHO_BATCH_INTERVAL = 600
RHO_STEP_SIZE = 100
SEED = 42
ACTOR_LR = 1e-4
EPOCHS = 20
VAL_EVERY_N_EPOCHS = 5


@dataclass
class StateSyncResult:
    best_epoch: int
    best_val_service_rate: float
    best_checkpoint_path: Path


def _run_live_rho_and_cache_manifest(instance: str, output_dir: Path) -> tuple[Path, list, float]:
    """
    Solves RHO once for this instance (RHO doesn't depend on the actor's weights, so this is
    never repeated across epochs). Returns three things:
      - rho_manifest_path: saved JSON, used as `imitation_solution_path` so the actor's y*
        target is derived from RHO's assignment instead of the Li&Lim-optimal one.
      - driver_runs: RHO's raw per-vehicle trajectory (position/route over time) - the actual
        state-sync source, injected into the actor's own iterations later.
      - rho_rate: RHO's own service rate, logged only for reference.
    """
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    rho_out_dir = output_dir / "rho_baseline" / instance
    rho_out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=rho_out_dir, MODE="coaml", BATCH_INTERVAL=RHO_BATCH_INTERVAL, STEP_SIZE=RHO_STEP_SIZE, SEED=SEED)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)
    # mode="offline" = the plain rolling-horizon solver (no ML) - this IS RHO.
    pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path)
    driver_runs = pipeline.solve_pdptw(cleared_payload, mode="offline")
    rho_rate = _instance_service_rate(config, cleared_payload, driver_runs)

    rho_manifest_payload = {
        PayloadKeys.DEPOT: cleared_payload[PayloadKeys.DEPOT],
        PayloadKeys.REQUESTS: cleared_payload[PayloadKeys.REQUESTS],
        PayloadKeys.DRIVERS: driver_runs,
        PayloadKeys.TIME_MATRIX: cleared_payload.get(PayloadKeys.TIME_MATRIX, None),
    }
    rho_manifest_path = rho_out_dir / "rho_manifest.json"
    save_json(rho_manifest_payload, rho_manifest_path)
    return rho_manifest_path, driver_runs, rho_rate


def _train_one_instance(
    instance: str, model: torch.nn.Module | None, optimizer, rho_manifest_path: Path, rho_driver_runs: list,
    output_dir: Path, epoch: int,
):
    """
    One training pass over `instance`: the actor solves it with ITS OWN (shorter) horizon,
    makes one gradient step per rolling-horizon iteration (inside solve_pdptw, mode="train"),
    and - via rho_state_sync_manifest below - has its vehicle state forced back to RHO's own
    state after every iteration, so it never drifts into a situation RHO's manifest doesn't
    match.
    """
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    train_out_dir = output_dir / "train" / f"epoch_{epoch}" / instance
    train_out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=train_out_dir, MODE="coaml", BATCH_INTERVAL=ACTOR_BATCH_INTERVAL, STEP_SIZE=ACTOR_STEP_SIZE, SEED=SEED, IMITATION_SCORING_RULE="exponential_prefix")
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)

    pipeline = COAMLPipeline(
        config, cleared_payload, model=model,
        imitation_solution_path=rho_manifest_path,
        rho_state_sync_manifest=rho_driver_runs,
    )
    if model is None:
        pipeline.load_model_weights(ACTOR_CHECKPOINT)
    if optimizer is None:
        optimizer = torch.optim.Adam(pipeline.model.parameters(), lr=ACTOR_LR)

    pipeline.solve_pdptw(cleared_payload, mode="train", optimizer=optimizer)
    valid_losses = [l for l in pipeline.loss_history if l is not None]
    mean_fy_loss = sum(valid_losses) / len(valid_losses) if valid_losses else None
    return pipeline.model, optimizer, mean_fy_loss


def run(output_dir: Path) -> StateSyncResult:
    """
    Full train/val loop, in three stages:
      1. Cache one live-RHO baseline per TRAIN instance (once, before any epoch).
      2. Train the actor epoch by epoch, state-synced to that baseline (see _train_one_instance).
      3. Every VAL_EVERY_N_EPOCHS epochs, evaluate on VAL_INSTANCES (mode="eval", no training,
         no state sync needed - the actor just runs normally with its own horizon) and keep the
         best-val checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print(f"=== Caching live RHO baselines (bi{RHO_BATCH_INTERVAL}/ss{RHO_STEP_SIZE}) for {len(TRAIN_INSTANCES)} train instances ===")
    rho_manifest_path_by_instance: dict[str, Path] = {}
    rho_driver_runs_by_instance: dict[str, list] = {}
    for instance in TRAIN_INSTANCES:
        manifest_path, driver_runs, rho_rate = _run_live_rho_and_cache_manifest(instance, output_dir)
        rho_manifest_path_by_instance[instance] = manifest_path
        rho_driver_runs_by_instance[instance] = driver_runs
        print(f"  {instance}: rho_service_rate={rho_rate:.4f}")

    model = None
    optimizer = None
    best_val_service_rate = -1.0
    best_epoch = -1
    best_checkpoint_path = output_dir / "best_actor_checkpoint.pt"
    val_curve = []
    overfit_curve = []

    for epoch in range(1, EPOCHS + 1):
        shuffled = TRAIN_INSTANCES.copy()
        rng.shuffle(shuffled)

        epoch_losses = []
        for instance in shuffled:
            try:
                model, optimizer, mean_fy_loss = _train_one_instance(
                    instance, model, optimizer,
                    rho_manifest_path_by_instance[instance], rho_driver_runs_by_instance[instance],
                    output_dir, epoch,
                )
            except InfeasibleAssignmentError as e:
                print(f"[state_sync_loop_rho600] epoch {epoch}: SKIPPING {instance} - {e}")
                continue
            except ManifestConsistencyError as e:
                print(f"[state_sync_loop_rho600] epoch {epoch}: SKIPPING {instance} - {e}")
                continue
            if mean_fy_loss is not None:
                epoch_losses.append(mean_fy_loss)

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
        print(f"[state_sync_loop_rho600] epoch {epoch}/{EPOCHS}: mean_fy_loss={avg_loss}")

        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == EPOCHS:
            config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=ACTOR_BATCH_INTERVAL, STEP_SIZE=ACTOR_STEP_SIZE, SEED=SEED)
            val_rates = _per_instance_service_rates(VAL_INSTANCES, model, config_template, output_dir, epoch, tag="val")
            overfit_rates = _per_instance_service_rates(OVERFIT_CHECK_INSTANCES, model, config_template, output_dir, epoch, tag="overfit_check")
            val_rate = sum(val_rates.values()) / max(len(val_rates), 1)
            overfit_rate = sum(overfit_rates.values()) / max(len(overfit_rates), 1)
            val_curve.append({"epoch": epoch, "service_rate": val_rate, "per_instance": val_rates})
            overfit_curve.append({"epoch": epoch, "service_rate": overfit_rate, "per_instance": overfit_rates})
            print(f"[state_sync_loop_rho600] epoch {epoch}: val={val_rate:.4f} overfit_check={overfit_rate:.4f}")

            if val_rate > best_val_service_rate:
                best_val_service_rate = val_rate
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict()}, best_checkpoint_path)

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    return StateSyncResult(best_epoch=best_epoch, best_val_service_rate=best_val_service_rate, best_checkpoint_path=best_checkpoint_path)


if __name__ == "__main__":
    result = run(REPO_ROOT / "outputs" / "srl_behavior_cloning_state_sync_rho600")
    print(f"=== DONE: best_epoch={result.best_epoch} best_val_service_rate={result.best_val_service_rate:.4f} ===")
