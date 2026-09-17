"""
2026-09-17: same as srl_behavior_cloning_vs_rho_loop.py (see its docstring
for the full method/design rationale), but with an even longer RHO horizon
- bi600/ss100 instead of bi400/ss100 - actor stays at bi200/ss100 (see
chat). Tests whether widening the horizon gap further changes the actor's
ability to imitate RHO (and the ManifestConsistencyError skip rate).
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

ACTOR_BATCH_INTERVAL = 200
ACTOR_STEP_SIZE = 100
# 2026-09-17: widened further from 400 to 600 (see chat/module docstring).
RHO_BATCH_INTERVAL = 600
RHO_STEP_SIZE = 100
SEED = 42
ACTOR_LR = 1e-4
EPOCHS = 20
VAL_EVERY_N_EPOCHS = 5


@dataclass
class BehaviorCloningVsRhoResult:
    best_epoch: int
    best_val_service_rate: float
    best_checkpoint_path: Path


def _run_live_rho_and_cache_manifest(instance: str, output_dir: Path) -> tuple[Path, float]:
    """Computed ONCE per instance - RHO's solution doesn't depend on the actor, see module docstring."""
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    rho_out_dir = output_dir / "rho_baseline" / instance
    rho_out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=rho_out_dir, MODE="coaml", BATCH_INTERVAL=RHO_BATCH_INTERVAL, STEP_SIZE=RHO_STEP_SIZE, SEED=SEED)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)
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
    return rho_manifest_path, rho_rate


def _train_one_instance(instance: str, model: torch.nn.Module | None, optimizer, rho_manifest_path: Path, output_dir: Path, epoch: int):
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    train_out_dir = output_dir / "train" / f"epoch_{epoch}" / instance
    train_out_dir.mkdir(parents=True, exist_ok=True)
    # 2026-09-17: IMITATION_SCORING_RULE="exponential_prefix" - see
    # srl_behavior_cloning_vs_rho_loop.py's matching comment/chat for why.
    config = Config(OUTPUT_DIR=train_out_dir, MODE="coaml", BATCH_INTERVAL=ACTOR_BATCH_INTERVAL, STEP_SIZE=ACTOR_STEP_SIZE, SEED=SEED, IMITATION_SCORING_RULE="exponential_prefix")
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)

    pipeline = COAMLPipeline(config, cleared_payload, model=model, imitation_solution_path=rho_manifest_path)
    if model is None:
        pipeline.load_model_weights(ACTOR_CHECKPOINT)
    if optimizer is None:
        optimizer = torch.optim.Adam(pipeline.model.parameters(), lr=ACTOR_LR)

    pipeline.solve_pdptw(cleared_payload, mode="train", optimizer=optimizer)
    valid_losses = [l for l in pipeline.loss_history if l is not None]
    mean_fy_loss = sum(valid_losses) / len(valid_losses) if valid_losses else None
    return pipeline.model, optimizer, mean_fy_loss


def run(output_dir: Path) -> BehaviorCloningVsRhoResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print(f"=== Caching live RHO baselines (bi{RHO_BATCH_INTERVAL}/ss{RHO_STEP_SIZE}) for {len(TRAIN_INSTANCES)} train instances ===")
    rho_manifest_by_instance: dict[str, Path] = {}
    rho_rate_by_instance: dict[str, float] = {}
    for instance in TRAIN_INSTANCES:
        manifest_path, rho_rate = _run_live_rho_and_cache_manifest(instance, output_dir)
        rho_manifest_by_instance[instance] = manifest_path
        rho_rate_by_instance[instance] = rho_rate
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
                    instance, model, optimizer, rho_manifest_by_instance[instance], output_dir, epoch,
                )
            except InfeasibleAssignmentError as e:
                print(f"[bc_vs_rho_loop_bi600] epoch {epoch}: SKIPPING {instance} - {e}")
                continue
            except ManifestConsistencyError as e:
                print(f"[bc_vs_rho_loop_bi600] epoch {epoch}: SKIPPING {instance} - {e}")
                continue
            if mean_fy_loss is not None:
                epoch_losses.append(mean_fy_loss)

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
        print(f"[bc_vs_rho_loop_bi600] epoch {epoch}/{EPOCHS}: mean_fy_loss={avg_loss}")

        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == EPOCHS:
            config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=ACTOR_BATCH_INTERVAL, STEP_SIZE=ACTOR_STEP_SIZE, SEED=SEED)
            val_rates = _per_instance_service_rates(VAL_INSTANCES, model, config_template, output_dir, epoch, tag="val")
            overfit_rates = _per_instance_service_rates(OVERFIT_CHECK_INSTANCES, model, config_template, output_dir, epoch, tag="overfit_check")
            val_rate = sum(val_rates.values()) / max(len(val_rates), 1)
            overfit_rate = sum(overfit_rates.values()) / max(len(overfit_rates), 1)
            val_curve.append({"epoch": epoch, "service_rate": val_rate, "per_instance": val_rates})
            overfit_curve.append({"epoch": epoch, "service_rate": overfit_rate, "per_instance": overfit_rates})
            print(f"[bc_vs_rho_loop_bi600] epoch {epoch}: val={val_rate:.4f} overfit_check={overfit_rate:.4f}")

            if val_rate > best_val_service_rate:
                best_val_service_rate = val_rate
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict()}, best_checkpoint_path)

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    mean_train_rho_rate = sum(rho_rate_by_instance.values()) / len(rho_rate_by_instance)
    print(f"[bc_vs_rho_loop_bi600] mean RHO service rate over {len(TRAIN_INSTANCES)} train instances = {mean_train_rho_rate:.4f}")

    csv_path = output_dir / "bc_vs_rho_curves.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = (
            ["epoch", "val_service_rate_avg"] + [f"val_{inst}" for inst in VAL_INSTANCES]
            + ["overfit_check_service_rate_avg"] + [f"overfit_{inst}" for inst in OVERFIT_CHECK_INSTANCES]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v, o in zip(val_curve, overfit_curve):
            row = {"epoch": v["epoch"], "val_service_rate_avg": v["service_rate"], "overfit_check_service_rate_avg": o["service_rate"]}
            row.update({f"val_{inst}": rate for inst, rate in v["per_instance"].items()})
            row.update({f"overfit_{inst}": rate for inst, rate in o["per_instance"].items()})
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    epochs_x = [r["epoch"] for r in val_curve]
    for inst in VAL_INSTANCES:
        ax.plot(epochs_x, [r["per_instance"][inst] for r in val_curve], color="tab:blue", alpha=0.25, linewidth=1)
    for inst in OVERFIT_CHECK_INSTANCES:
        ax.plot(epochs_x, [r["per_instance"][inst] for r in overfit_curve], color="tab:orange", alpha=0.25, linewidth=1)
    ax.plot(epochs_x, [r["service_rate"] for r in val_curve], marker="o", label="val avg (9 instances)", color="tab:blue", linewidth=2.5)
    ax.plot(epochs_x, [r["service_rate"] for r in overfit_curve], marker="o", label="train-subset avg (9 instances, overfit check)", color="tab:orange", linewidth=2.5)
    ax.axhline(mean_train_rho_rate, color="tab:red", linestyle=":", linewidth=1.5, label=f"mean RHO rate on train instances ({mean_train_rho_rate:.3f})")
    if best_epoch in epochs_x:
        ax.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=1.5)
        ax.scatter([best_epoch], [best_val_service_rate], color="tab:green", zorder=5, s=80,
                   label=f"best val model (epoch {best_epoch}, {best_val_service_rate:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Service rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Behavior Cloning vs. live RHO (bi200 actor vs. bi600 RHO) - 38-instance split")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "bc_vs_rho_curves.png", dpi=150)
    fig.savefig(output_dir / "bc_vs_rho_curves.pdf")
    plt.close(fig)

    print(f"[bc_vs_rho_loop_bi600] best val service rate = {best_val_service_rate:.4f} at epoch {best_epoch} -> {best_checkpoint_path}")
    return BehaviorCloningVsRhoResult(best_epoch=best_epoch, best_val_service_rate=best_val_service_rate, best_checkpoint_path=best_checkpoint_path)


if __name__ == "__main__":
    _output_dir = REPO_ROOT / "outputs" / "srl_behavior_cloning_vs_rho_loop_bi600" / "run1"
    _output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(_output_dir)
    except Exception:
        import traceback
        crash_path = _output_dir / "crash_traceback.txt"
        with open(crash_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! srl_behavior_cloning_vs_rho_loop_bi600 CRASHED - full traceback written to {crash_path}")
        raise
