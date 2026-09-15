"""
2026-09-15: RHO-outcome-advantage method (see srl_rho_outcome_advantage_loop.py's
docstring for the full mechanism - RHO baseline at bi400/ss100, actor rollout
at bi200/ss100 with Gaussian exploration perturbation sigma=0.2, Option A
uniform advantage broadcast, loss=-advantage*score, lr=1e-4) run on the
same balanced 12-instance train/test split used by the older
run_srl_balanced_*_12instances.py scripts (see run_srl_balanced_frozen_12instances.py),
instead of the full 38-instance split - a quick/cheap test of the method on
the smaller, already-familiar instance set (see chat, 2026-09-15).

Difference from srl_rho_outcome_advantage_loop.py: the 12-instance split has
no separate VAL/OVERFIT_CHECK subsets (only TRAIN/TEST), so validation here
runs on the 12 TEST_INSTANCES every VAL_EVERY_N_EPOCHS epochs and there is
no second "overfit check" curve.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.pipeline.co_base import InfeasibleAssignmentError
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    TRAIN_INSTANCES, TEST_INSTANCES,
)
from rtv_solver.pipeline.srl_training_loop import (
    REPO_ROOT, MANIFEST_DIR, _instance_service_rate, _per_instance_service_rates,
)
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

ACTOR_BATCH_INTERVAL = 200
ACTOR_STEP_SIZE = 100
RHO_BATCH_INTERVAL = 400
RHO_STEP_SIZE = 100
SEED = 42
ACTOR_LR = 1e-4
EPOCHS = 20
VAL_EVERY_N_EPOCHS = 5
MINIBATCH_SIZE = 10


@dataclass
class RhoOutcomeAdvantageResult:
    best_epoch: int
    best_val_service_rate: float
    best_checkpoint_path: Path


def _run_rho_baseline(instance: str, output_dir: Path, epoch: int, seed: int) -> float:
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    inst_out_dir = output_dir / "rho_baseline" / f"epoch_{epoch}" / instance
    inst_out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=inst_out_dir, MODE="coaml", BATCH_INTERVAL=RHO_BATCH_INTERVAL, STEP_SIZE=RHO_STEP_SIZE, SEED=seed)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)
    pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path)
    driver_runs = pipeline.solve_pdptw(cleared_payload, mode="offline")
    return _instance_service_rate(config, cleared_payload, driver_runs)


def _run_actor_episode(instance: str, model: torch.nn.Module | None, output_dir: Path, epoch: int, seed: int):
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    inst_out_dir = output_dir / "train" / f"epoch_{epoch}" / instance
    inst_out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=inst_out_dir, MODE="coaml", BATCH_INTERVAL=ACTOR_BATCH_INTERVAL, STEP_SIZE=ACTOR_STEP_SIZE, SEED=seed)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)

    outcome_advantage_buffer: list[torch.Tensor] = []
    pipeline = COAMLPipeline(
        config, cleared_payload, model=model, imitation_solution_path=input_path,
        outcome_advantage_buffer=outcome_advantage_buffer,
        outcome_advantage_sigma=0.2,
    )
    if model is None:
        pipeline.load_model_weights(ACTOR_CHECKPOINT)
    driver_runs = pipeline.solve_pdptw(cleared_payload, mode="eval")
    service_rate = _instance_service_rate(config, cleared_payload, driver_runs)
    return service_rate, pipeline.model, outcome_advantage_buffer


def run(output_dir: Path) -> RhoOutcomeAdvantageResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    model = None
    actor_optimizer = None
    best_val_service_rate = -1.0
    best_epoch = -1
    best_checkpoint_path = output_dir / "best_actor_checkpoint.pt"
    val_curve = []

    for epoch in range(1, EPOCHS + 1):
        shuffled = TRAIN_INSTANCES.copy()
        rng.shuffle(shuffled)

        epoch_buffer: list[tuple[torch.Tensor, float]] = []
        for instance in shuffled:
            try:
                rho_rate = _run_rho_baseline(instance, output_dir, epoch, SEED)
                actor_rate, model, raw_scores = _run_actor_episode(instance, model, output_dir, epoch, SEED)
            except InfeasibleAssignmentError as e:
                print(f"[rho_outcome_advantage_12instances] epoch {epoch}: SKIPPING {instance} - {e}")
                continue
            advantage = actor_rate - rho_rate
            epoch_buffer.extend((score, advantage) for score in raw_scores)

        if model is not None and actor_optimizer is None:
            actor_optimizer = torch.optim.Adam(model.parameters(), lr=ACTOR_LR)

        if epoch_buffer:
            num_batches = max(1, len(epoch_buffer) // MINIBATCH_SIZE)
            total_loss = 0.0
            actor_optimizer.zero_grad()
            for _ in range(num_batches):
                batch = rng.sample(epoch_buffer, min(MINIBATCH_SIZE, len(epoch_buffer)))
                scores = torch.stack([s for s, _ in batch])
                targets = torch.tensor([t for _, t in batch], dtype=scores.dtype)
                loss = (-targets * scores).mean()
                total_loss = total_loss + loss
            total_loss.backward()
            actor_optimizer.step()
            print(f"[rho_outcome_advantage_12instances] epoch {epoch}/{EPOCHS}: {num_batches} mini-batches, total_loss={total_loss.item():.4f}, buffer_size={len(epoch_buffer)}")
        else:
            print(f"[rho_outcome_advantage_12instances] epoch {epoch}/{EPOCHS}: empty buffer, skipping update")

        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == EPOCHS:
            config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=ACTOR_BATCH_INTERVAL, STEP_SIZE=ACTOR_STEP_SIZE, SEED=SEED)
            val_rates = _per_instance_service_rates(TEST_INSTANCES, model, config_template, output_dir, epoch, tag="val")
            val_rate = sum(val_rates.values()) / max(len(val_rates), 1)
            val_curve.append({"epoch": epoch, "service_rate": val_rate, "per_instance": val_rates})
            print(f"[rho_outcome_advantage_12instances] epoch {epoch}: val={val_rate:.4f}")

            if val_rate > best_val_service_rate:
                best_val_service_rate = val_rate
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict()}, best_checkpoint_path)

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    csv_path = output_dir / "rho_outcome_advantage_curves.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["epoch", "val_service_rate_avg"] + [f"val_{inst}" for inst in TEST_INSTANCES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in val_curve:
            row = {"epoch": v["epoch"], "val_service_rate_avg": v["service_rate"]}
            row.update({f"val_{inst}": rate for inst, rate in v["per_instance"].items()})
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    epochs_x = [r["epoch"] for r in val_curve]
    for inst in TEST_INSTANCES:
        ax.plot(epochs_x, [r["per_instance"][inst] for r in val_curve], color="tab:blue", alpha=0.25, linewidth=1)
    ax.plot(epochs_x, [r["service_rate"] for r in val_curve], marker="o", label="test avg (12 instances)", color="tab:blue", linewidth=2.5)
    if best_epoch in epochs_x:
        ax.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=1.5)
        ax.scatter([best_epoch], [best_val_service_rate], color="tab:green", zorder=5, s=80,
                   label=f"best test model (epoch {best_epoch}, {best_val_service_rate:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Service rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("RHO-outcome-advantage training (12-instance balanced split) - individual instances (thin) + average (bold)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "rho_outcome_advantage_curves.png", dpi=150)
    fig.savefig(output_dir / "rho_outcome_advantage_curves.pdf")
    plt.close(fig)

    print(f"[rho_outcome_advantage_12instances] best val service rate = {best_val_service_rate:.4f} at epoch {best_epoch} -> {best_checkpoint_path}")
    return RhoOutcomeAdvantageResult(best_epoch=best_epoch, best_val_service_rate=best_val_service_rate, best_checkpoint_path=best_checkpoint_path)


if __name__ == "__main__":
    _output_dir = REPO_ROOT / "outputs" / "srl_rho_outcome_advantage_loop_12instances" / "run1"
    _output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(_output_dir)
    except Exception:
        import traceback
        crash_path = _output_dir / "crash_traceback.txt"
        with open(crash_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! srl_rho_outcome_advantage_loop_12instances CRASHED - full traceback written to {crash_path}")
        raise
