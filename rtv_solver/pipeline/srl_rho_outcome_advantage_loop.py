"""
2026-09-14: full 38-instance / 30-epoch RHO-outcome-advantage training loop
- see chat, follows the single-instance smoke test in
srl_rho_outcome_advantage.py (which confirmed the mechanism runs and that
lr=1e-2 over-corrects/destabilizes after a few epochs - see that module's
docstring). Uses lr=1e-4 here per that finding.

Per-instance cycle (one "episode" = one instance, see srl deck):
    1. RHO baseline: COAMLPipeline(mode="offline") at bi400/ss100 (pure
       cost-minimization, no NN - same objective RHO itself optimizes).
    2. Actor rollout: COAMLPipeline(mode="eval") at bi200/ss100, actor's own
       scores decide, outcome_advantage_buffer collects the raw
       (gradient-attached) score of every selected candidate.
    3. advantage = actor_service_rate - rho_service_rate (higher service
       rate = better, so a positive advantage means the actor beat RHO).
    4. Broadcast advantage over this instance's buffered scores (Option A,
       uniform - see srl deck's "Advantage Broadcast" slide) -> each entry
       becomes an (score_tensor, target_float) pair.
    5. Append to the epoch-wide buffer (NOT trained yet).

Epoch cycle (all 38 TRAIN_INSTANCES, shuffled = one "group", see chat):
    - After all 38 instances: gradient-accumulated mini-batch training -
      sample several random batches of 10 from the epoch buffer, sum their
      losses, ONE backward()+step() call at the end. (NOT several separate
      backward()/step() calls on the same buffer - confirmed in the smoke
      test that a second step() after the first mutates the model in-place
      and breaks a second backward() on the same stored graph.)
    - loss per entry = -advantage * score (policy-gradient-style
      direction/magnitude signal, NOT MSE - score and advantage are not the
      same unit, see chat).
    - Every VAL_EVERY_N_EPOCHS epochs: validate (mode="eval", no training)
      on VAL_INSTANCES + OVERFIT_CHECK_INSTANCES, track best-val checkpoint,
      reload it at the end - same pattern as srl_training_loop.py.
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
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.srl_train_val_test_split import (
    TRAIN_INSTANCES, VAL_INSTANCES, OVERFIT_CHECK_INSTANCES,
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
EPOCHS = 30
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
    overfit_curve = []

    for epoch in range(1, EPOCHS + 1):
        shuffled = TRAIN_INSTANCES.copy()
        rng.shuffle(shuffled)

        epoch_buffer: list[tuple[torch.Tensor, float]] = []
        for instance in shuffled:
            rho_rate = _run_rho_baseline(instance, output_dir, epoch, SEED)
            actor_rate, model, raw_scores = _run_actor_episode(instance, model, output_dir, epoch, SEED)
            advantage = actor_rate - rho_rate
            # Option A: uniform broadcast - every score from this instance
            # gets the SAME advantage as its target.
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
            # single backward()/step() over the SUM of all sampled mini-batch
            # losses - see module docstring for why not several separate steps.
            total_loss.backward()
            actor_optimizer.step()
            print(f"[rho_outcome_advantage] epoch {epoch}/{EPOCHS}: {num_batches} mini-batches, total_loss={total_loss.item():.4f}, buffer_size={len(epoch_buffer)}")
        else:
            print(f"[rho_outcome_advantage] epoch {epoch}/{EPOCHS}: empty buffer, skipping update")

        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == EPOCHS:
            config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=ACTOR_BATCH_INTERVAL, STEP_SIZE=ACTOR_STEP_SIZE, SEED=SEED)
            val_rates = _per_instance_service_rates(VAL_INSTANCES, model, config_template, output_dir, epoch, tag="val")
            overfit_rates = _per_instance_service_rates(OVERFIT_CHECK_INSTANCES, model, config_template, output_dir, epoch, tag="overfit_check")
            val_rate = sum(val_rates.values()) / len(val_rates)
            overfit_rate = sum(overfit_rates.values()) / len(overfit_rates)
            val_curve.append({"epoch": epoch, "service_rate": val_rate, "per_instance": val_rates})
            overfit_curve.append({"epoch": epoch, "service_rate": overfit_rate, "per_instance": overfit_rates})
            print(f"[rho_outcome_advantage] epoch {epoch}: val={val_rate:.4f} overfit_check={overfit_rate:.4f}")

            if val_rate > best_val_service_rate:
                best_val_service_rate = val_rate
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict()}, best_checkpoint_path)

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    csv_path = output_dir / "rho_outcome_advantage_curves.csv"
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
    if best_epoch in epochs_x:
        ax.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=1.5)
        ax.scatter([best_epoch], [best_val_service_rate], color="tab:green", zorder=5, s=80,
                   label=f"best val model (epoch {best_epoch}, {best_val_service_rate:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Service rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("RHO-outcome-advantage training - individual instances (thin) + averages (bold)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "rho_outcome_advantage_curves.png", dpi=150)
    fig.savefig(output_dir / "rho_outcome_advantage_curves.pdf")
    plt.close(fig)

    print(f"[rho_outcome_advantage] best val service rate = {best_val_service_rate:.4f} at epoch {best_epoch} -> {best_checkpoint_path}")
    return RhoOutcomeAdvantageResult(best_epoch=best_epoch, best_val_service_rate=best_val_service_rate, best_checkpoint_path=best_checkpoint_path)


if __name__ == "__main__":
    run(REPO_ROOT / "outputs" / "srl_rho_outcome_advantage_loop" / "run1")
