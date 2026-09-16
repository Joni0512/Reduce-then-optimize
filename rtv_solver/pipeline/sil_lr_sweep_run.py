"""
2026-09-16: plain SIL (behavior cloning against the precomputed Li&Lim
optimal manifest, NOT live RHO - no critic, no RL) training-from-scratch
run on the stratified 38/9/9 split (srl_train_val_test_split.py), for a
small actor_lr sweep (10 fixed values, same seed) requested in chat -
separate from the SRL/RHO-outcome-advantage method family, this is the
"classic" SIL baseline.

Model starts from scratch (COAMLPipeline with model=None and no
load_model_weights() call -> build_scoring_model() inside __init__), NOT
warm-started from an existing checkpoint (unlike srl_rho_outcome_advantage_loop.py
and srl_behavior_cloning_vs_rho_loop.py, which fine-tune an already-SIL-
trained actor).

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.sil_lr_sweep_run <actor_lr>
"""
from __future__ import annotations

import csv
import random
import sys
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
from rtv_solver.pipeline.co_base import InfeasibleAssignmentError
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.srl_train_val_test_split import (
    TRAIN_INSTANCES, VAL_INSTANCES, OVERFIT_CHECK_INSTANCES,
)
from rtv_solver.pipeline.srl_training_loop import (
    REPO_ROOT, MANIFEST_DIR, _per_instance_service_rates,
)
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42
EPOCHS = 5
VAL_EVERY_N_EPOCHS = 1  # only 5 epochs total, validate every one


def _train_one_instance(instance: str, model, optimizer, output_dir: Path, epoch: int, actor_lr: float):
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    train_out_dir = output_dir / "train" / f"epoch_{epoch}" / instance
    train_out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=train_out_dir, MODE="coaml", BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)

    # imitation_solution_path=input_path -> the precomputed Li&Lim-optimal
    # manifest for this instance (same file the pipeline itself runs on),
    # NOT a live RHO run - this is plain SIL, see module docstring.
    pipeline = COAMLPipeline(config, cleared_payload, model=model, imitation_solution_path=input_path)
    # model stays None on the very first call -> COAMLPipeline builds a
    # fresh randomly-initialized model internally (build_scoring_model()) -
    # training from scratch, no checkpoint loaded (see module docstring).
    if optimizer is None:
        optimizer = torch.optim.Adam(pipeline.model.parameters(), lr=actor_lr)

    pipeline.solve_pdptw(cleared_payload, mode="train", optimizer=optimizer)
    valid_losses = [l for l in pipeline.loss_history if l is not None]
    mean_fy_loss = sum(valid_losses) / len(valid_losses) if valid_losses else None
    return pipeline.model, optimizer, mean_fy_loss


def run(actor_lr: float, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

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
                model, optimizer, mean_fy_loss = _train_one_instance(instance, model, optimizer, output_dir, epoch, actor_lr)
            except InfeasibleAssignmentError as e:
                print(f"[sil_lr_sweep actor_lr={actor_lr:.6g}] epoch {epoch}: SKIPPING {instance} - {e}")
                continue
            if mean_fy_loss is not None:
                epoch_losses.append(mean_fy_loss)

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
        print(f"[sil_lr_sweep actor_lr={actor_lr:.6g}] epoch {epoch}/{EPOCHS}: mean_fy_loss={avg_loss}")

        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == EPOCHS:
            config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
            val_rates = _per_instance_service_rates(VAL_INSTANCES, model, config_template, output_dir, epoch, tag="val")
            overfit_rates = _per_instance_service_rates(OVERFIT_CHECK_INSTANCES, model, config_template, output_dir, epoch, tag="overfit_check")
            val_rate = sum(val_rates.values()) / max(len(val_rates), 1)
            overfit_rate = sum(overfit_rates.values()) / max(len(overfit_rates), 1)
            val_curve.append({"epoch": epoch, "service_rate": val_rate, "per_instance": val_rates})
            overfit_curve.append({"epoch": epoch, "service_rate": overfit_rate, "per_instance": overfit_rates})
            print(f"[sil_lr_sweep actor_lr={actor_lr:.6g}] epoch {epoch}: val={val_rate:.4f} overfit_check={overfit_rate:.4f}")

            if val_rate > best_val_service_rate:
                best_val_service_rate = val_rate
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict()}, best_checkpoint_path)

    csv_path = output_dir / "sil_lr_sweep_curves.csv"
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
    ax.plot(epochs_x, [r["service_rate"] for r in val_curve], marker="o", label="val avg (9 instances)", color="tab:blue", linewidth=2.5)
    ax.plot(epochs_x, [r["service_rate"] for r in overfit_curve], marker="o", label="train-subset avg (9 instances, overfit check)", color="tab:orange", linewidth=2.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Service rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Plain SIL from scratch, actor_lr={actor_lr:.6g}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "sil_lr_sweep_curves.png", dpi=150)
    fig.savefig(output_dir / "sil_lr_sweep_curves.pdf")
    plt.close(fig)

    print(f"[sil_lr_sweep actor_lr={actor_lr:.6g}] best val service rate = {best_val_service_rate:.4f} at epoch {best_epoch} -> {best_checkpoint_path}")


if __name__ == "__main__":
    _actor_lr = float(sys.argv[1])
    _output_dir = REPO_ROOT / "outputs" / "sil_lr_sweep" / f"actor_lr_{_actor_lr:.6g}"
    _output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(_actor_lr, _output_dir)
    except Exception:
        import traceback
        crash_path = _output_dir / "crash_traceback.txt"
        with open(crash_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! sil_lr_sweep_run CRASHED (actor_lr={_actor_lr}) - full traceback written to {crash_path}")
        raise
