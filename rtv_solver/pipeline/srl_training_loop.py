"""
2026-09-13: multi-instance SRL training loop with a real train/val split -
see chat. Generalizes train_srl_single_instance.py's single-instance episode
loop (actor+critic co-adapting, replay buffer, TD-bootstrap target) to the
new stratified 38/9/9 Li&Lim split (srl_train_val_test_split.py):

- One "episode" here = one full pass (in shuffled order) over all 38 TRAIN
  instances, i.e. what SIL's training_loop.py calls an "epoch". Actor,
  critic, critic_optimizer, actor_optimizer, and replay_buffer all persist
  across instances AND across epochs - never reset mid-run.
- Critic is pretrained ONCE before the epoch loop (not refreshed), same as
  the existing 12-instance scripts.
- Every VAL_EVERY_N_EPOCHS epochs: run mode="eval" (no exploration, no
  gradient step) on the 9 VAL_INSTANCES AND the 9 OVERFIT_CHECK_INSTANCES
  (a fixed subset of TRAIN_INSTANCES). Only VAL_INSTANCES' pooled service
  rate decides the best checkpoint; OVERFIT_CHECK_INSTANCES is logged
  purely so the resulting plot can show train-fit vs. val-fit over time.
- Best-val checkpoint is saved AND reloaded into the live model before
  returning (training_loop.py's SIL loop was found to skip this reload -
  see code review finding #2, fixed here from the start).

TD-bootstrap + replay buffer + GAT critic is the current default setup in
the codebase (see run_srl_balanced_td_bootstrap_gat_bufferfix_12instances.py
and friends) - reused here as-is; reward_mode ("local" vs "local_positive")
is the one axis this loop is built to compare, with actor_lr/critic_lr as
the swept hyperparameters (gamma, tau fixed - see chat/srl deck).
"""
from __future__ import annotations

import copy
import csv
import random
from dataclasses import dataclass
from pathlib import Path

# 2026-09-15: force the non-interactive Agg backend before importing pyplot -
# sweep trials run outside the main thread, and macOS's default GUI backend
# raises RuntimeError there, crashing every trial right after training
# finishes (at the plot-save step, before the result reaches wandb).
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
from rtv_solver.handlers.stats_parser import StatsParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.replay_buffer import ReplayBuffer
from rtv_solver.pipeline.srl_train_val_test_split import (
    TRAIN_INSTANCES, VAL_INSTANCES, OVERFIT_CHECK_INSTANCES,
)
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = REPO_ROOT / "solutions" / "li_lim" / "manifests"


@dataclass
class SRLTrainingLoopResult:
    best_epoch: int
    best_val_service_rate: float
    best_checkpoint_path: Path
    val_curve: list[dict]
    overfit_curve: list[dict]


def _instance_service_rate(config: Config, cleared_payload: dict, driver_runs: list) -> float:
    full_payload_object = PayloadParser.get_payload_object(
        cleared_payload, dwell_pickup_default=config.DWELL_PICKUP, dwell_alight_default=config.DWELL_ALIGHT, online=False,
    )
    all_requests = RequestHandler(full_payload_object.requests, config=config).get_all_requests()
    stats_payload = {
        PayloadKeys.DEPOT: cleared_payload[PayloadKeys.DEPOT],
        PayloadKeys.REQUESTS: cleared_payload[PayloadKeys.REQUESTS],
        PayloadKeys.DRIVERS: driver_runs,
        PayloadKeys.TIME_MATRIX: cleared_payload.get(PayloadKeys.TIME_MATRIX, None),
    }
    _, episode_stats, _ = StatsParser(config, payload=stats_payload).evaluate(stats_payload)
    num_requests = len(all_requests)
    num_serviced = len(episode_stats.serviced_requests)
    return num_serviced / num_requests if num_requests > 0 else 0.0


def _per_instance_service_rates(
    instances: list[str], model: torch.nn.Module, config_template: Config,
    output_dir: Path, epoch_num: int, tag: str,
) -> dict[str, float]:
    """Returns {instance: service_rate} - caller decides how to aggregate."""
    rates: dict[str, float] = {}
    for instance in instances:
        input_path = MANIFEST_DIR / f"{instance}.json"
        payload = PayloadParser.load_input_data(input_path)
        cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
        inst_out_dir = output_dir / tag / f"epoch_{epoch_num}" / instance
        inst_out_dir.mkdir(parents=True, exist_ok=True)
        config = Config(
            OUTPUT_DIR=inst_out_dir, MODE="coaml",
            BATCH_INTERVAL=config_template.BATCH_INTERVAL, STEP_SIZE=config_template.STEP_SIZE,
            SEED=config_template.SEED,
        )
        pipeline = COAMLPipeline(config, cleared_payload, model=model, imitation_solution_path=input_path)
        try:
            driver_runs = pipeline.solve_pdptw(cleared_payload, mode="eval")
        except InfeasibleAssignmentError as e:
            # 2026-09-15: see InfeasibleAssignmentError's docstring (co_base.py) -
            # skip this instance's validation rate rather than crashing the
            # whole epoch loop over one rare, instance-specific ILP conflict.
            print(f"[srl_training_loop] {tag} epoch {epoch_num}: SKIPPING {instance} - {e}")
            continue
        rates[instance] = _instance_service_rate(config, cleared_payload, driver_runs)
    return rates


def _pooled_service_rate(
    instances: list[str], model: torch.nn.Module, config_template: Config,
    output_dir: Path, epoch_num: int, tag: str,
) -> float:
    """Kept for srl_cluster_trial.py's single-point test-set evaluation (no per-instance curve needed there)."""
    rates = _per_instance_service_rates(instances, model, config_template, output_dir, epoch_num, tag)
    return sum(rates.values()) / len(rates)


def run_srl_training_loop(
    reward_mode: str,
    actor_lr: float,
    critic_lr: float,
    output_dir: Path,
    actor_checkpoint: str,
    gamma: float = 0.99,
    tau: float = 0.001,
    epochs: int = 20,
    val_every_n_epochs: int = 5,
    critic_pretrain_epochs: int = 10,
    batch_interval: int = 200,
    step_size: int = 100,
    seed: int = 42,
    gnn_aggregator: str = "gat",
    replay_capacity: int = 40,
    replay_batch_size: int = 12,
    replay_update_group_size: int = 3,
    max_cardinality: int = 2,
) -> SRLTrainingLoopResult:
    if reward_mode not in ("local", "local_positive"):
        raise ValueError(f"Expected reward_mode in ('local', 'local_positive'), got {reward_mode!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_loggers(output_dir)
    set_seed(seed, debug=False)
    rng = random.Random(seed)

    # --- critic pretraining, ONCE, before the epoch loop ---
    critic = CriticGNN(aggregator=gnn_aggregator)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    pretrain_dir = output_dir / "critic_pretrain"
    for epoch in range(critic_pretrain_epochs):
        for instance in TRAIN_INSTANCES:
            input_path = MANIFEST_DIR / f"{instance}.json"
            inst_out_dir = pretrain_dir / instance
            inst_out_dir.mkdir(parents=True, exist_ok=True)
            config = Config(OUTPUT_DIR=inst_out_dir, MODE="coaml", BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed)
            payload = PayloadParser.load_input_data(input_path)
            cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
            pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path, critic=critic, critic_optimizer=critic_optimizer)
            pipeline.load_model_weights(actor_checkpoint)
            pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=True, reward_mode=reward_mode)
        print(f"[srl_training_loop reward_mode={reward_mode}] critic pretrain epoch {epoch} done")

    # --- shared actor/critic/replay-buffer state, persists across epochs and instances ---
    model = None  # first pipeline() call below loads actor_checkpoint and creates a fresh model
    actor_optimizer = None
    target_critic = copy.deepcopy(critic)
    replay_buffer = ReplayBuffer(capacity=replay_capacity)

    best_val_service_rate = -1.0
    best_epoch = -1
    best_checkpoint_path = output_dir / "best_actor_checkpoint.pt"
    val_curve: list[dict] = []
    overfit_curve: list[dict] = []

    for epoch in range(epochs):
        epoch_num = epoch + 1
        shuffled_train_instances = TRAIN_INSTANCES.copy()
        rng.shuffle(shuffled_train_instances)

        for instance in shuffled_train_instances:
            input_path = MANIFEST_DIR / f"{instance}.json"
            inst_out_dir = output_dir / "train" / f"epoch_{epoch_num}" / instance
            inst_out_dir.mkdir(parents=True, exist_ok=True)
            config = Config(OUTPUT_DIR=inst_out_dir, MODE="coaml", BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed, MAX_CARDINALITY=max_cardinality)
            payload = PayloadParser.load_input_data(input_path)
            cleared_payload = PayloadParser.clear_vehicle_manifests(payload)

            # 2026-09-10 Polyak target_critic blend (tau fixed, see srl deck) -
            # every instance step, same formula as train_srl_single_instance.py.
            with torch.no_grad():
                for target_param, live_param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * live_param.data + (1 - tau) * target_param.data)

            pipeline = COAMLPipeline(
                config, cleared_payload, imitation_solution_path=input_path,
                model=model, critic=critic, critic_optimizer=critic_optimizer,
                target_critic=target_critic,
                replay_buffer=replay_buffer, replay_batch_size=replay_batch_size,
                replay_update_group_size=replay_update_group_size,
                critic_target_mode="td_bootstrap", gamma=gamma,
            )
            if model is None:
                pipeline.load_model_weights(actor_checkpoint)
            if actor_optimizer is None:
                actor_optimizer = torch.optim.Adam(pipeline.model.parameters(), lr=actor_lr)

            try:
                pipeline.solve_pdptw(cleared_payload, mode="srl", optimizer=actor_optimizer, train_critic=True, reward_mode=reward_mode)
            except InfeasibleAssignmentError as e:
                # 2026-09-15: see InfeasibleAssignmentError's docstring
                # (co_base.py) - a structural trip-generation gap, not a bug
                # to fix per-occurrence. Skip this one instance for this
                # epoch rather than losing the whole multi-hour training run
                # to a rare, instance-specific ILP conflict.
                print(f"[srl_training_loop reward_mode={reward_mode}] epoch {epoch_num}: SKIPPING {instance} - {e}")
                continue
            model = pipeline.model  # carry actor weights forward

        print(f"[srl_training_loop reward_mode={reward_mode}] epoch {epoch_num}/{epochs} training done")

        if epoch_num % val_every_n_epochs == 0 or epoch_num == epochs:
            config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed)
            val_rates = _per_instance_service_rates(VAL_INSTANCES, model, config_template, output_dir, epoch_num, tag="val")
            overfit_rates = _per_instance_service_rates(OVERFIT_CHECK_INSTANCES, model, config_template, output_dir, epoch_num, tag="overfit_check")
            # max(..., 1) guards against every instance in a round hitting
            # InfeasibleAssignmentError and being skipped (see above) - an
            # empty dict would otherwise ZeroDivisionError here.
            val_rate = sum(val_rates.values()) / max(len(val_rates), 1)
            overfit_rate = sum(overfit_rates.values()) / max(len(overfit_rates), 1)
            val_curve.append({"epoch": epoch_num, "service_rate": val_rate, "per_instance": val_rates})
            overfit_curve.append({"epoch": epoch_num, "service_rate": overfit_rate, "per_instance": overfit_rates})
            print(f"[srl_training_loop reward_mode={reward_mode}] epoch {epoch_num}: val={val_rate:.4f} overfit_check={overfit_rate:.4f}")

            if val_rate > best_val_service_rate:
                best_val_service_rate = val_rate
                best_epoch = epoch_num
                torch.save({"model_state_dict": model.state_dict()}, best_checkpoint_path)

    # 2026-09-13: reload the best-val checkpoint into the live model before
    # returning - training_loop.py's SIL loop was found to skip this (code
    # review finding #2), fixed here from the start.
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    # 2026-09-13: CSV now also carries every individual instance's service
    # rate per checkpoint epoch (not just the pooled average) - see chat.
    csv_path = output_dir / "srl_train_val_curves.csv"
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

    # 2026-09-13: plot shows (1) every individual val/overfit-check instance
    # as a thin line, (2) the pooled averages as bold lines, and (3) a
    # vertical marker at the epoch whose checkpoint was actually selected as
    # "best" (best_checkpoint_path) - see chat.
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
    ax.set_title(f"SRL training (reward_mode={reward_mode}) - individual instances (thin) + averages (bold)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # 2026-09-14: PNG for quick viewing/slides, PDF (vector) for the thesis
    # LaTeX document - see .claude/skills/experiment-results-export.
    fig.savefig(output_dir / "srl_train_val_curves.png", dpi=150)
    fig.savefig(output_dir / "srl_train_val_curves.pdf")
    plt.close(fig)

    print(f"[srl_training_loop reward_mode={reward_mode}] best val service rate = {best_val_service_rate:.4f} at epoch {best_epoch} -> {best_checkpoint_path}")

    return SRLTrainingLoopResult(
        best_epoch=best_epoch,
        best_val_service_rate=best_val_service_rate,
        best_checkpoint_path=best_checkpoint_path,
        val_curve=val_curve,
        overfit_curve=overfit_curve,
    )
