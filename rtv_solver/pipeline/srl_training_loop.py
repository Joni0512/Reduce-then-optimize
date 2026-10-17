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
import json
import os
import random
import time
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
from rtv_solver.online_rtv_solver import ManifestConsistencyError  # 2026-10-01: caught in _per_instance_service_rates
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import StatsParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.replay_buffer import ReplayBuffer
from rtv_solver.pipeline.critic_update import critic_td_update_step  # 2026-10-12 (joint buffer)
from rtv_solver.pipeline.actor_transition_buffer import ActorTransitionBuffer  # 2026-10-06
from rtv_solver.pipeline.srl_train_val_test_split import (
    TRAIN_INSTANCES, VAL_INSTANCES, OVERFIT_CHECK_INSTANCES,
)
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline import select_feature_builder_class

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# 2026-10-10: multiprocessing pool size of the trip generation (Config.MAX_THREAD_CNT). Default 16 = unchanged behavior. The CPU-scaling benchmark
# (scripts/benchmark_cpu_scaling.py, serial_std, see chat) showed the pool should match the allocated CPUs: 4 CPUs + pool 4 was ~8 % faster than
# 16 CPUs + pool 16, while pool 16 on 4 CPUs was ~28 % slower. Set SRL_MAX_THREAD_CNT=4 together with --cpus-per-task=4.
_MAX_THREAD_CNT = int(os.environ.get("SRL_MAX_THREAD_CNT", "16"))
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
    keep_active: bool = False,
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
            MAX_THREAD_CNT=_MAX_THREAD_CNT, OUTPUT_DIR=inst_out_dir, MODE="coaml",
            BATCH_INTERVAL=config_template.BATCH_INTERVAL, STEP_SIZE=config_template.STEP_SIZE,
            SEED=config_template.SEED,
            # 2026-09-30: KEEP_ACTIVE only feeds _check_consistency_of_manifests' post-hoc
            # assertion (online_rtv_solver.py) - it never influences trip generation, scoring,
            # or the ILP assignment itself, so this cannot change results. Default True crashes
            # with ManifestConsistencyError on some seeds because COAML's own solve path was
            # never built to actually keep active requests (see coaml_pipeline.py's own TODO at
            # the "currently KEEP_ACTIVE does not work in COAML" comment) - explicit False here
            # just disables an assertion the code can't satisfy, see chat.
            # 2026-10-01 (see chat): keep_active param (default False = unchanged behavior for all existing
            # callers). Behavior-cloning scripts pass True so validation uses the same commitment rule as
            # training/RHO - KEEP_ACTIVE DOES change the ILP (active_req constraints + 100x penalty in
            # co_tripCostMinimization.py / co_scoreMaximization.py), contrary to the note above.
            KEEP_ACTIVE=keep_active,
        )
        pipeline = COAMLPipeline(config, cleared_payload, model=model, imitation_solution_path=input_path)
        try:
            driver_runs = pipeline.solve_pdptw(cleared_payload, mode="eval")
        except (InfeasibleAssignmentError, ManifestConsistencyError) as e:
            # 2026-10-01: ManifestConsistencyError added (only reachable with keep_active=True) so one
            # instance can't kill a multi-day cluster job; skipped like an infeasible ILP (see chat).
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


def _run_actor_update_phase(
    actor_buffer: ActorTransitionBuffer,
    model: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    batch_size: int,
    rng: random.Random,
    grad_norms: list[float] | None = None,
    max_norm: float = 1.0,  # 2026-10-15: actor clipping threshold (default = original 1.0)
) -> tuple[list[float], list[float]]:
    """
    2026-10-14: optional `grad_norms` list collects the actor gradient norm BEFORE clipping of every step (clip_grad_norm_ returns it) - pure logging, no effect on training.
    2026-10-06: update phase of the "collect, then update" actor scheme (see chat,
    project_srl_minibatch_update_plan). Random permutation of all buffered windows
    (each exactly once), batches of batch_size; per batch: loss per window recomputed
    with the CURRENT actor + target critics (compute_srl_actor_loss_for_transition),
    mean over the batch's windows, one clipped Adam step. Critics are read-only here.
    The mean is realized by accumulating loss/len(batch) per window (FenchelYoungLoss
    only supports one instance at a time), which equals the gradient of the batch mean.
    Returns (per-batch mean losses, actor weight norm after each step).
    """
    batch_losses: list[float] = []
    weight_norms: list[float] = []
    for batch in actor_buffer.epoch_batches(batch_size, rng):
        actor_optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for transition in batch:
            # all instance pipelines of a run share ONE actor object; re-pointing guards the
            # rare case where a pipeline was built before the shared model existed.
            transition.pipeline.model = model
            loss = transition.pipeline.compute_srl_actor_loss_for_transition(transition)
            (loss / len(batch)).backward()
            batch_loss += loss.item() / len(batch)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        if grad_norms is not None:
            grad_norms.append(float(gn))  # 2026-10-14: pre-clip norm (logging only)
        actor_optimizer.step()
        with torch.no_grad():
            weight_norms.append(torch.sqrt(sum(p.pow(2).sum() for p in model.parameters())).item())
        batch_losses.append(batch_loss)
    return batch_losses, weight_norms


def _log_grad_norms(epoch_num: int, grad_norms: list[float], log: list[dict], output_dir, reward_mode: str, clip: float = 1.0) -> None:
    """2026-10-14: per-epoch statistics of the actor gradient norm before clipping (max_norm=1.0): how often the clip is active. Printed and written to grad_norm_log.json."""
    if not grad_norms:
        return
    ordered = sorted(grad_norms)
    entry = {
        "epoch": epoch_num, "steps": len(ordered), "mean": sum(ordered) / len(ordered), "median": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], "max": ordered[-1],
        "share_above_1": sum(g > 1.0 for g in ordered) / len(ordered), "clip_norm": clip, "share_above_clip": sum(g > clip for g in ordered) / len(ordered),
    }
    log.append(entry)
    print(f"[srl_training_loop reward_mode={reward_mode}] epoch {epoch_num}: actor grad norm before clipping (n={entry['steps']}) "
          f"mean {entry['mean']:.3f} median {entry['median']:.3f} p90 {entry['p90']:.3f} max {entry['max']:.3f}, norm above the clip threshold {clip:g} in {entry['share_above_clip'] * 100:.0f}% of steps")
    with open(output_dir / "grad_norm_log.json", "w") as f:
        json.dump(log, f, indent=1)


def _record_critic_stats(log: list[dict], output_dir, phase: str, epoch: int, instance: str, pipeline, pipeline2=None) -> None:
    """2026-10-17: one record per episode (= one instance rollout) of the critic MSE (critic 1 / 2), mean live-critic prediction and mean TD target,
    read from COAMLPipeline.last_episode_critic_stats; written to critic_loss_log.json after every episode. pipeline2 (critic pretrain with a twin
    critic: the second critic runs in its own pipeline) supplies critic2_loss / q2_mean. Logging only."""
    stats = dict(getattr(pipeline, "last_episode_critic_stats", {}))
    if pipeline2 is not None:
        stats2 = getattr(pipeline2, "last_episode_critic_stats", {})
        stats["critic2_loss"] = stats2.get("critic_loss", float("nan"))
        stats["q2_mean"] = stats2.get("q_mean", float("nan"))
    log.append({"phase": phase, "epoch": epoch, "instance": instance, **stats})
    with open(output_dir / "critic_loss_log.json", "w") as f:
        json.dump(log, f, indent=1)


def _polyak_update(target: torch.nn.Module, live: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, live_param in zip(target.parameters(), live.parameters()):
            target_param.data.copy_(tau * live_param.data + (1 - tau) * target_param.data)


def _run_joint_update_phase(
    actor_buffer: ActorTransitionBuffer,
    replay_buffer: ReplayBuffer,
    model: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic, critic_optimizer, target_critic,
    critic2, critic_optimizer2, target_critic2,
    num_steps: int,
    actor_batch_size: int,
    critic_steps_per_actor_step: int,
    critic_batch_size: int,
    tau_per_step: float,
    gamma: float,
    rng: random.Random,
    grad_norms: list[float] | None = None,
    actor_clip_norm: float = 1.0,
    critic_clip_norm: float = 1.0,  # 2026-10-15: clipping thresholds (defaults = original 1.0)
) -> tuple[list[float], list[float], list[float]]:
    """
    2026-10-12: update phase of the "buffered_joint" scheme (see chat). num_steps times: (1) draw actor_batch_size windows WITH replacement
    from the shared multi-epoch actor buffer -> one actor step exactly like _run_actor_update_phase (loss per window recomputed with the
    CURRENT actor + target critics, mean over the batch, clip 1.0, Adam; critics read-only); (2) critic_steps_per_actor_step critic TD updates
    (critic_td_update_step) on batches of critic_batch_size entries from the replay buffer; (3) one Polyak step of the target critics with
    tau_per_step. Returns (actor batch losses, actor weight norms, critic batch losses).
    """
    actor_losses: list[float] = []
    weight_norms: list[float] = []
    critic_losses: list[float] = []
    for _ in range(num_steps):
        batch = actor_buffer.sample_with_replacement(actor_batch_size, rng)
        actor_optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for transition in batch:
            transition.pipeline.model = model
            loss = transition.pipeline.compute_srl_actor_loss_for_transition(transition)
            (loss / len(batch)).backward()
            batch_loss += loss.item() / len(batch)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=actor_clip_norm)
        if grad_norms is not None:
            grad_norms.append(float(gn))  # 2026-10-14: pre-clip norm (logging only)
        actor_optimizer.step()
        with torch.no_grad():
            weight_norms.append(torch.sqrt(sum(p.pow(2).sum() for p in model.parameters())).item())
        actor_losses.append(batch_loss)

        for _ in range(critic_steps_per_actor_step):
            entries = replay_buffer.sample(critic_batch_size)
            critic_losses.append(critic_td_update_step(
                entries, critic, critic_optimizer, target_critic, gamma,
                critic2=critic2, critic_optimizer2=critic_optimizer2, target_critic2=target_critic2, max_norm=critic_clip_norm,
            ))

        _polyak_update(target_critic, critic, tau_per_step)
        if target_critic2 is not None:
            _polyak_update(target_critic2, critic2, tau_per_step)
    return actor_losses, weight_norms, critic_losses


def run_srl_training_loop(
    reward_mode: str,
    actor_lr: float,
    critic_lr: float,
    output_dir: Path,
    actor_checkpoint: str,
    gamma: float = 0.99,
    tau: float = 0.005,  # 2026-09-24 fix: was 0.001, a copy-paste bug - the intended/validated
    # value is 0.005, matching run_srl_balanced_td_bootstrap_gat_bufferfix_12instances.py
    # (see chat), which this loop's own docstring already claimed to reuse "as-is".
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
    deterministic: bool = False,
    use_twin_critic: bool = False,
    use_actor_warmstart: bool = True,
    gat_num_heads: int = 4,
    num_message_passing_layers: int = 2,
    weight_norm_trace: list[dict] | None = None,
    actor_update_mode: str = "per_window",
    actor_batch_size: int = 16,
    actor_buffer_capacity: int | None = None,
    # 2026-10-12: actor architecture switch (user request: test a GNN actor in the buffered scheme, no SIL warmstart).
    # Defaults reproduce the old behavior exactly (MLP, hidden 64); only used when use_actor_warmstart=False.
    actor_model_type: str = "mlp",
    actor_gnn_layers: int = 2,
    actor_gnn_aggregator: str = "gcn",
    # 2026-10-12: parameters of actor_update_mode="buffered_joint" (ignored in the other modes): shared buffer over joint_buffer_epochs epochs,
    # actor AND critic update from it after every collection phase - see _run_joint_update_phase.
    joint_buffer_epochs: int = 10,
    joint_critic_steps_per_actor_step: int = 7,
    joint_critic_batch_size: int = 12,
    joint_tau_per_step: float = 0.0083,
    # 2026-10-13: Perturb-and-MAP noise sigma and number of candidates for the SRL actor loss (ablation sweep, see chat). Defaults = Config's own
    # defaults (SIGMA 0.2, NUM_SAMPLES 20), so all earlier runs are unchanged. Only the training Config of the main epoch loop needs them (pretrain/validation never sample).
    sigma: float = 0.2,
    num_samples: int = 20,
    # 2026-10-15: gradient-clipping thresholds of actor and critic updates (clip_grad_norm_ max_norm); defaults 1.0 = the original hard-coded value, so earlier runs are unchanged.
    actor_clip_norm: float = 1.0,
    critic_clip_norm: float = 1.0,
    # 2026-10-17: buffer-based critic pretraining (pretrain_mode="buffer", see srl_buffer_pretrain.py) and the per-episode scheme actor_update_mode="episodic"
    # (user request, see chat: "Do one episode, store in buffer. Drop oldest episode from buffer; repeat a few times: sample batch - train actor - train critic").
    # Defaults reproduce the old behavior exactly.
    pretrain_mode: str = "per_episode",
    pretrain_instances: list[str] | None = None,
    pretrain_steps: int = 150,
    pretrain_batch: int = 20,
    episodic_buffer_episodes: int = 10,
    episodic_actor_steps: int = 1,
    episodic_actor_batch: int = 20,
    episodic_critic_steps: int = 1,
    episodic_critic_batch: int = 20,
    episodic_tau: float = 0.005,
) -> SRLTrainingLoopResult:
    """
    2026-10-06: added `actor_update_mode` (see chat, "collect, then update"):
    - "per_window" (default): the ORIGINAL scheme, completely unchanged - one actor
      gradient step per rolling-horizon window while the instances are played.
    - "buffered": per epoch, phase A plays all train instances with the actor frozen
      (windows are stored in an ActorTransitionBuffer, no actor step; critic training/
      Polyak as before), phase B shuffles the stored windows and updates the actor in
      batches of `actor_batch_size` (mean loss per batch, one step per batch, critics
      read-only). `actor_buffer_capacity=None` clears the buffer after every epoch
      (exactly one epoch of windows); an int keeps the newest N windows across epochs
      (FIFO ring, not yet used in any experiment).

    2026-09-24: added `use_twin_critic` (see chat) - TD3-style second,
    independently-initialized CriticGNN (same aggregator as the first,
    matching TD3's own same-architecture-different-init design). When True,
    builds critic2/critic_optimizer2/target_critic2 alongside the existing
    critic, Polyak-updates target_critic2 on the exact same per-instance
    schedule/tau as target_critic, and passes all three through to
    COAMLPipeline (see its __init__ docstring for the min-target/mean-action
    semantics). False (default) is the original single-critic behavior,
    unchanged.

    2026-09-17: added `deterministic` (see chat) - was previously hardcoded
    to set_seed(seed, debug=False), meaning torch.use_deterministic_algorithms
    was NEVER enabled here despite a fixed seed. Investigating why identical
    actor_lr/critic_lr pairs across sweep trials sometimes succeed and
    sometimes collapse (GAT's attention uses scatter/segment ops that are a
    known non-determinism source without this flag) - pass deterministic=True
    for a controlled reproducibility check. NOT the config.DEBUG print-spam
    flag (see stats_parser.py's gating) - this only affects set_seed()'s
    torch determinism enforcement, independent of Config.DEBUG.

    2026-10-01: added `weight_norm_trace` (see chat) - if given a list, every main-loop
    instance's pipeline.weight_norm_history (one actor weight L2 norm per rolling-horizon
    gradient step, see coaml_pipeline.py) is appended to it as
    {"epoch": epoch_num, "instance": instance, "norms": [...]}, for analyzing how far/fast the
    actor moves per step (e.g. comparing a stable vs. a collapsing critic_lr run). None
    (default) skips this - no behavior change for existing callers.
    """
    if reward_mode not in ("local", "local_positive", "local_served"):  # 2026-10-17: + local_served
        raise ValueError(f"Expected reward_mode in ('local', 'local_positive', 'local_served'), got {reward_mode!r}")
    if actor_update_mode not in ("per_window", "buffered", "buffered_joint", "episodic"):
        raise ValueError(f"Expected actor_update_mode in ('per_window', 'buffered', 'buffered_joint', 'episodic'), got {actor_update_mode!r}")
    if pretrain_mode not in ("per_episode", "buffer"):
        raise ValueError(f"Expected pretrain_mode in ('per_episode', 'buffer'), got {pretrain_mode!r}")
    buffered = actor_update_mode in ("buffered", "buffered_joint", "episodic")  # 2026-10-12: joint also collects with a frozen actor; 2026-10-17: so does episodic (per episode)
    joint = actor_update_mode == "buffered_joint"
    episodic = actor_update_mode == "episodic"  # 2026-10-17: windows of the last episodic_buffer_episodes episodes, update after EVERY episode
    defer_critic = joint or episodic  # critic is trained in the update step, not inside the rollout

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_loggers(output_dir)
    set_seed(seed, debug=deterministic)
    rng = random.Random(seed)

    # --- critic pretraining, ONCE, before the epoch loop ---
    critic = CriticGNN(aggregator=gnn_aggregator, gat_num_heads=gat_num_heads, num_message_passing_layers=num_message_passing_layers)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    # 2026-09-24: twin critic (see chat) - critic2 gets the SAME pretraining
    # loop as critic, run separately below, so both critics enter the main
    # epoch loop with equal pretrain exposure (only independent init/gradient
    # noise differs) - pretraining just critic and not critic2 would give
    # critic1 an unfair head start beyond initialization.
    critic2 = CriticGNN(aggregator=gnn_aggregator, gat_num_heads=gat_num_heads, num_message_passing_layers=num_message_passing_layers) if use_twin_critic else None
    critic_optimizer2 = torch.optim.Adam(critic2.parameters(), lr=critic_lr) if use_twin_critic else None

    # use_actor_warmstart=False: build ONE randomly-initialized actor up front and reuse it
    # for every pretrain/training call (instead of loading actor_checkpoint each time) - see
    # chat, matches the reference SRL paper's "same random model initialization for SIL, PPO,
    # and SRL" (no SIL-pretrained warmstart). Built once here so it stays the SAME random
    # weights throughout (a fresh COAMLPipeline call with model=None would otherwise
    # reinitialize randomly on every single call, which is not equivalent to one consistent
    # random init).
    pretrain_actor_model = None
    if not use_actor_warmstart:
        config_template = Config(MAX_THREAD_CNT=_MAX_THREAD_CNT, OUTPUT_DIR=output_dir, BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed, KEEP_ACTIVE=False)
        active_feature_builder = select_feature_builder_class(config_template)
        # 2026-10-12: model type/layers/aggregator now parameters (defaults = old "mlp", hidden 64); "gnn" gets the candidate
        # conflict graph automatically via COAMLPipeline._score (model.requires_graph), also in the buffered update phase.
        pretrain_actor_model = build_scoring_model(actor_model_type, feature_dim=active_feature_builder.FEATURE_SIZE, hidden_dim=64,
                                                   num_message_passing_layers=actor_gnn_layers, aggregator=actor_gnn_aggregator)

    pretrain_dir = output_dir / "critic_pretrain"
    critic_loss_log: list[dict] = []  # 2026-10-17: per-episode critic loss / Q / target statistics (pretrain + main phase), see _record_critic_stats
    if pretrain_mode == "buffer":
        # 2026-10-17: collect the pretrain instances once into a buffer, then pretrain the critics from it (see srl_buffer_pretrain.py); replaces the per-episode pretrain
        if use_actor_warmstart:
            raise ValueError("pretrain_mode='buffer' is only implemented for use_actor_warmstart=False (random-init actor)")
        from rtv_solver.pipeline.srl_buffer_pretrain import run_buffer_pretrain
        run_buffer_pretrain(
            critic, critic_optimizer, critic2, critic_optimizer2, pretrain_actor_model, list(pretrain_instances or []), pretrain_steps, pretrain_batch, reward_mode,
            batch_interval, step_size, seed, output_dir, MANIFEST_DIR, _MAX_THREAD_CNT, critic_clip_norm=critic_clip_norm,
        )
    for epoch in range(critic_pretrain_epochs if pretrain_mode == "per_episode" else 0):
        for instance in TRAIN_INSTANCES:
            input_path = MANIFEST_DIR / f"{instance}.json"
            inst_out_dir = pretrain_dir / instance
            inst_out_dir.mkdir(parents=True, exist_ok=True)
            # KEEP_ACTIVE=False: see the KEEP_ACTIVE comment in _per_instance_service_rates
            # above - only disables a post-hoc assertion, does not affect results.
            config = Config(MAX_THREAD_CNT=_MAX_THREAD_CNT, OUTPUT_DIR=inst_out_dir, MODE="coaml", BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed, KEEP_ACTIVE=False)
            payload = PayloadParser.load_input_data(input_path)
            cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
            pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path, critic=critic, critic_optimizer=critic_optimizer, model=pretrain_actor_model, critic_clip_norm=critic_clip_norm)
            if use_actor_warmstart:
                pipeline.load_model_weights(actor_checkpoint)
            pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=True, reward_mode=reward_mode)

            if use_twin_critic:
                inst_out_dir2 = pretrain_dir / f"{instance}_critic2"
                inst_out_dir2.mkdir(parents=True, exist_ok=True)
                config2 = Config(MAX_THREAD_CNT=_MAX_THREAD_CNT, OUTPUT_DIR=inst_out_dir2, MODE="coaml", BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed, KEEP_ACTIVE=False)
                pipeline2 = COAMLPipeline(config2, cleared_payload, imitation_solution_path=input_path, critic=critic2, critic_optimizer=critic_optimizer2, model=pretrain_actor_model, critic_clip_norm=critic_clip_norm)
                if use_actor_warmstart:
                    pipeline2.load_model_weights(actor_checkpoint)
                pipeline2.solve_pdptw(cleared_payload, mode="eval", train_critic=True, reward_mode=reward_mode)
            _record_critic_stats(critic_loss_log, output_dir, "pretrain", epoch, instance, pipeline, pipeline2 if use_twin_critic else None)  # 2026-10-17
        _pre = [r["critic_loss"] for r in critic_loss_log if r["phase"] == "pretrain" and r["epoch"] == epoch]
        print(f"[srl_training_loop reward_mode={reward_mode}] critic pretrain epoch {epoch} done, mean critic MSE per episode {sum(_pre) / max(len(_pre), 1):.4f}")

    # --- shared actor/critic/replay-buffer state, persists across epochs and instances ---
    # model=None (default): first pipeline() call below loads actor_checkpoint and creates a
    # fresh model. use_actor_warmstart=False: reuse the SAME random-init actor that was already
    # used throughout critic pretraining above, so training continues from those exact weights
    # instead of loading a checkpoint.
    model = pretrain_actor_model if not use_actor_warmstart else None
    actor_optimizer = None
    target_critic = copy.deepcopy(critic)
    # 2026-09-24: twin critic (see chat) - target_critic2 mirrors target_critic,
    # copied from the now-pretrained critic2.
    target_critic2 = copy.deepcopy(critic2) if use_twin_critic else None
    # 2026-10-12: joint mode keeps ~joint_buffer_epochs epochs of critic transitions (~370 windows per epoch -> 400 as upper estimate)
    grad_norm_log: list[dict] = []  # 2026-10-14: per-epoch actor gradient-norm statistics (buffered / buffered_joint modes)
    replay_buffer = ReplayBuffer(capacity=2000 if episodic else (joint_buffer_epochs * 400 if joint else replay_capacity))  # 2026-10-17: episodic never evicts by capacity (episodes are dropped explicitly)
    actor_buffer = ActorTransitionBuffer(capacity=actor_buffer_capacity, compress=joint or episodic) if buffered else None  # 2026-10-06; compress: 2026-10-12 (+ episodic 2026-10-17: needs the episode tag)
    episode_counter = 0  # 2026-10-17: episodic mode - global episode index (1, 2, ... over all epochs), used as the FIFO tag of both buffers

    best_val_service_rate = -1.0
    best_epoch = -1
    best_checkpoint_path = output_dir / "best_actor_checkpoint.pt"
    val_curve: list[dict] = []
    overfit_curve: list[dict] = []

    for epoch in range(epochs):
        epoch_num = epoch + 1
        shuffled_train_instances = TRAIN_INSTANCES.copy()
        if joint:
            actor_buffer.current_epoch = epoch_num  # 2026-10-12: tag for drop_epochs_before
        episodic_epoch_grad_norms: list[float] = []  # 2026-10-17
        rng.shuffle(shuffled_train_instances)
        collect_start = time.perf_counter()

        for instance in shuffled_train_instances:
            input_path = MANIFEST_DIR / f"{instance}.json"
            inst_out_dir = output_dir / "train" / f"epoch_{epoch_num}" / instance
            inst_out_dir.mkdir(parents=True, exist_ok=True)
            config = Config(MAX_THREAD_CNT=_MAX_THREAD_CNT, OUTPUT_DIR=inst_out_dir, MODE="coaml", BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed, MAX_CARDINALITY=max_cardinality, KEEP_ACTIVE=False, SIGMA=sigma, NUM_SAMPLES=num_samples)  # 2026-10-13: sigma/num_samples (ablation)
            payload = PayloadParser.load_input_data(input_path)
            cleared_payload = PayloadParser.clear_vehicle_manifests(payload)

            # 2026-09-10 Polyak target_critic blend (tau fixed, see srl deck) -
            # every instance step, same formula as train_srl_single_instance.py.
            # 2026-10-12: joint mode does its Polyak steps inside _run_joint_update_phase instead (tau per update step).
            with torch.no_grad():
                for target_param, live_param in zip(target_critic.parameters(), critic.parameters()):
                    if not (joint or episodic):  # 2026-10-17: episodic does its Polyak steps inside the update step
                        target_param.data.copy_(tau * live_param.data + (1 - tau) * target_param.data)

            # 2026-09-24: twin critic (see chat) - target_critic2 blended on the
            # exact same schedule/tau as target_critic above.
            if use_twin_critic and not (joint or episodic):
                with torch.no_grad():
                    for target_param, live_param in zip(target_critic2.parameters(), critic2.parameters()):
                        target_param.data.copy_(tau * live_param.data + (1 - tau) * target_param.data)

            if episodic:
                episode_counter += 1
                actor_buffer.current_epoch = episode_counter  # 2026-10-17: episode tag (drop_epochs_before is used with episode indices here)
                replay_buffer.current_episode = episode_counter
            pipeline = COAMLPipeline(
                config, cleared_payload, imitation_solution_path=input_path,
                model=model, critic=critic, critic_optimizer=critic_optimizer,
                target_critic=target_critic,
                critic2=critic2, critic_optimizer2=critic_optimizer2, target_critic2=target_critic2,
                replay_buffer=replay_buffer, replay_batch_size=replay_batch_size,
                replay_update_group_size=replay_update_group_size,
                critic_target_mode="td_bootstrap", gamma=gamma,
                actor_transition_buffer=actor_buffer,  # 2026-10-06: None in per_window mode = old behavior
                defer_critic_updates=defer_critic,  # 2026-10-12: joint mode trains the critic in the update phase; 2026-10-17: also episodic
                actor_clip_norm=actor_clip_norm, critic_clip_norm=critic_clip_norm,  # 2026-10-15
            )
            if model is None:
                pipeline.load_model_weights(actor_checkpoint)
            if actor_optimizer is None:
                actor_optimizer = torch.optim.Adam(pipeline.model.parameters(), lr=actor_lr)

            try:
                # 2026-10-06: buffered mode passes optimizer=None - the actor must not step during the
                # rollout (windows are only stored); per_window passes the real optimizer as before.
                pipeline.solve_pdptw(cleared_payload, mode="srl", optimizer=None if buffered else actor_optimizer, train_critic=True, reward_mode=reward_mode)
            except InfeasibleAssignmentError as e:
                # 2026-09-15: see InfeasibleAssignmentError's docstring
                # (co_base.py) - a structural trip-generation gap, not a bug
                # to fix per-occurrence. Skip this one instance for this
                # epoch rather than losing the whole multi-hour training run
                # to a rare, instance-specific ILP conflict.
                print(f"[srl_training_loop reward_mode={reward_mode}] epoch {epoch_num}: SKIPPING {instance} - {e}")
                continue
            model = pipeline.model  # carry actor weights forward
            _record_critic_stats(critic_loss_log, output_dir, "train", epoch_num, instance, pipeline)  # 2026-10-17
            if episodic:
                # 2026-10-17: Do one episode, store in buffer; drop the oldest episode; then episodic_actor_steps x (actor step + critic steps + Polyak) from the buffers
                first_kept = episode_counter - episodic_buffer_episodes + 1
                actor_buffer.drop_epochs_before(first_kept)
                replay_buffer.drop_episodes_before(first_kept)
                if len(actor_buffer) > 0 and len(replay_buffer) > 0:
                    ep_actor_losses, _ep_norms, ep_critic_losses = _run_joint_update_phase(
                        actor_buffer, replay_buffer, model, actor_optimizer,
                        critic, critic_optimizer, target_critic, critic2, critic_optimizer2, target_critic2,
                        episodic_actor_steps, episodic_actor_batch, episodic_critic_steps, episodic_critic_batch,
                        episodic_tau, gamma, rng, grad_norms=episodic_epoch_grad_norms, actor_clip_norm=actor_clip_norm, critic_clip_norm=critic_clip_norm,
                    )
                    critic_loss_log[-1]["critic_loss"] = sum(ep_critic_losses) / len(ep_critic_losses)  # the rollout itself did no critic update (deferred): record the update-step loss
                    critic_loss_log[-1]["actor_loss"] = sum(ep_actor_losses) / len(ep_actor_losses)
                    critic_loss_log[-1]["buffer_windows"] = len(actor_buffer)
                    with open(output_dir / "critic_loss_log.json", "w") as f:
                        json.dump(critic_loss_log, f, indent=1)
            if weight_norm_trace is not None:
                weight_norm_trace.append({"epoch": epoch_num, "instance": instance, "norms": pipeline.weight_norm_history})

        if episodic:
            _log_grad_norms(epoch_num, episodic_epoch_grad_norms, grad_norm_log, output_dir, reward_mode, actor_clip_norm)  # 2026-10-17: per-epoch actor gradient statistics
        if buffered and not episodic:
            # 2026-10-06: phase B of "collect, then update". Partial instances (skipped after an
            # InfeasibleAssignmentError) keep the windows stored before the failure.
            collect_s = time.perf_counter() - collect_start
            update_start = time.perf_counter()
            num_windows = len(actor_buffer)
            if joint and num_windows > 0 and model is not None:
                # 2026-10-12: joint scheme - same number of actor steps per epoch as "buffered" (windows of THIS epoch / batch size), but
                # sampled with replacement from the multi-epoch buffer, plus critic updates and Polyak per step.
                epoch_windows = sum(1 for it in actor_buffer._items if it[1] == epoch_num)
                num_steps = max(1, -(-epoch_windows // actor_batch_size))
                epoch_grad_norms: list[float] = []
                batch_losses, weight_norms, critic_losses = _run_joint_update_phase(
                    actor_buffer, replay_buffer, model, actor_optimizer,
                    critic, critic_optimizer, target_critic, critic2, critic_optimizer2, target_critic2,
                    num_steps, actor_batch_size, joint_critic_steps_per_actor_step, joint_critic_batch_size,
                    joint_tau_per_step, gamma, rng, grad_norms=epoch_grad_norms, actor_clip_norm=actor_clip_norm, critic_clip_norm=critic_clip_norm,
                )
                _log_grad_norms(epoch_num, epoch_grad_norms, grad_norm_log, output_dir, reward_mode, actor_clip_norm)
                if weight_norm_trace is not None:
                    weight_norm_trace.append({"epoch": epoch_num, "instance": "joint_update_phase", "norms": weight_norms})
                print(
                    f"[srl_training_loop reward_mode={reward_mode}] epoch {epoch_num}: joint update phase - "
                    f"buffer {num_windows} windows (this epoch {epoch_windows}), {num_steps} steps x (1 actor batch of {actor_batch_size} + "
                    f"{joint_critic_steps_per_actor_step} critic batches of {joint_critic_batch_size}), "
                    f"mean actor loss {sum(batch_losses) / len(batch_losses):.4f}, mean critic loss {sum(critic_losses) / max(len(critic_losses), 1):.4f}, "
                    f"collect {collect_s:.0f}s, update {time.perf_counter() - update_start:.0f}s"
                )
            elif num_windows > 0 and model is not None:
                epoch_grad_norms = []
                batch_losses, weight_norms = _run_actor_update_phase(actor_buffer, model, actor_optimizer, actor_batch_size, rng, grad_norms=epoch_grad_norms, max_norm=actor_clip_norm)
                _log_grad_norms(epoch_num, epoch_grad_norms, grad_norm_log, output_dir, reward_mode, actor_clip_norm)
                if weight_norm_trace is not None:
                    weight_norm_trace.append({"epoch": epoch_num, "instance": "actor_update_phase", "norms": weight_norms})
                print(
                    f"[srl_training_loop reward_mode={reward_mode}] epoch {epoch_num}: actor update phase - "
                    f"{num_windows} windows, {len(batch_losses)} batches of <= {actor_batch_size}, "
                    f"mean batch loss {sum(batch_losses) / len(batch_losses):.4f}, "
                    f"collect {collect_s:.0f}s, update {time.perf_counter() - update_start:.0f}s"
                )
            if joint:
                dropped = actor_buffer.drop_epochs_before(epoch_num - joint_buffer_epochs + 1)  # 2026-10-12: keep the newest joint_buffer_epochs epochs
                if dropped:
                    print(f"[srl_training_loop reward_mode={reward_mode}] epoch {epoch_num}: dropped {dropped} windows older than {joint_buffer_epochs} epochs")
            elif actor_buffer_capacity is None:
                actor_buffer.clear()  # exactly one epoch of windows, see run_srl_training_loop's docstring

        _tr = [r["critic_loss"] for r in critic_loss_log if r["phase"] == "train" and r["epoch"] == epoch_num and r["critic_loss"] == r["critic_loss"]]  # NaN-free (joint mode defers)
        print(f"[srl_training_loop reward_mode={reward_mode}] epoch {epoch_num}/{epochs} training done, mean critic MSE per episode {sum(_tr) / max(len(_tr), 1):.4f} (n={len(_tr)})")

        if epoch_num % val_every_n_epochs == 0 or epoch_num == epochs:
            config_template = Config(MAX_THREAD_CNT=_MAX_THREAD_CNT, OUTPUT_DIR=output_dir, BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed, KEEP_ACTIVE=False)
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

            # 2026-10-07: partial results after EVERY validation (see chat, wandb sweep) - result.json is only
            # written by the trial scripts after the whole run, so a crash/24h timeout at epoch 18 used to lose
            # the already measured val curve. Extra file only, no influence on training.
            with open(output_dir / "partial_val_curve.json", "w") as f:
                json.dump({
                    "last_epoch": epoch_num, "epochs": epochs,
                    "best_epoch": best_epoch, "best_val_service_rate": best_val_service_rate,
                    "val_curve": val_curve, "overfit_curve": overfit_curve,
                }, f, indent=2)

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
    # 2026-09-24: instances can be missing from a given epoch's per_instance dict
    # (InfeasibleAssignmentError skip, see 01e8c80) - direct [inst] lookups here
    # crashed live sweep trials as recently as 2026-09-22 (KeyError 'lc202'/'lc204',
    # see chat). Only plot the (epoch, rate) points where that instance is present.
    for inst in VAL_INSTANCES:
        xs_i = [r["epoch"] for r in val_curve if inst in r["per_instance"]]
        ys_i = [r["per_instance"][inst] for r in val_curve if inst in r["per_instance"]]
        ax.plot(xs_i, ys_i, color="tab:blue", alpha=0.25, linewidth=1)
    for inst in OVERFIT_CHECK_INSTANCES:
        xs_i = [r["epoch"] for r in overfit_curve if inst in r["per_instance"]]
        ys_i = [r["per_instance"][inst] for r in overfit_curve if inst in r["per_instance"]]
        ax.plot(xs_i, ys_i, color="tab:orange", alpha=0.25, linewidth=1)
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
