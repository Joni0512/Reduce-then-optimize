"""
2026-10-17: buffer-based critic pretraining (user request, see chat): "Collect 10 episodes experience - store in buffer; Repeat: sample batch - train critic".
The given instances are played ONCE each with the (random-init) actor (mode="eval", no actor step), every window goes into a ReplayBuffer as (step, r_t, next_step)
with no critic update during the rollouts; then `steps` times a batch is drawn from the buffer and critic 1 (+ critic 2) is updated with target r_t (critic_td_update_step with
gamma=0, so the bootstrap term vanishes - same target as the old per-episode pretrain). No Polyak, no actor update. Measured after every step: MSE of both critics against r_t
on ALL buffer windows (critic_pretrain_log.json). Used by run_srl_training_loop(pretrain_mode="buffer"); the standalone test is srl_critic_pretrain_buffer_test.py.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.critic_update import critic_td_update_step
from rtv_solver.pipeline.replay_buffer import ReplayBuffer
from rtv_solver.structure.config import Config


def _buffer_mse(entries, critics) -> list[tuple[float, float]]:
    out = []
    with torch.no_grad():
        for c in critics:
            qs = [c(e.step.request_features, e.step.vehicle_features, e.step.edge_index).item() for e in entries]
            out.append((sum((q - e.reward) ** 2 for q, e in zip(qs, entries)) / len(entries), sum(qs) / len(qs)))
    return out


def run_buffer_pretrain(critic, critic_optimizer, critic2, critic_optimizer2, actor, instances: list[str], steps: int, batch: int, reward_mode: str,
                        batch_interval: int, step_size: int, seed: int, output_dir: Path, manifest_dir: Path, max_thread_cnt: int,
                        critic_clip_norm: float = 1.0, collect_epochs: int = 1, exploration_sigma: float | None = None) -> list[dict]:
    """Returns the per-step log (also written to output_dir/critic_pretrain_log.json)."""
    frozen1 = copy.deepcopy(critic)
    frozen2 = copy.deepcopy(critic2) if critic2 is not None else None  # only needed by the TD interface; gamma=0 makes the bootstrap term vanish
    buffer = ReplayBuffer(capacity=100000)  # 2026-10-26: large enough for collect_epochs x instances x windows (also at half horizon)
    for collect_epoch, instance in [(e, i) for e in range(collect_epochs) for i in instances]:  # 2026-10-26: collect_epochs passes over the instances (with exploration_sigma noise each pass differs)
        input_path = manifest_dir / f"{instance}.json"
        inst_dir = output_dir / "critic_pretrain_collect" / instance
        inst_dir.mkdir(parents=True, exist_ok=True)
        config = Config(MAX_THREAD_CNT=max_thread_cnt, OUTPUT_DIR=inst_dir, MODE="coaml", BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed, KEEP_ACTIVE=False)
        payload = PayloadParser.load_input_data(input_path)
        cleared = PayloadParser.clear_vehicle_manifests(payload)
        start = len(buffer)
        pipeline = COAMLPipeline(
            config, cleared, imitation_solution_path=input_path, model=actor,
            critic=critic, critic_optimizer=critic_optimizer, target_critic=frozen1, critic2=critic2, critic_optimizer2=critic_optimizer2, target_critic2=frozen2,
            replay_buffer=buffer, replay_batch_size=batch, critic_target_mode="td_bootstrap", gamma=0.0, defer_critic_updates=True, critic_clip_norm=critic_clip_norm,
            exploration_sigma=exploration_sigma,
        )
        pipeline.solve_pdptw(cleared, mode="eval", train_critic=True, reward_mode=reward_mode)
        print(f"[buffer pretrain] epoch {collect_epoch} collected {instance}: {len(buffer) - start} windows, sum r = {sum(e.reward for e in buffer.entries[start:]):.0f}")

    entries = list(buffer.entries)
    mean_target = sum(e.reward for e in entries) / len(entries)
    critics = [critic] + ([critic2] if critic2 is not None else [])
    log = []

    def _record(step: int, batch_loss) -> None:
        stats = _buffer_mse(entries, critics)
        rec = {"step": step, "batch_loss": batch_loss, "mse1": stats[0][0], "q1": stats[0][1], "target_mean": mean_target}
        if critic2 is not None:
            rec["mse2"], rec["q2"] = stats[1]
        log.append(rec)

    _record(0, None)
    for step in range(1, steps + 1):
        batch_loss = critic_td_update_step(buffer.sample(batch), critic, critic_optimizer, frozen1, 0.0, critic2=critic2, critic_optimizer2=critic_optimizer2,
                                           target_critic2=frozen2, max_norm=critic_clip_norm)
        _record(step, batch_loss)
        if step % 25 == 0 or step == steps:
            print(f"[buffer pretrain] step {step}/{steps}: MSE on all {len(entries)} windows critic1 {log[-1]['mse1']:.3f}" + (f" critic2 {log[-1]['mse2']:.3f}" if critic2 is not None else "")
                  + f" | mean Q {log[-1]['q1']:.2f} (mean r {mean_target:.2f})")
    with open(output_dir / "critic_pretrain_log.json", "w") as f:
        json.dump({"instances": instances, "windows": len(entries), "steps": steps, "batch": batch, "reward_mode": reward_mode, "log": log}, f, indent=1)
    return log
