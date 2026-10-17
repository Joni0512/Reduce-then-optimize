"""
2026-10-17: standalone test of the buffer-based critic pretraining (user request, see chat; text of the procedure: "Collect 10 episodes experience - store in
buffer; Repeat: sample batch - train critic (pre-training)"). ONLY the pretraining is run and measured, the main training comes later.

1. Collect: PRETRAIN_INSTANCES (10 instances, every one of the six types lc1/lc2/lr1/lr2/lrc1/lrc2 at least once) are played ONCE each with the random-init
   actor (mode="eval", no actor step), all windows go into a ReplayBuffer with their raw (step, r_t, next_step); no critic update during the rollouts
   (defer_critic_updates=True).
2. Train: STEPS times draw a batch of BATCH windows from the buffer (an independent draw per step, i.e. WITH replacement across steps) and update critic 1 and
   critic 2 (critic_td_update_step with gamma=0 -> target = r_t, the same target as the old per-episode pretrain; no Polyak, no actor).
3. Measure after every step (and at step 0): MSE of both critics against r_t on ALL buffer windows (+ per instance at the end), mean Q of both critics,
   mean target. Writes critic_pretrain_log.json, critic weights and a figure (PNG + PDF in figures_export/SRL/).

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_critic_pretrain_buffer_test [seed=301] [steps=50] [batch=20] [critic_lr=0.002] [reward_mode=local_served]
"""
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from rtv_solver.pipeline import srl_buffered_actor_lr_trial as trial  # import side effects: feature-builder patches (pickup_slack, feature size) as in the real runs
from rtv_solver.pipeline.srl_training_loop import REPO_ROOT, MANIFEST_DIR, _MAX_THREAD_CNT
from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.critic_update import critic_td_update_step
from rtv_solver.pipeline.replay_buffer import ReplayBuffer
from rtv_solver.pipeline import select_feature_builder_class
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

PRETRAIN_INSTANCES = ["lc101", "lc105", "lc201", "lr101", "lr109", "lr201", "lr207", "lrc101", "lrc106", "lrc201"]


def _buffer_stats(entries, critics) -> list[dict]:
    """Per critic: MSE against r_t and mean Q over the given entries (no gradient)."""
    out = []
    with torch.no_grad():
        for c in critics:
            qs = [c(e.step.request_features, e.step.vehicle_features, e.step.edge_index).item() for e in entries]
            out.append({"mse": sum((q - e.reward) ** 2 for q, e in zip(qs, entries)) / len(entries), "q_mean": sum(qs) / len(qs)})
    return out


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 301
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    critic_lr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.002
    reward_mode = sys.argv[5] if len(sys.argv) > 5 else "local_served"
    run_id = f"pretrain_buf_s{seed}_n{steps}_b{batch}_{reward_mode}"
    output_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_loggers(output_dir)
    set_seed(seed, debug=False)
    print(f"=== {run_id}: instances={PRETRAIN_INSTANCES} steps={steps} batch={batch} critic_lr={critic_lr} reward_mode={reward_mode} seed={seed} ===")

    config_template = Config(MAX_THREAD_CNT=_MAX_THREAD_CNT, OUTPUT_DIR=output_dir, BATCH_INTERVAL=trial.BATCH_INTERVAL, STEP_SIZE=trial.STEP_SIZE, SEED=seed, KEEP_ACTIVE=False)
    actor = build_scoring_model("mlp", feature_dim=select_feature_builder_class(config_template).FEATURE_SIZE, hidden_dim=64, num_message_passing_layers=2, aggregator="gcn")
    critic = CriticGNN(aggregator="gat", gat_num_heads=4, num_message_passing_layers=2)
    critic2 = CriticGNN(aggregator="gat", gat_num_heads=4, num_message_passing_layers=2)
    opt1, opt2 = torch.optim.Adam(critic.parameters(), lr=critic_lr), torch.optim.Adam(critic2.parameters(), lr=critic_lr)
    frozen1, frozen2 = copy.deepcopy(critic), copy.deepcopy(critic2)  # only needed by the TD interface; gamma=0 makes the bootstrap term vanish

    buffer = ReplayBuffer(capacity=2000)
    spans = {}
    for instance in PRETRAIN_INSTANCES:
        input_path = MANIFEST_DIR / f"{instance}.json"
        inst_dir = output_dir / "collect" / instance
        inst_dir.mkdir(parents=True, exist_ok=True)
        config = Config(MAX_THREAD_CNT=_MAX_THREAD_CNT, OUTPUT_DIR=inst_dir, MODE="coaml", BATCH_INTERVAL=trial.BATCH_INTERVAL, STEP_SIZE=trial.STEP_SIZE, SEED=seed, KEEP_ACTIVE=False)
        payload = PayloadParser.load_input_data(input_path)
        cleared = PayloadParser.clear_vehicle_manifests(payload)
        start = len(buffer)
        pipeline = COAMLPipeline(
            config, cleared, imitation_solution_path=input_path, model=actor,
            critic=critic, critic_optimizer=opt1, target_critic=frozen1, critic2=critic2, critic_optimizer2=opt2, target_critic2=frozen2,
            replay_buffer=buffer, replay_batch_size=batch, critic_target_mode="td_bootstrap", gamma=0.0, defer_critic_updates=True,
        )
        pipeline.solve_pdptw(cleared, mode="eval", train_critic=True, reward_mode=reward_mode)
        spans[instance] = (start, len(buffer))
        print(f"collected {instance}: {len(buffer) - start} windows, sum r = {sum(e.reward for e in buffer.entries[start:]):.0f}")

    entries = list(buffer.entries)
    mean_target = sum(e.reward for e in entries) / len(entries)
    print(f"buffer: {len(entries)} windows from {len(PRETRAIN_INSTANCES)} instances, mean r_t = {mean_target:.3f}")

    log = []
    s1, s2 = _buffer_stats(entries, [critic, critic2])
    log.append({"step": 0, "batch_loss": None, "mse1": s1["mse"], "mse2": s2["mse"], "q1": s1["q_mean"], "q2": s2["q_mean"], "target_mean": mean_target})
    for step in range(1, steps + 1):
        batch_entries = buffer.sample(batch)
        batch_loss = critic_td_update_step(batch_entries, critic, opt1, frozen1, 0.0, critic2=critic2, critic_optimizer2=opt2, target_critic2=frozen2)
        s1, s2 = _buffer_stats(entries, [critic, critic2])
        log.append({"step": step, "batch_loss": batch_loss, "mse1": s1["mse"], "mse2": s2["mse"], "q1": s1["q_mean"], "q2": s2["q_mean"], "target_mean": mean_target})
        print(f"step {step}/{steps}: batch loss {batch_loss:.3f} | MSE on all {len(entries)} windows: critic1 {s1['mse']:.3f} critic2 {s2['mse']:.3f} | mean Q {s1['q_mean']:.2f}/{s2['q_mean']:.2f} (mean r {mean_target:.2f})")

    per_instance = {}
    for instance, (a, b) in spans.items():
        i1, i2 = _buffer_stats(entries[a:b], [critic, critic2])
        per_instance[instance] = {"windows": b - a, "mean_r": sum(e.reward for e in entries[a:b]) / (b - a), "mse1": i1["mse"], "mse2": i2["mse"], "q1": i1["q_mean"], "q2": i2["q_mean"]}
    with open(output_dir / "critic_pretrain_log.json", "w") as f:
        json.dump({"run_id": run_id, "seed": seed, "steps": steps, "batch": batch, "critic_lr": critic_lr, "reward_mode": reward_mode, "instances": PRETRAIN_INSTANCES,
                   "windows": len(entries), "log": log, "per_instance": per_instance}, f, indent=1)
    torch.save({"critic": critic.state_dict(), "critic2": critic2.state_dict()}, output_dir / "pretrained_critics.pt")
    print("=== per instance after pretraining ===")
    for k, v in per_instance.items():
        print(f"{k}: windows {v['windows']} mean r {v['mean_r']:.2f} | MSE {v['mse1']:.2f}/{v['mse2']:.2f} | Q {v['q1']:.2f}/{v['q2']:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [r["step"] for r in log]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    a.plot(xs, [r["mse1"] for r in log], color="#0072B2", lw=2, marker="o", ms=3, label="critic 1")
    a.plot(xs, [r["mse2"] for r in log], color="#D55E00", lw=2, marker="o", ms=3, label="critic 2")
    a.set_yscale("log"); a.set_xlabel("pretraining step (batch %d)" % batch); a.set_ylabel("MSE vs r_t on all %d buffer windows" % len(entries)); a.grid(alpha=0.3); a.legend(frameon=False)
    a.set_title("Critic MSE during buffer pretraining")
    b.plot(xs, [r["q1"] for r in log], color="#009E73", lw=2, label="mean Q critic 1")
    b.plot(xs, [r["q2"] for r in log], color="#E69F00", lw=2, label="mean Q critic 2")
    b.axhline(mean_target, color="#CC79A7", lw=2, ls="--", label="mean target r_t")
    b.set_xlabel("pretraining step"); b.set_ylabel("value"); b.grid(alpha=0.3); b.legend(frameon=False); b.set_title("Mean Q vs mean target (same windows)")
    fig.suptitle(f"{run_id}: 10 instances, {len(entries)} windows (source: outputs/srl_training_sweep/{run_id}/critic_pretrain_log.json)", fontsize=9)
    fig.tight_layout()
    out = REPO_ROOT / "figures_export" / "SRL"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"critic_pretrain_buffer_{run_id}.{ext}", dpi=160)
    print("DONE", output_dir)


if __name__ == "__main__":
    main()
