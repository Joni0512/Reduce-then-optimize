"""
2026-08-21: test a chosen softmax temperature tau (Algorithm 1 step 5, see
srl_target_action.py's softmax_target_action()) on a real iteration -
pretrains a critic (reward_mode="local") on the given train split, takes one
live iteration of a held-out instance, draws candidates, scores them, and
prints the resulting softmax weights + how concentrated they are.

Not part of the training pipeline - diagnostic only, mirrors
diagnose_q_spread.py's structure.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.srl_target_action import sample_candidate_assignments, score_candidates, softmax_target_action
from rtv_solver.pipeline.train_critic import MANIFEST_DIR
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_one_episode(instance, critic, critic_optimizer, actor_checkpoint, batch_interval, step_size, seed, output_dir, train_critic):
    input_path = MANIFEST_DIR / f"{instance}.json"
    instance_output_dir = output_dir / instance
    instance_output_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=instance_output_dir, MODE="coaml", BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)

    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path, critic=critic, critic_optimizer=critic_optimizer)
    pipeline.load_model_weights(actor_checkpoint)
    pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=train_critic, reward_mode="local")
    return pipeline


def _train_critic(actor_checkpoint, batch_interval, step_size, seed, epochs, train_instances, output_dir):
    critic = CriticGNN()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for instance in train_instances:
            _run_one_episode(instance, critic, critic_optimizer, actor_checkpoint, batch_interval, step_size, seed, output_dir, train_critic=True)
        print(f"pretrain epoch {epoch} done")
    return critic


def test_tau(
    actor_checkpoint: str,
    train_instances: list[str],
    instances: list[str],
    tau: float,
    batch_interval: int = 200,
    step_size: int = 100,
    seed: int = 42,
    pretrain_epochs: int = 10,
    num_samples: int = 10,
    sigma: float = 1.0,
    run_label: str = "mixed",
) -> None:
    output_dir = REPO_ROOT / "outputs" / "test_softmax_tau" / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pretraining critic on {len(train_instances)} instances (reward_mode=local)...")
    critic = _train_critic(actor_checkpoint, batch_interval, step_size, seed, pretrain_epochs, train_instances, output_dir)

    rows = []
    for instance in instances:
        print(f"--- {instance} ---")
        throwaway_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        pipeline = _run_one_episode(instance, critic, throwaway_optimizer, actor_checkpoint, batch_interval, step_size, seed, output_dir, train_critic=False)

        theta = pipeline.last_iteration_theta
        oracle = pipeline.last_iteration_oracle
        trip_costs = pipeline.last_iteration_trip_costs
        active_requests = pipeline.last_iteration_active_requests
        vehicles = pipeline.last_iteration_vehicles
        requests = pipeline.last_iteration_requests
        current_time = pipeline.last_iteration_current_time

        candidates = sample_candidate_assignments(theta, oracle, num_samples=num_samples, sigma=sigma)
        q_values = score_candidates(
            candidates, requests, vehicles, trip_costs, active_requests, current_time,
            pipeline.feature_builder, pipeline.match_graph_builder, pipeline.match_feature_builder, critic,
        )
        weights, target_action = softmax_target_action(candidates, q_values, tau)

        q_floats = [q.item() for q in q_values]
        w_floats = weights.tolist()
        max_weight = max(w_floats)
        print(f"Q-values: {[round(q, 4) for q in q_floats]}")
        print(f"softmax weights (tau={tau}): {[round(w, 4) for w in w_floats]}")
        print(f"max weight: {max_weight:.4f} (1.0 = fully concentrated on one candidate, {1/num_samples:.4f} = uniform)")

        for i in range(num_samples):
            rows.append({"instance": instance, "candidate_index": i, "q_value": q_floats[i], "weight": w_floats[i]})

    csv_path = output_dir / f"softmax_weights_tau{tau}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance", "candidate_index", "q_value", "weight"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    # plot: one subplot per instance, bar chart of weights
    fig, axes = plt.subplots(1, len(instances), figsize=(4 * len(instances), 4), squeeze=False)
    for ax, instance in zip(axes[0], instances):
        inst_rows = [r for r in rows if r["instance"] == instance]
        ax.bar(range(len(inst_rows)), [r["weight"] for r in inst_rows], color="tab:blue", alpha=0.85)
        ax.axhline(1 / num_samples, color="gray", linestyle=":", label="uniform")
        ax.set_title(instance)
        ax.set_xlabel("Candidate index")
        ax.set_ylabel("Softmax weight")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Softmax target-action weights, tau={tau} (reward_mode=local, {run_label})")
    fig.tight_layout()

    png_path = REPO_ROOT / "figures_export" / f"softmax_weights_tau{tau}_{run_label}.png"
    pdf_path = REPO_ROOT / "figures_export" / f"softmax_weights_tau{tau}_{run_label}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    print(f"Saved {png_path} and {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a softmax temperature tau on real iterations.")
    parser.add_argument("--actor_checkpoint", type=str, required=True)
    parser.add_argument("--train_instances", type=str, required=True, help="Comma-separated train instances for critic pretraining.")
    parser.add_argument("--instances", type=str, required=True, help="Comma-separated held-out instances to test tau on.")
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--batch_interval", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--run_label", type=str, default="mixed")
    args = parser.parse_args()

    test_tau(
        actor_checkpoint=args.actor_checkpoint,
        train_instances=[s.strip() for s in args.train_instances.split(",") if s.strip()],
        instances=[s.strip() for s in args.instances.split(",") if s.strip()],
        tau=args.tau,
        batch_interval=args.batch_interval,
        step_size=args.step_size,
        seed=args.seed,
        pretrain_epochs=args.pretrain_epochs,
        num_samples=args.num_samples,
        sigma=args.sigma,
        run_label=args.run_label,
    )
