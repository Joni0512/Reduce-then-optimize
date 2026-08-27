"""
2026-08-20: one-off diagnostic for choosing the softmax temperature tau
(SRL actor-critic integration, step 5 - see chat/
figures_export/srl_actor_critic_steps_slide.pptx). Not part of the training
pipeline - trains a critic exactly like train_critic.py (reward_mode="local"
is our standing default per user instruction), then takes ONE real iteration
of a held-out instance, draws num_samples perturbed candidate assignments,
scores each with the critic, and reports the spread (mean/std/min/max) of
those Q-values - the raw material for picking tau empirically instead of
guessing it.
"""
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.srl_target_action import sample_candidate_assignments, score_candidates
from rtv_solver.pipeline.train_critic import TRAIN_INSTANCES, VAL_INSTANCES, MANIFEST_DIR
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _train_critic(actor_checkpoint: str, batch_interval: int, step_size: int, seed: int, epochs: int, train_instances: list[str]) -> torch.nn.Module:
    critic = CriticGNN()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    output_dir = REPO_ROOT / "outputs" / "diagnose_q_spread" / "critic_pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        for instance in train_instances:
            _run_one_episode(instance, critic, critic_optimizer, actor_checkpoint, batch_interval, step_size, seed, output_dir, train_critic=True)
        print(f"pretrain epoch {epoch} done")
    return critic


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


def diagnose(
    actor_checkpoint: str,
    instances: list[str] | None = None,
    batch_interval: int = 200,
    step_size: int = 100,
    seed: int = 42,
    pretrain_epochs: int = 10,
    num_samples: int = 10,
    sigma: float = 1.0,
    train_instances: list[str] | None = None,
) -> None:
    instances = instances or ["lc108"]
    train_instances = train_instances if train_instances is not None else TRAIN_INSTANCES
    print("Pretraining critic on the given train split (reward_mode=local)...")
    critic = _train_critic(actor_checkpoint, batch_interval, step_size, seed, pretrain_epochs, train_instances)

    for instance in instances:
        _diagnose_one_instance(critic, actor_checkpoint, instance, batch_interval, step_size, seed, num_samples, sigma, pretrain_epochs)


def _diagnose_one_instance(
    critic: torch.nn.Module,
    actor_checkpoint: str,
    instance: str,
    batch_interval: int,
    step_size: int,
    seed: int,
    num_samples: int,
    sigma: float,
    pretrain_epochs: int,
) -> None:
    print(f"Running one real episode on held-out instance '{instance}' to capture a live iteration...")
    output_dir = REPO_ROOT / "outputs" / "diagnose_q_spread"
    # reuse a throwaway optimizer here - train_critic=False, this episode never updates critic weights
    throwaway_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    pipeline = _run_one_episode(instance, critic, throwaway_optimizer, actor_checkpoint, batch_interval, step_size, seed, output_dir, train_critic=False)

    theta = pipeline.last_iteration_theta
    oracle = pipeline.last_iteration_oracle
    trip_costs = pipeline.last_iteration_trip_costs
    active_requests = pipeline.last_iteration_active_requests
    vehicles = pipeline.last_iteration_vehicles
    requests = pipeline.last_iteration_requests
    current_time = pipeline.last_iteration_current_time

    print(f"theta shape: {theta.shape}, drawing {num_samples} candidates with sigma={sigma}...")
    candidates = sample_candidate_assignments(theta, oracle, num_samples=num_samples, sigma=sigma)

    q_values = score_candidates(
        candidates, requests, vehicles, trip_costs, active_requests, current_time,
        pipeline.feature_builder, pipeline.match_graph_builder, pipeline.match_feature_builder, critic,
    )
    q_floats = [q.item() for q in q_values]

    print(f"Q-values ({num_samples} candidates): {[round(q, 4) for q in q_floats]}")
    q_mean = statistics.mean(q_floats)
    q_std = statistics.pstdev(q_floats)
    q_min, q_max = min(q_floats), max(q_floats)
    print(f"mean={q_mean:.4f}  std={q_std:.4f}  min={q_min:.4f}  max={q_max:.4f}")

    # save CSV
    csv_path = output_dir / f"q_spread_{instance}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_index", "q_value"])
        for i, q in enumerate(q_floats):
            writer.writerow([i, q])
        writer.writerow([])
        writer.writerow(["mean", q_mean])
        writer.writerow(["std", q_std])
        writer.writerow(["min", q_min])
        writer.writerow(["max", q_max])
    print(f"Saved {csv_path}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(range(num_samples), q_floats, color="tab:blue", alpha=0.8)
    ax.axhline(q_mean, color="tab:orange", linestyle="--", label=f"mean = {q_mean:.4f}")
    ax.fill_between([-0.5, num_samples - 0.5], q_mean - q_std, q_mean + q_std, color="tab:orange", alpha=0.15, label=f"+/- std ({q_std:.4f})")
    ax.set_xlabel("Candidate index (perturbation sample)")
    ax.set_ylabel("Q(s_j, y^(i))")
    ax.set_title(
        f"Q-value spread across {num_samples} perturbed candidates\n"
        f"instance={instance}, sigma={sigma}, reward=local, pretrain_epochs={pretrain_epochs}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = REPO_ROOT / "figures_export" / f"q_spread_diagnostic_{instance}.png"
    pdf_path = REPO_ROOT / "figures_export" / f"q_spread_diagnostic_{instance}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    print(f"Saved {png_path} and {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose Q-value spread across perturbed candidates, to inform softmax temperature tau.")
    parser.add_argument("--actor_checkpoint", type=str, required=True)
    parser.add_argument("--instances", type=str, default="lc108", help="Comma-separated held-out instances for the live iteration (default: a class-1 val instance).")
    parser.add_argument("--batch_interval", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--train_instances", type=str, default="", help="Comma-separated train instances for critic pretraining. Empty = class-1 default.")
    args = parser.parse_args()

    diagnose(
        actor_checkpoint=args.actor_checkpoint,
        instances=[s.strip() for s in args.instances.split(",") if s.strip()],
        batch_interval=args.batch_interval,
        step_size=args.step_size,
        seed=args.seed,
        pretrain_epochs=args.pretrain_epochs,
        num_samples=args.num_samples,
        sigma=args.sigma,
        train_instances=[s.strip() for s in args.train_instances.split(",") if s.strip()] or None,
    )
