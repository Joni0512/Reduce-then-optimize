"""
2026-08-14: standalone smoke test for the SRL critic wiring (see chat) - runs
ONE Li&Lim class-1 instance through COAMLPipeline.solve_pdptw with a fresh,
untrained CriticGNN attached, using mode="offline" (pure cost-minimization,
the same objective RHO uses - so this exercises the critic plumbing without
depending on any learned actor at all).

This is not checking whether the critic learns anything useful yet (one
random-weights forward/backward pass proves nothing about that) - it only
checks that the graph/feature/return pipeline runs end to end and produces
sane numbers: right number of buffered steps, G_t values inside [0, 1], no
crashes, no NaNs.
"""
import argparse
from pathlib import Path

import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = REPO_ROOT / "solutions" / "li_lim" / "manifests"


def run_critic_single_instance(
    instance: str = "lc101",
    batch_interval: int = 200,
    step_size: int = 100,
    seed: int = 42,
) -> None:
    input_path = MANIFEST_DIR / f"{instance}.json"
    if not input_path.exists():
        raise FileNotFoundError(f"No manifest for instance '{instance}': {input_path}")

    output_dir = REPO_ROOT / "outputs" / "critic_single_instance" / f"bi{batch_interval}_ss{step_size}" / instance
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        OUTPUT_DIR=output_dir,
        MODE="coaml",
        BATCH_INTERVAL=batch_interval,
        STEP_SIZE=step_size,
        SEED=seed,
    )
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)

    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)

    critic = CriticGNN()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    pipeline = COAMLPipeline(
        config,
        cleared_payload,
        imitation_solution_path=input_path,
        critic=critic,
        critic_optimizer=critic_optimizer,
    )

    # mode="offline" = pure cost-minimization (RHO baseline) - no scoring
    # model influences the decisions, so this checks the critic plumbing
    # independent of whether the actor is any good.
    pipeline.solve_pdptw(cleared_payload, mode="offline")

    returns = pipeline.last_episode_returns
    print(f"\n=== Critic smoke test: {instance}, bi={batch_interval}, ss={step_size} ===")
    print(f"buffered iterations: {len(returns)}")
    print(f"G_t values: {[round(g, 3) for g in returns]}")

    if not returns:
        print("WARNING: no iterations were buffered - nothing to check.")
        return

    # 2026-08-15: G_t is now a negative penalty count (0 or negative, no
    # fixed lower bound), not a [0,1] ratio - so "positive" is the sanity
    # check here, not "out of [0,1]". See mc_return_builder.py.
    positive = [g for g in returns if g > 0.0]
    has_nan = any(g != g for g in returns)  # NaN != NaN
    print(f"min={min(returns):.3f}  max={max(returns):.3f}  mean={sum(returns)/len(returns):.3f}")
    print(f"positive values (should be none): {positive}")
    print(f"contains NaN: {has_nan}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test the SRL critic wiring on one Li&Lim class-1 instance.")
    parser.add_argument("--instance", type=str, default="lc101")
    parser.add_argument("--batch_interval", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_critic_single_instance(
        instance=args.instance,
        batch_interval=args.batch_interval,
        step_size=args.step_size,
        seed=args.seed,
    )
