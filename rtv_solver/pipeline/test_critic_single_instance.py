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

2026-08-15: first --mode eval run (lc101, bi=200/ss=100) gave much worse G_t
than --mode offline (min -53 vs -3) - expected, not a bug: no `model=` is
passed in here, so COAMLPipeline builds a fresh, untrained ScoringMLP, and
random scores make the ILP pick badly (~0/53 served) versus offline's
cost-driven decisions (52/53 served). See chat for the full comparison.

2026-08-18: added --coaml_model_weights, following the exact pattern main.py
uses (coaml_pipeline.py line ~357-361): build COAMLPipeline first with its
default (fresh, untrained) MODEL_TYPE='mlp' ScoringMLP, then call
pipeline.load_model_weights(path) before solve_pdptw(mode=...). This lets
--mode eval run with an actual trained actor instead of random weights, so
its scores meaningfully drive which requests get served - a prerequisite for
evaluating the critic against a non-random policy (see chat).
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
    mode: str = "offline",
    coaml_model_weights: str | None = None,
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

    # 2026-08-18: loaded AFTER pipeline construction, matching main.py's
    # pattern exactly (coaml_pipeline.py line ~357-361) - COAMLPipeline
    # always builds its own fresh model first (config.MODEL_TYPE, default
    # "mlp"), load_model_weights() then overwrites those weights in place.
    if coaml_model_weights:
        pipeline.load_model_weights(coaml_model_weights)

    # 2026-08-15: mode is now a parameter, not hardcoded - "offline" (pure
    # cost-minimization, RHO baseline) never lets scores drive a decision, so
    # it can't exercise the perturb-scores-and-ask-the-critic mechanism
    # meaningfully (see chat). "eval" uses the actor's own (here: freshly
    # initialized, untrained) scores to decide, which at least makes the
    # scores relevant to what actually happens.
    pipeline.solve_pdptw(cleared_payload, mode=mode)

    returns = pipeline.last_episode_returns
    predictions = pipeline.last_episode_predictions
    print(f"\n=== Critic smoke test: {instance}, mode={mode}, bi={batch_interval}, ss={step_size} ===")
    print(f"buffered iterations: {len(returns)}")
    print(f"G_t (reality):    {[round(g, 3) for g in returns]}")
    print(f"Q_theta (prediction): {[round(q, 3) for q in predictions]}")

    if not returns:
        print("WARNING: no iterations were buffered - nothing to check.")
        return

    # 2026-08-15: diagnostic only, NOT the training target - shows which
    # window each permanent miss actually happened in, by taking consecutive
    # differences of the cumulative G_t. Kept separate from G_t on purpose:
    # training on this instead would make the critic myopic again (see chat).
    # Only len(returns)-1 values exist (a difference needs two neighbours);
    # the last window's own contribution is already visible in G_t's last
    # entry directly, no extra diff needed for it.
    local_misses = [round(returns[i] - returns[i + 1], 3) for i in range(len(returns) - 1)]
    print(f"local misses per window (diagnostic only): {local_misses}")

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
    parser.add_argument("--mode", type=str, default="offline", choices=["train", "eval", "optimal", "offline"])
    parser.add_argument(
        "--coaml_model_weights",
        type=str,
        default="",
        help="Path to a saved COAML checkpoint (from COAMLTrainingLoop/save_model_weights) to load before "
        "solving. Empty = fresh untrained model (old default behavior). Only meaningful for --mode eval.",
    )
    args = parser.parse_args()

    run_critic_single_instance(
        instance=args.instance,
        batch_interval=args.batch_interval,
        step_size=args.step_size,
        seed=args.seed,
        mode=args.mode,
        coaml_model_weights=args.coaml_model_weights or None,
    )
