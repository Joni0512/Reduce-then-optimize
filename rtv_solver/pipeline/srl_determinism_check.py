"""
2026-09-17: determinism control check (see chat) - runs run_srl_training_loop()
TWICE with the identical actor_lr/critic_lr and deterministic=True (forces
set_seed(..., debug=True) -> torch.use_deterministic_algorithms(True)), a
short EPOCHS count to keep runtime down, and compares the resulting val
curves for exact equality.

Motivation: across the actor_lr x critic_lr wandb sweep, some trials with
near-identical hyperparameters (e.g. cluster_c3c6e04e: actor_lr=0.00219,
critic_lr=0.00264 -> succeeded at 0.61, vs. several other trials with
similar-magnitude values that collapsed to 0.0) gave contradictory outcomes.
set_seed() was found to be hardcoded to debug=False in srl_training_loop.py
(fixed in the same commit as this script), meaning torch's deterministic-
algorithms flag was never enabled despite a fixed seed - GAT's attention
mechanism uses scatter/segment reduction ops that are a known source of
run-to-run floating-point non-determinism without that flag. This script
tests whether deterministic=True actually removes the discrepancy.

Uses cluster_c3c6e04e's own actor_lr/critic_lr as the test point (a value
that produced BOTH a success and, at very similar magnitudes, several
collapses elsewhere in the sweep - the most informative point to test).

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_determinism_check
"""
from __future__ import annotations

from pathlib import Path

from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

from rtv_solver.pipeline.srl_training_loop import run_srl_training_loop, REPO_ROOT
from rtv_solver.pipeline.srl_rho_outcome_advantage import ACTOR_CHECKPOINT

ACTOR_LR = 0.002193283565932468
CRITIC_LR = 0.002638378950453865
EPOCHS = 3
VAL_EVERY_N_EPOCHS = 1
CRITIC_PRETRAIN_EPOCHS = 2  # shortened from the sweep default (10) to keep this check's runtime down


def main():
    val_curves = []
    for run_idx in (1, 2):
        output_dir = REPO_ROOT / "outputs" / "srl_determinism_check" / f"run{run_idx}"
        result = run_srl_training_loop(
            reward_mode="local",
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            output_dir=output_dir,
            actor_checkpoint=ACTOR_CHECKPOINT,
            epochs=EPOCHS,
            val_every_n_epochs=VAL_EVERY_N_EPOCHS,
            critic_pretrain_epochs=CRITIC_PRETRAIN_EPOCHS,
            deterministic=True,
        )
        val_rates = [v["service_rate"] for v in result.val_curve]
        val_curves.append(val_rates)
        print(f"[determinism_check] run{run_idx}: val_curve={val_rates}")

    if val_curves[0] == val_curves[1]:
        print("[determinism_check] IDENTICAL - deterministic=True reproduces the exact same val curve across runs.")
    else:
        print(f"[determinism_check] DIFFERENT - run1={val_curves[0]} vs run2={val_curves[1]} - deterministic=True did NOT remove run-to-run variance.")


if __name__ == "__main__":
    main()
