from __future__ import annotations

import torch

from rtv_solver.pipeline.episode_buffer import EpisodeStep


def build_td_targets(
    step_return_pairs: list[tuple[EpisodeStep, float]],
    target_critic: torch.nn.Module,
    gamma: float,
) -> list[tuple[EpisodeStep, float]]:
    """
    2026-09-09: TD-bootstrap critic target (see chat, docs/SRL_Design.md's
    TD Bootstrap plan section) - additive alternative to the plain Monte
    Carlo target already used elsewhere in the pipeline.

    Replaces each step's r_t with:
        target_t = r_t + gamma * target_critic(next_step)   (all steps but the last)
        target_t = r_t                                       (last step, no next_step)

    step_return_pairs must be the CHRONOLOGICALLY ORDERED list produced by
    EpisodeBuffer.with_returns(reward_mode="local") - r_t is the per-step
    local reward, not the cumulative Monte Carlo return G_t. Do NOT pass a
    shuffled replay-buffer batch here: next_step only makes sense as "the
    next iteration of this same episode".
    """
    td_pairs: list[tuple[EpisodeStep, float]] = []
    for i, (step, r_t) in enumerate(step_return_pairs):
        if i + 1 < len(step_return_pairs):
            next_step, _ = step_return_pairs[i + 1]
            # Bootstrap value is a fixed target, not something target_critic
            # should receive a gradient through here.
            with torch.no_grad():
                bootstrap = target_critic(
                    next_step.request_features, next_step.vehicle_features, next_step.edge_index
                ).item()
            target = r_t + gamma * bootstrap
        else:
            # Terminal step: no next_step to bootstrap from, target is the
            # immediate reward alone.
            target = r_t
        td_pairs.append((step, target))
    return td_pairs
