from __future__ import annotations

import torch

from rtv_solver.pipeline.episode_buffer import EpisodeStep


def td_target_for_step(
    r_t: float,
    next_step: EpisodeStep | None,
    target_critic: torch.nn.Module,
    gamma: float,
) -> float:
    """
    2026-09-10: single-step TD target (see chat) - y = r_t + gamma*(1-done)*
    Q_target(s_{t+1}), extracted out of build_td_targets() so the SAME
    formula can be evaluated either once per episode (build_td_targets(),
    the non-replay-buffer path) OR once per REPLAY-BUFFER SAMPLE (see
    coaml_pipeline.py's batch sampling loop) - fixes the replay-buffer
    staleness bug: targets used to be computed once, at the end of the
    episode that produced them, then frozen in the buffer forever after;
    now they get recomputed with target_critic's CURRENT weights every time
    an entry is sampled, matching DDPG/DQN's own replay-buffer semantics
    (target computed at minibatch-sample time, not at insertion time).

    done = 1 (next_step is None, terminal step) -> bootstrap term is 0,
    target is r_t alone. done = 0 (normal step) -> full bootstrap term.
    """
    done = 1.0 if next_step is None else 0.0
    if done == 1.0:
        q_next = 0.0
    else:
        # Bootstrap value is a fixed target, not something target_critic
        # should receive a gradient through here.
        with torch.no_grad():
            q_next = target_critic(
                next_step.request_features, next_step.vehicle_features, next_step.edge_index
            ).item()
    return r_t + gamma * (1.0 - done) * q_next


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

    2026-09-10: only used by the non-replay-buffer path now (this function
    evaluates the whole episode at once, right when it finishes) - the
    replay-buffer path evaluates td_target_for_step() above instead, once
    per SAMPLED entry, so it can use target_critic's weights at sample time
    rather than this episode's now-fixed snapshot. Reimplemented on top of
    td_target_for_step() so both paths share one formula.
    """
    td_pairs: list[tuple[EpisodeStep, float]] = []
    for i, (step, r_t) in enumerate(step_return_pairs):
        next_step = step_return_pairs[i + 1][0] if i + 1 < len(step_return_pairs) else None
        target = td_target_for_step(r_t, next_step, target_critic, gamma)
        td_pairs.append((step, target))
    return td_pairs
