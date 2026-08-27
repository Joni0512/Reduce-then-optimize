"""
2026-08-20: building blocks of the SRL actor-critic integration (Algorithm 1,
steps 1-5 so far - see chat/figures_export/srl_actor_critic_integration_steps.tex).

Not wired into COAMLPipeline yet. Remaining step (feed the target action as
y_star into the existing FenchelYoungLoss to update the actor) still to be
built.

2026-08-20: per user instruction, all critic work (this file included) uses
reward_mode="local" (r_t) unless explicitly told otherwise - see
mc_return_builder.py's docstring for the local vs. cumulative tradeoff.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import torch

from rtv_solver.pipeline.match_graph_features import MatchGraphFeatureBuilder
from rtv_solver.pipeline.match_solution_graph import MatchSolutionGraphBuilder
from rtv_solver.structure.request import Request
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.vehicle import Vehicle


def sample_candidate_assignments(
    theta: torch.Tensor,
    oracle: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int,
    sigma: float,
) -> List[torch.Tensor]:
    """
    Draw num_samples perturbations of theta and solve the MAP oracle for
    each, returning the resulting binary assignment vectors y^(i).

    This is the same perturb-and-solve loop FenchelYoungLoss.forward() already
    runs internally (loss_FYscoring.py, lines 76-82) - reused here verbatim,
    except the individual y^(i) are kept and returned instead of being
    reduced into a single dot-product/loss value. theta is only read, not
    part of the returned candidates' graph (oracle calls happen under
    no_grad, see map_oracle.py's oracle() docstring).

    Args:
        theta:       (n,) raw actor scores for this iteration.
        oracle:      Callable[[Tensor (n,)], Tensor (n,)] - MAP oracle
                     (see map_oracle.py's make_map_oracle()). Must be safe to
                     call under torch.no_grad().
        num_samples: how many perturbations/candidates to draw (m in the
                     algorithm sketch).
        sigma:       standard deviation of the Gaussian perturbation.

    Returns:
        List of num_samples binary assignment tensors y^(i) in {0,1}^n
        (in code: y_k), one per perturbation.
    """
    if theta.ndim != 1:
        raise ValueError(f"Expected a 1-D score tensor, got shape {theta.shape}.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")

    candidates: List[torch.Tensor] = []
    for _ in range(num_samples):
        noise = torch.randn_like(theta) * sigma
        theta_k = theta + noise
        y_k = oracle(theta_k.detach())
        candidates.append(y_k)
    return candidates


def score_candidates(
    candidates: List[torch.Tensor],
    requests: Sequence[Request],
    vehicles: dict[int, Vehicle],
    trip_costs: Sequence[TripCost],
    active_requests: dict,
    current_time: float,
    actor_feature_builder,
    match_graph_builder: MatchSolutionGraphBuilder,
    match_feature_builder: MatchGraphFeatureBuilder,
    critic: torch.nn.Module,
) -> List[torch.Tensor]:
    """
    Algorithm 1 step 4 (see module docstring) - build a MatchGraph per
    candidate assignment y^(i) (step 3, MatchSolutionGraphBuilder.
    build_from_candidate()), then score each with the critic.

    Reuses the exact same match_graph_builder/match_feature_builder
    instances COAMLPipeline already keeps for the real (picked) solution
    (coaml_pipeline.py ~line 127-128), so feature normalization stays
    consistent between the buffered "real" graphs and these candidate ones.

    Returns one scalar Q-value tensor per candidate, same order as
    `candidates`. Q(s, y^(i)) - see step 4 in
    figures_export/srl_actor_critic_integration_steps.tex.
    """
    q_values: List[torch.Tensor] = []
    for y in candidates:
        candidate_graph = match_graph_builder.build_from_candidate(requests, vehicles, trip_costs, y)
        request_features, vehicle_features = match_feature_builder.build(
            requests, vehicles, active_requests, candidate_graph, current_time, actor_feature_builder,
        )
        q_value = critic(request_features, vehicle_features, candidate_graph.edge_index)
        q_values.append(q_value)
    return q_values


def softmax_target_action(
    candidates: List[torch.Tensor],
    q_values: List[torch.Tensor],
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Algorithm 1 step 5 (see module docstring) - softmax over the candidates'
    Q-values, then the target action is the softmax-weighted mixture of the
    candidate assignments (a continuous {0,1}^n-ish target, not a hard pick -
    see chat's LaTeX derivation, figures_export/srl_actor_critic_integration_steps.tex).

        w_i = exp(Q_i / tau) / sum_l exp(Q_l / tau)
        a_hat = sum_i w_i * y_i

    tau controls how sharply the softmax favors the best-scoring candidate(s):
    tau -> 0 approaches a hard argmax, tau -> inf approaches a uniform average
    of all candidates regardless of Q. See diagnose_q_spread.py for how to
    empirically measure the Q-value spread this should be scaled against -
    2026-08-21 measurements found std(Q) ranging ~0.001-0.036 depending on
    the instance/iteration (how many real alternatives existed), so a single
    fixed tau is a compromise, not a universally "correct" value.

    Args:
        candidates: list of m binary assignment tensors y^(i), same order as q_values.
        q_values:   list of m scalar Q-value tensors, same order as candidates.
        tau:        softmax temperature (> 0).

    Returns:
        (weights, target_action) - weights: (m,) tensor of softmax weights,
        target_action: same shape as each y^(i), the weighted mixture a_hat_j.
    """
    if len(candidates) != len(q_values):
        raise ValueError(f"candidates ({len(candidates)}) and q_values ({len(q_values)}) must have the same length.")
    if len(candidates) == 0:
        raise ValueError("candidates must not be empty.")
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}.")

    q_stack = torch.stack([q.reshape(()) for q in q_values])  # (m,)
    weights = torch.softmax(q_stack / tau, dim=0)  # (m,)

    candidate_stack = torch.stack(candidates)  # (m, n)
    target_action = torch.sum(weights.unsqueeze(-1) * candidate_stack, dim=0)  # (n,)

    return weights, target_action
