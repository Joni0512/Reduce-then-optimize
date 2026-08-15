from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from rtv_solver.pipeline.mc_return_builder import MonteCarloReturnBuilder
from rtv_solver.structure.request import Request


@dataclass
class EpisodeStep:
    """One rolling-horizon iteration's critic input, collected during an episode."""
    request_features: torch.Tensor
    vehicle_features: torch.Tensor
    edge_index: torch.Tensor
    current_time: float


class EpisodeBuffer:
    """
    Collects one episode's critic inputs (one MatchGraph per rolling-horizon
    iteration) so they can all be trained on together once the episode - and
    with it, the Monte Carlo return G_t for every step - is actually known.

    2026-08-14: len(buffer) after an episode is NOT just end_time/STEP_SIZE -
    a rolling-horizon loop pass only calls solve_iteration() (which is where
    add() gets called) when that window actually has a selected request;
    empty windows are skipped and never buffered. On top of that, the exact
    count can differ slightly between otherwise-identical runs (same seed) -
    seen empirically on lc101 (bi=200/ss=100): 9 buffered steps in one run,
    10 in the next. Likely comes from TripHandler's multiprocessing.Pool
    trip-cost generation - worker completion order can affect ILP tie-breaks
    even with the seed fixed. Not a bug introduced by the critic wiring, just
    something to know before treating a specific buffered count as "the"
    right number for an instance.

    Short-lived on purpose: build one per episode (one solve_pdptw call),
    fill it during the rolling-horizon loop, consume it with with_returns()
    right after the loop ends, then let it go. This is NOT a persistent,
    cross-episode replay buffer like TD methods use - pure Monte Carlo
    training doesn't need one, see the critic training discussion.
    """

    def __init__(self) -> None:
        self.steps: list[EpisodeStep] = []

    def add(
        self,
        request_features: torch.Tensor,
        vehicle_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_time: float,
    ) -> None:
        self.steps.append(EpisodeStep(request_features, vehicle_features, edge_index, current_time))

    def __len__(self) -> int:
        return len(self.steps)

    def with_returns(
        self,
        requests: Sequence[Request],
        serviced_request_ids: Sequence[int],
    ) -> list[tuple[EpisodeStep, float]]:
        """
        Call once, right after the rolling-horizon loop for this episode
        has finished. Pairs every buffered step with its Monte Carlo
        return G_t (mc_return_builder.py), ready for training.
        """
        iteration_times = [step.current_time for step in self.steps]
        returns = MonteCarloReturnBuilder(requests).build(iteration_times, serviced_request_ids)
        return list(zip(self.steps, returns))

    def clear(self) -> None:
        self.steps.clear()
