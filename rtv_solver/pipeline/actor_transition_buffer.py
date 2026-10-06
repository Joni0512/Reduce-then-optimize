"""
2026-10-06: actor transition buffer for the "collect, then update" SRL actor
scheme (see chat / docs: project_srl_minibatch_update_plan). OPT-IN - the old
per-window actor update (COAMLPipeline._compute_srl_actor_loss called inline in
solve_iteration) is untouched and stays the default; this module is only used
when a COAMLPipeline is built with actor_transition_buffer=<this buffer>.

One ActorTransition = one rolling-horizon window (one solve_iteration call) of
one instance. It stores everything needed to re-run the actor loss LATER, with
the actor/critic weights of that later moment:
  - the actor input (feature tensor incl. reject rows) so theta can be
    recomputed with the then-current actor weights,
  - the combinatorial structure (trip_costs, maps, requests, vehicles,
    active_requests, current_time) for the MAP oracle and the critic's
    match-graph features,
  - a reference to the COAMLPipeline that produced it (it owns the
    per-instance feature builder / match builders / config the loss needs;
    actor and target critics are shared objects across all pipelines).
NOT stored: theta and the target action a_hat - both depend on the weights at
update time and are rebuilt there.

All structural objects are deep-copied together (one deepcopy call, so shared
references between e.g. trip_costs and requests stay shared) because
Vehicle/Trip objects are mutated right after the window is solved
(vehicle.apply_trip_insertion) - without the snapshot the update phase would
see the post-decision state instead of the state the old inline loss saw.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ActorTransition:
    pipeline: Any  # COAMLPipeline that collected this window (owns per-instance builders/config)
    feature_tensor: torch.Tensor  # (n_trips + n_reject, F), actor input incl. reject rows
    reject_vehicle_ids: list[int]
    trip_costs: list
    trip_list: list
    single_trip_map: dict
    vehicle_to_trips_cost_map: dict
    trip_to_vehicle_cost_map: dict
    requests: list
    vehicles: dict
    active_requests: dict
    current_time: float


class ActorTransitionBuffer:
    """
    FIFO store of ActorTransition entries.

    capacity=None (default): unbounded; the caller clears it after every update
    phase, so it holds exactly one epoch of windows ("collect, then update").
    capacity=int: ring buffer, oldest transitions are evicted first (for later
    experiments that keep more than one epoch).
    """

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be None or >= 1, got {capacity}")
        self.capacity = capacity
        self._items: list[ActorTransition] = []

    def add(self, transition: ActorTransition) -> None:
        self._items.append(transition)
        if self.capacity is not None and len(self._items) > self.capacity:
            self._items.pop(0)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def epoch_batches(self, batch_size: int, rng: random.Random) -> list[list[ActorTransition]]:
        """
        Random permutation of all stored transitions split into batches of
        batch_size (last batch may be smaller). Every transition appears
        exactly once - sampling WITHOUT replacement, unlike the critic's
        ReplayBuffer.sample().
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        order = list(range(len(self._items)))
        rng.shuffle(order)
        return [
            [self._items[i] for i in order[start:start + batch_size]]
            for start in range(0, len(order), batch_size)
        ]
