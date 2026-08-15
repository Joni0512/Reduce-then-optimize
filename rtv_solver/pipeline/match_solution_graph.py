from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from rtv_solver.structure.assignment_result import AssignmentResult
from rtv_solver.structure.request import Request
from rtv_solver.structure.vehicle import Vehicle
from rtv_solver.structure.vehicle_stop import VehicleStop


@dataclass(frozen=True)
class MatchGraph:
    """
    Graph for ONE rolling-horizon iteration's actual result (not all
    candidates, only the picked solution). Two node types, in this order:

        [request nodes..., vehicle nodes...]

    A request node has no edges if it was not assigned to a vehicle this
    iteration (isolated node = "not dispatched yet").
    """

    edge_index: torch.Tensor
    request_ids: tuple[int, ...]
    vehicle_ids: tuple[int, ...]
    is_assigned: tuple[bool, ...]  # one flag per request node, same order as request_ids

    @property
    def num_requests(self) -> int:
        return len(self.request_ids)

    @property
    def num_vehicles(self) -> int:
        return len(self.vehicle_ids)

    @property
    def num_nodes(self) -> int:
        return self.num_requests + self.num_vehicles


class MatchSolutionGraphBuilder:
    """
    Builds the "solution graph" the critic will later run on.

    Follows the DVSP critic from the SRL paper (Hoppe et al., "Structured
    Reinforcement Learning", NeurIPS 2025, Appendix D.4): requests and
    vehicles are nodes, and an edge only exists where the solver actually
    put a request on a vehicle's route this iteration.

    This is a DIFFERENT graph than CandidateConflictGraphBuilder (used by
    the actor GNN): that one connects candidates that competed with each
    other and were NOT jointly picked. Here, an edge means "these two ended
    up together in the chosen solution".

    2026-08-06: scope decided in chat - V1 only includes requests currently
    up for decision (trip_handler.requests: new + still-open active
    requests). Already-boarded requests are left out as graph nodes; that
    state is already covered by the vehicle features instead.
    """

    def build(
        self,
        requests: Sequence[Request],
        vehicles: dict[int, Vehicle],
        result: AssignmentResult,
        *,
        device: torch.device | str | None = None,
    ) -> MatchGraph:
        request_ids = tuple(int(r.id) for r in requests)
        vehicle_ids = tuple(sorted(vehicles.keys()))

        # node index lookup: requests come first, vehicles right after
        request_index = {rid: i for i, rid in enumerate(request_ids)}
        vehicle_index = {vid: len(request_ids) + i for i, vid in enumerate(vehicle_ids)}

        edge_pairs: set[tuple[int, int]] = set()

        for vehicle_id, (_, sequence) in result.vehicle_assignment.items():
            if vehicle_id not in vehicle_index:
                continue  # not one of our graph's vehicles, should not happen

            ordered_ids = self._ordered_request_ids_from_sequence(sequence)
            # only keep ids that are actual graph nodes (drops boarded requests, see class docstring)
            known_ids = [rid for rid in ordered_ids if rid in request_index]

            for rid in known_ids:
                self._add_undirected(edge_pairs, request_index[rid], vehicle_index[vehicle_id])

            # consecutive requests on the same route get connected to each other too
            for left, right in zip(known_ids, known_ids[1:]):
                self._add_undirected(edge_pairs, request_index[left], request_index[right])

        is_assigned = tuple(rid in result.request_assignment for rid in request_ids)

        graph = MatchGraph(
            edge_index=self._edge_pairs_to_tensor(edge_pairs, device),
            request_ids=request_ids,
            vehicle_ids=vehicle_ids,
            is_assigned=is_assigned,
        )
        self._validate(graph)
        return graph

    @staticmethod
    def _ordered_request_ids_from_sequence(sequence: list[VehicleStop]) -> list[int]:
        """
        Turns a route's stop sequence into an ordered list of request ids,
        one entry per request (its first stop, always the pickup).
        """
        seen: set[int] = set()
        ordered: list[int] = []
        for stop in sequence:
            if stop.type not in (VehicleStop.ACT_PICKUP, VehicleStop.ACT_DROPOFF):
                continue  # depot/rebalance stops have no request id
            request_id = stop.get_requestID_stripped_iteration()
            if request_id not in seen:
                seen.add(request_id)
                ordered.append(request_id)
        return ordered

    @staticmethod
    def _add_undirected(edge_pairs: set[tuple[int, int]], a: int, b: int) -> None:
        # two directed edges per connection, same convention as CandidateConflictGraphBuilder
        edge_pairs.add((a, b))
        edge_pairs.add((b, a))

    @staticmethod
    def _edge_pairs_to_tensor(
        edge_pairs: set[tuple[int, int]],
        device: torch.device | str | None,
    ) -> torch.Tensor:
        if not edge_pairs:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        ordered_edges = sorted(edge_pairs)
        return torch.tensor(ordered_edges, dtype=torch.long, device=device).t().contiguous()

    @staticmethod
    def _validate(graph: MatchGraph) -> None:
        if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if graph.edge_index.numel() > 0:
            min_index = int(graph.edge_index.min().item())
            max_index = int(graph.edge_index.max().item())
            if min_index < 0 or max_index >= graph.num_nodes:
                raise ValueError("edge_index contains a node index outside the graph range.")
        if len(graph.is_assigned) != graph.num_requests:
            raise ValueError("is_assigned must have one entry per request node.")
