"""
MAP oracle factory for the Fenchel-Young training pipeline.

The oracle wraps CO_ScoreMaximization: given a (perturbed) score vector it
solves the ILP and returns the resulting binary assignment vector.  It is
intentionally separate from the loss module so that different ILP formulations
can be plugged in without touching the loss.

Usage per solver iteration::

    oracle = make_map_oracle(
        single_trip_map, trips, trip_costs,
        vehicle_to_trips_cost_map, trip_to_vehicle_cost_map,
        requests, active_requests, config,
    )
    loss = fy_loss(scores, y_star, oracle)
"""
from __future__ import annotations

from typing import Callable, Dict, List

import torch
from gurobipy import GRB

from rtv_solver.pipeline.co_scoreMaximization import CO_ScoreMaximization
from rtv_solver.structure.config import Config
from rtv_solver.structure.request import Request
from rtv_solver.structure.trip_cost import TripCost


def make_map_oracle(
    single_trip_map: Dict[int, int],
    trips: list,
    trip_costs: List[TripCost],
    vehicle_to_trips_cost_map: Dict[int, List[int]],
    trip_to_vehicle_cost_map: Dict[int, List[int]],
    requests: List[Request],
    active_requests: Dict[int, Request],
    config: Config,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a MAP oracle callable for the current solver iteration.

    A single CO_ScoreMaximization instance is created and reused across all
    oracle calls (i.e. across all perturbation samples) because the structural
    inputs are fixed for the lifetime of this oracle.  The oracle itself is a
    stateless closure.

    Args:
        single_trip_map:           {request_id: trip_cost_index}
        trips:                     list of Trip / SharedTrip objects
        trip_costs:                list of TripCost objects (length n)
        vehicle_to_trips_cost_map: {vehicle_id: [trip_cost_indices]}
        trip_to_vehicle_cost_map:  {trip_no: [trip_cost_indices]}
        requests:                  request batch for this iteration
        active_requests:           already-accepted requests that must be served
        config:                    solver configuration

    Returns:
        oracle: Callable[[Tensor (n,)], Tensor (n,)]
            Accepts a score vector (with or without perturbation) and returns a
            binary assignment vector y ∈ {0,1}^n.
            Safe to call under torch.no_grad(); does not track gradients.

    Note:
        Each call to oracle() solves one ILP.  When used inside
        FenchelYoungLoss with num_samples=k, this means k ILP solves per
        training step.  For large instances consider reducing num_samples.
    """
    trip_cost_count = len(trip_costs)

    solver = CO_ScoreMaximization(
        single_trip_map,
        trips,
        trip_costs,
        vehicle_to_trips_cost_map,
        trip_to_vehicle_cost_map,
        config,
    )

    def oracle(scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (n,) float tensor — perturbed scores, no grad required.

        Returns:
            y: (n,) float32 tensor with values in {0.0, 1.0}.
               All zeros if the ILP did not reach OPTIMAL or SUBOPTIMAL.
        """
        scores_np = scores.detach().cpu().numpy().astype(float)

        gurobi_model, x_t, _x_r = solver.solve_ilp(
            scores_np,
            requests,
            active_requests,
            keep_active=config.KEEP_ACTIVE,
        )
        return extract_y_binary(gurobi_model, x_t, trip_cost_count)

    return oracle


def extract_y_binary(
    gurobi_model,
    x_t,
    trip_cost_count: int,
) -> torch.Tensor:
    """
    Convert the solved Gurobi trip-selection variables into a binary tensor.

    This is exposed as a standalone function so that coaml_pipeline_solver can
    reuse it to build y_star from the CO_TripCostMinimization solution without
    solving the ILP a second time.

    Args:
        gurobi_model:    Solved gurobi.Model instance.
        x_t:             Gurobi variable dict {i: Var} for i in range(trip_cost_count).
        trip_cost_count: Number of trip-cost entries (= len(trip_costs)).

    Returns:
        y: float32 Tensor of shape (trip_cost_count,) with values in {0.0, 1.0}.
           Returns a zero vector if the model status is not OPTIMAL or SUBOPTIMAL.
    """
    y = torch.zeros(trip_cost_count, dtype=torch.float32)
    if gurobi_model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        for i in range(trip_cost_count):
            y[i] = float(x_t[i].X)
    return y
