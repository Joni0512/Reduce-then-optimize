from abc import ABC, abstractmethod
from gurobipy import GRB
import gurobipy as gp

from rtv_solver.structure.config import Config
from rtv_solver.structure.request import Request
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.trip import Trip

from rtv_solver.structure.assignment_result import AssignmentResult
import torch

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)


class CO(ABC):
    """
    Separation of concerns for the combinatorial optimization layer. The result must always be a clear assignment to the vehicles independent of the score representation or other values.
    - defines the interface for solving specific ILPs
    - defines how solutions are transformed into assignments
    - turns model results into assignments as the structure will remain and the different optimizers might use different scores and objectives
    """
    # not yet sure whether an ABC actually makes sense here as we rather need different optimizers for different goals (different scores, objectives, etc.) but we might keep the constraints; we will definitely not change the solver code in the backend as it is too cumbersome for now
    def __init__(self, 
                 config: Config):
        self.single_trip_map = {}               # {request_id: trip_id}
        self.trips = []       
        self.trip_costs = []
        self.vehicle_to_trips_cost_map = {}     # {vehicle_id: [trip_cost_index]}
        self.trip_to_vehicle_cost_map = {}      # {trip_id: [trip_cost_index]}
        self.config = config

    def reset(self, single_trip_map: dict[int, int], trips: list[Trip], trip_costs: list[TripCost], vehicle_to_trips_cost_map: dict[int, list[int]], trip_to_vehicle_cost_map: dict[int, list[int]]):
        """
        Reset the combinatorial optimizer to a new problem instance for each iteration and in order to implement the interfaces for injection patterns.
        """
        self.single_trip_map = single_trip_map
        self.trips = trips
        self.trip_costs = trip_costs
        self.vehicle_to_trips_cost_map = vehicle_to_trips_cost_map
        self.trip_to_vehicle_cost_map = trip_to_vehicle_cost_map

    @abstractmethod
    def solve_ilp() -> tuple[gp.Model, dict[int, gp.Var], dict[int, gp.Var]]:
        raise NotImplementedError

    @abstractmethod 
    def run() -> AssignmentResult:
        raise NotImplementedError

    def transform_solution_to_assignment(
        self,
        model,
        x_t,
        x_r,
        requests: list[Request],
        x_reject: dict[int, gp.Var] | None = None,
        reject_vehicle_ids: list[int] | None = None,
    ) -> 'AssignmentResult':
        """
        Decode and extract assignment solution from Gurobi solution
        
        If we keep the constraints and the basic structure the same, this function should always be able to work even when we use a different objective.
        """
        vehicle_assignment = {}
        request_assignment = {}
        trip_sizes = []
        unassigned_trip_count = 0
        trip_count = 0
        added_distance = 0

        request_count = len(requests)
        reject_vehicle_ids = reject_vehicle_ids or []
        x_reject = x_reject or {}

        if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
            console_logger.info(f"[{self.__class__.__name__}] ILP solved in {model.Runtime:.3f} s.")

            for vehicle_id in self.vehicle_to_trips_cost_map:
                for i in self.vehicle_to_trips_cost_map[vehicle_id]:
                    if x_t[i].X == 1:
                        trip_cost = self.trip_costs[i]
                        added_distance += trip_cost.cost
                        trip_no = trip_cost.trip_no
                        trip = self.trips[trip_no]
                        trips = []
                        if isinstance(trip, Trip):
                            trips.append(trip)
                        else: # instance is SharedTrip
                            for sub_trip_no in trip.trips:
                                trips.append(self.trips[sub_trip_no])
                        trip_sizes.append(len(trips))
                        vehicle_assignment[vehicle_id] = (trips, trip_cost.sequence)
                        console_logger.debug(f"[{self.__class__.__name__}] Assignment: {trip_cost}")
                        console_logger.info(f"[{self.__class__.__name__}] Assignment: {trip_cost.simple_str()}")

                # Optional reject-action sanity check for CO variants that expose x_reject.
                if vehicle_id in reject_vehicle_ids and vehicle_id in x_reject:
                    if x_reject[vehicle_id].X == 1 and vehicle_id in vehicle_assignment:
                        raise ValueError(
                            f"Inconsistent solution for vehicle {vehicle_id}: reject and trip selected."
                        )

            for request in requests:
                found_assignment = False
                trip_no = self.single_trip_map[request.id]
                cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]
                for index in cost_map_indices:
                    if x_t[index].X == 1:
                        trip_cost = self.trip_costs[index]
                        vehicle_id = trip_cost.vehicle_id
                        request_assignment[request.id] = vehicle_id
                        found_assignment = True
                        trip_count +=1
                        break

                if not found_assignment:
                    unassigned_trip_count += 1
        else:
            unassigned_trip_count = request_count
            self._handle_infeasibility(model)
            
        console_logger.info(f'[{self.__class__.__name__}] Assignment: new requests / unassigned / assigned: {request_count} / {unassigned_trip_count} / {trip_count}')
        # TODO make information better, some requests are re-assigned although they were already assigned
        return AssignmentResult(
            vehicle_assignment,
            request_assignment,
            {},
            unassigned_trip_count,
            trip_count,
            added_distance,
            trip_sizes,
            model.Status,
            model.Runtime
        )

    def transform_optimal_solution_to_assignment(
        self,
        y_star: torch.Tensor,
        requests: list[Request],
        reject_vehicle_ids: list[int] | None = None,
    ) -> 'AssignmentResult':
        """
        Decode an externally provided binary assignment vector `y_star`.

        Expected layout:
        - first `len(self.trip_costs)` entries correspond to trip-cost choices
        - optional tail entries correspond to reject actions in the order of `reject_vehicle_ids`

        Preconditions (strict):
        - y_star is 1D and binary {0,1}
        - if reject actions are provided: one action per vehicle (selected trip OR selected reject), therefore total selected actions equals number of vehicles.
        """
        reject_vehicle_ids = reject_vehicle_ids or []
        trip_cost_count = len(self.trip_costs)
        vehicle_ids = list(self.vehicle_to_trips_cost_map.keys())
        vehicle_id_set = set(vehicle_ids)

        if y_star.ndim != 1:
            raise ValueError(f"y_star must be 1D, got shape {tuple(y_star.shape)}")

        expected_size = trip_cost_count + len(reject_vehicle_ids)
        if y_star.shape[0] != expected_size:
            raise ValueError(
                f"y_star has length {y_star.shape[0]}, expected {expected_size} "
                f"(trip_costs={trip_cost_count} + reject_actions={len(reject_vehicle_ids)})."
            )

        y = y_star.detach().cpu().to(torch.float32)
        if not torch.all((y == 0) | (y == 1)):
            raise ValueError("y_star must be binary (0/1).")

        y_trip = y[:trip_cost_count] # similar to x_t in the transform_solution_to_assignment method
        y_reject = y[trip_cost_count:]

        if reject_vehicle_ids:
            unknown_vehicles = set(reject_vehicle_ids) - vehicle_id_set
            if unknown_vehicles:
                raise ValueError(
                    f"reject_vehicle_ids contains unknown vehicle IDs: {sorted(unknown_vehicles)}"
                )

        selected_trip_indices = [i for i in range(trip_cost_count) if y_trip[i].item() == 1.0]
        selected_reject_by_vehicle = {
            vehicle_id: int(y_reject[idx].item())
            for idx, vehicle_id in enumerate(reject_vehicle_ids)
        }

        # Per-vehicle action consistency.
        for vehicle_id in vehicle_ids:
            selected_trip_count = sum(
                1
                for tc_idx in self.vehicle_to_trips_cost_map[vehicle_id]
                if y_trip[tc_idx].item() == 1.0
            )
            selected_reject = selected_reject_by_vehicle.get(vehicle_id, 0)
            total_actions = selected_trip_count + selected_reject

            if reject_vehicle_ids:
                if total_actions != 1:
                    raise ValueError(
                        f"Vehicle {vehicle_id} has {total_actions} selected actions; expected exactly 1."
                    )
            else:
                if total_actions > 1:
                    raise ValueError(
                        f"Vehicle {vehicle_id} has {total_actions} selected trip actions; expected at most 1."
                    )

        if reject_vehicle_ids:
            total_selected_actions = int(y_trip.sum().item() + y_reject.sum().item())
            if total_selected_actions != len(vehicle_ids):
                raise ValueError(
                    f"Selected actions {total_selected_actions} must equal vehicle count {len(vehicle_ids)}."
                )

        vehicle_assignment: dict[int, tuple[list[Trip], list[int]]] = {}
        request_assignment: dict[int, int] = {}
        trip_sizes: list[int] = []
        added_distance = 0.0

        for idx in selected_trip_indices:
            trip_cost = self.trip_costs[idx]
            trip_vehicle_id = trip_cost.vehicle_id
            trip = self.trips[trip_cost.trip_no]

            trips = []
            if isinstance(trip, Trip):
                trips.append(trip)
            else:  # SharedTrip
                for sub_trip_no in trip.trips:
                    trips.append(self.trips[sub_trip_no])

            vehicle_assignment[trip_vehicle_id] = (trips, trip_cost.sequence)
            trip_sizes.append(len(trips))
            added_distance += float(trip_cost.cost)

            for assigned_trip in trips:
                if assigned_trip.request_id in request_assignment:
                    raise ValueError(
                        f"Request {assigned_trip.request_id} assigned multiple times in y_star."
                    )
                request_assignment[assigned_trip.request_id] = trip_vehicle_id

        request_count = len(requests)
        trip_count = len(request_assignment)
        unassigned_trip_count = request_count - trip_count

        console_logger.info(
            f'[{self.__class__.__name__}] Optimal assignment: new requests / unassigned / assigned: '
            f'{request_count} / {unassigned_trip_count} / {trip_count}'
        )
        return AssignmentResult(
            vehicle_assignment,
            request_assignment,
            {},
            unassigned_trip_count,
            trip_count,
            added_distance,
            trip_sizes,
            GRB.OPTIMAL,
            None,
        )

    def _handle_infeasibility(self, model):
        # Compute IIS (conflicting constraints)
        model.Params.OutputFlag = 1
        model.computeIIS()
        model.write(self.config.OUTPUT_DIR / "infeasible.ilp")   # human-readable
        model.write(self.config.OUTPUT_DIR / "infeasible.lp")    # full model
        model.write(self.config.OUTPUT_DIR / "infeasible.mps")   # optional

        # Print which constraints are in IIS
        console_logger.error("\n--- IIS constraints ---")
        for constraint in model.getConstrs():
            if constraint.IISConstr:
                console_logger.error("IIS:", constraint.ConstrName)
        raise Exception(f"Gurobi solver ended with code: {model.Status}") # Code 3 INFEASIBLE

    @staticmethod
    def extract_y_binary(
            gurobi_model,
            x_t: dict[int, gp.Var],
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
        trip_cost_count = len(x_t)
        y = torch.zeros(trip_cost_count, dtype=torch.float32)
        if gurobi_model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            for i in range(trip_cost_count):
                y[i] = float(x_t[i].X)
        return y