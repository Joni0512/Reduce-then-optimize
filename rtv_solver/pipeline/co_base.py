from abc import ABC, abstractmethod
from gurobipy import GRB

from rtv_solver.structure.config import Config
from rtv_solver.structure.request import Request
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.trip import Trip

from rtv_solver.structure.assignment_result import AssignmentResult

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
                 single_trip_map: dict[int, int], 
                 trips: list[Trip], 
                 trip_costs: list[TripCost], 
                 vehicle_to_trips_cost_map: dict[int, list[int]], 
                 trip_to_vehicle_cost_map: dict[int, list[int]],
                 config: Config):
        self.single_trip_map = single_trip_map                              # {request_id: trip_id}
        self.trips = trips              
        self.trip_costs = trip_costs
        self.vehicle_to_trips_cost_map = vehicle_to_trips_cost_map  # {vehicle_id: [trip_cost_index]}
        self.trip_to_vehicle_cost_map = trip_to_vehicle_cost_map    # {trip_id: [trip_cost_index]}
        self.config = config

    @abstractmethod 
    def run() -> AssignmentResult:
        raise NotImplementedError

    def transform_solution_to_assignment(self, model, x_t, x_r, requests: list[Request]) -> 'AssignmentResult':
        """
        Decode and extract assignment solution from Gurobi solution
        
        If we keep the constraints and the basic structure the same, this function should always be able to work even when we use a different objective.
        TODO move to a separate decoder, but no priority.
        """
        vehicle_assignment = {}
        request_assignment = {}
        trip_sizes = []
        unassigned_trip_count = 0
        trip_count = 0
        added_distance = 0

        request_count = len(requests)

        if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
            console_logger.info("Total time spent on optimization: {0}".format(model.Runtime))

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
                        console_logger.info(f"Assignment: {trip_cost}")

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
            
        console_logger.info(f'Assignment: new requests / unassigned / assigned: {request_count} / {unassigned_trip_count} / {trip_count}')
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