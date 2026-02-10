from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import gurobipy as gp
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
    """
    # not yet sure whether an ABC actually makes sense here as we rather need different optimizers for different goals (different scores, objectives, etc.) but we might keep the constraints; we will definitely not change the solver code in the backend as it is too cumbersome for now
    def __init__(self, config: Config):
        self.config = config

    # @abstractmethod 
    # NOTE not sure whether the interface is fixed and not required for now 
    # def run(self):
    #     raise NotImplementedError


class COTripCostMinimization(CO):
    """
    Combinatorial Optimization Layer that minimizes the TripCosts across all the vehicles.
    """
    def __init__(self, 
                 config, 
                 single_trip_map: dict[int, int], 
                 trips: list[Trip], 
                 trip_costs: list[TripCost], 
                 vehicle_to_trips_cost_map: dict[int, list[int]], 
                 trip_to_vehicle_cost_map: dict[int, list[int]]):
        super().__init__(config)
        self.single_trip_map = single_trip_map                              # {request_id: trip_id}
        self.trips = trips              
        self.trip_costs = trip_costs
        self.vehicle_to_trips_cost_map = vehicle_to_trips_cost_map  # {vehicle_id: [trip_cost_index]}
        self.trip_to_vehicle_cost_map = trip_to_vehicle_cost_map    # {trip_id: [trip_cost_index]}

    def run(self, requests: list[Request], active_requests: dict[int, Request]):
        model, x_t, x_r = self.solve_ilp(requests, active_requests, penalty=self.config.ilp_penalty, keep_active=self.config.keep_active)
        assignment_result = self.transform_solution_to_assignment(model, x_t, x_r, requests)
        return assignment_result

    def solve_ilp(self, requests: list[Request], active_requests: dict[int, Request], penalty: int = 100_000, keep_active: bool = True):
        """
        Build and immediatelty solves ILP of trip_costs for associated requests and vehicles

        :param list requests: Requests that are considered in this method call.
        :param list active_requests: Requests that have been accepted in prior iterations and that need to be kept based on the keep_active bool.
        :param bool keep_active: To get the actual best result, we do not care about what has been previously accepted. Prior iterations only influence the solutions by already boarded requests. If a new combination becomes better, we do not want to be constrained by trips that have been accepted because the solver saw only a partial (earlier picture) and it should not be stuck with previously selected requests.
        """
        # NOTE If no active vehicles are available, it will still output logs as no optimization is possible. This only occurs if the vehicles are basically offline as the end_time of their operation has finished.
        trip_count = len(self.trip_costs)
        request_count = len(requests)

        console_logger.debug("Started building optimization problem")
        # setup Integer Linear Program 
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            model = gp.Model('RTV assignment - Service rate + Minimum distance', env=env)
            model.Params.OutputFlag = 0
            
            # define trip variables with related costs
            trip_costs_obj = np.fromiter((tc.cost for tc in self.trip_costs), dtype=float, count=trip_count)
            x_t = model.addVars(trip_count,
                            lb=0,
                            ub=1,
                            obj=trip_costs_obj,
                            name="t", 
                            vtype=GRB.BINARY)
            
            # create penalties per request
            request_ids = np.array([r.id for r in requests])
            priorities  = np.array([r.priority for r in requests])
            penalties = priorities.copy()
            if keep_active:
                penalties[np.isin(request_ids, list(active_requests))] = 100
            x_r = model.addVars(request_count,
                            lb=0,
                            ub=1,
                            obj=penalties * penalty,
                            name="r", 
                            vtype=GRB.BINARY)

            # constraint: each vehicle has at most on trip
            model.addConstrs((gp.quicksum(x_t[i] for i in self.vehicle_to_trips_cost_map[vehicle_id]) <= 1 for vehicle_id in list(self.vehicle_to_trips_cost_map.keys())), "veh")

            # constraint: each request is either rejected or served by a single trip 
            # active requests are handled with extra care
            request_no = 0
            for request in requests:
                trip_no = self.single_trip_map[request.id]
                cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]

                model.addConstr(x_r[request_no]+gp.quicksum(x_t[i] for i in cost_map_indices) == 1, "req_{0}".format(request.id))
                
                # all the previously assigned requests should be picked up
                if request.id in active_requests and keep_active:
                    model.addConstr(x_r[request_no] == 0, "active_req_{0}".format(request.id))
                request_no+=1

            model.setParam('TimeLimit', self.config.ilp_timeout)
            model.optimize()

            return model, x_t, x_r

    def transform_solution_to_assignment(self, model, x_t, x_r, requests: list[Request]) -> 'AssignmentResult':

        # extract solution from Gurobi assignment
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
        model.write(self.config.output_dir / "infeasible.ilp")   # human-readable
        model.write(self.config.output_dir / "infeasible.lp")    # full model
        model.write(self.config.output_dir / "infeasible.mps")   # optional

        # Print which constraints are in IIS
        console_logger.error("\n--- IIS constraints ---")
        for constraint in model.getConstrs():
            if constraint.IISConstr:
                console_logger.error("IIS:", constraint.ConstrName)
        raise Exception(f"Gurobi solver ended with code: {model.Status}") # Code 3 INFEASIBLE


class CORebalancingHeuristic(CO):
    # TODO move rebalancing logic from TripHandler
    def __init__(self, config):
        super().__init__(config)
        raise NotImplementedError