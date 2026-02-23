import numpy as np
import gurobipy as gp
from gurobipy import GRB

from rtv_solver.structure.config import Config
from rtv_solver.structure.request import Request
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.trip import Trip

from rtv_solver.pipeline import CO

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)


class CO_TripCostMinimization(CO):
    """
    Combinatorial Optimization Layer that minimizes the TripCosts across all the vehicles.
    """
    def __init__(self, config: Config):
        super().__init__(config)

    def run(self, requests: list[Request], active_requests: dict[int, Request]):
        # define trip variables with related costs
        trip_costs_obj = np.fromiter((tc.cost for tc in self.trip_costs), dtype=float, count=len(self.trip_costs))
        model, x_t, x_r = self.solve_ilp(trip_costs_obj, requests, active_requests, penalty=self.config.ILP_PENALTY, keep_active=self.config.KEEP_ACTIVE)
        assignment_result = self.transform_solution_to_assignment(model, x_t, x_r, requests)
        return assignment_result

    def solve_ilp(self, trip_objective_scores: np.ndarray, requests: list[Request], active_requests: dict[int, Request], penalty: int = 100_000, keep_active: bool = True):
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
            
            x_t = model.addVars(trip_count,
                            lb=0,
                            ub=1,
                            obj=trip_objective_scores, # moved this out so we can move the creation of the scores to an NN model
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

            model.setParam('TimeLimit', self.config.ILP_TIMEOUT)
            model.optimize()

            return model, x_t, x_r