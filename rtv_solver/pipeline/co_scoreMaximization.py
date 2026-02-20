import gurobipy as gp
from gurobipy import GRB

from rtv_solver.pipeline import CO

from rtv_solver.structure.config import Config
from rtv_solver.structure.request import Request
from rtv_solver.structure.assignment_result import AssignmentResult
from rtv_solver.structure.trip import Trip
from rtv_solver.structure.trip_cost import TripCost

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

class CO_ScoreMaximization(CO):
    def __init__(self,
                 single_trip_map: dict[int, int], 
                 trips: list[Trip], 
                 trip_costs: list[TripCost], 
                 vehicle_to_trips_cost_map: dict[int, list[int]], 
                 trip_to_vehicle_cost_map: dict[int, list[int]],
                 config: Config):
        super().__init__(single_trip_map, trips, trip_costs, vehicle_to_trips_cost_map, trip_to_vehicle_cost_map, config)

    def run(self, feature_scores, requests: list[Request], active_requests: dict[int, Request]) -> AssignmentResult:
        model, x_t, x_r = self.solve_ilp(feature_scores, requests, active_requests, keep_active=self.config.KEEP_ACTIVE)
        assignment_result = self.transform_solution_to_assignment(model, x_t, x_r, requests)
        return assignment_result
        
    def solve_ilp(self, feature_scores, requests: list[Request], active_requests: dict[int, Request], keep_active: bool = True):
        """
        """
        trip_count = len(self.trip_costs)
        request_count = len(requests)

        assert trip_count == feature_scores.shape[0]

        console_logger.debug("Started building optimization problem")
        # setup Integer Linear Program 
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            model = gp.Model('RTV assignment - Service rate + Minimum distance', env=env)
            model.Params.OutputFlag = 0

            # TODO implement the ILP solver for the score maximization
            x_t = model.addVars(trip_count,
                                lb=0,
                                ub=1,
                                obj=feature_scores,  # positive numbers
                                name="t",
                                vtype=GRB.BINARY)

            x_r = model.addVars(request_count,
                                lb=0,
                                ub=1,
                                obj=0,  # rejection has no effect on objective
                                name="r",
                                vtype=GRB.BINARY)

            model.ModelSense = GRB.MAXIMIZE  # maximize service quality
            
            # constraint: each vehicle has at most on trip
            model.addConstrs(
                (gp.quicksum(x_t[i] for i in self.vehicle_to_trips_cost_map[vehicle_id]) <= 1
                for vehicle_id in self.vehicle_to_trips_cost_map.keys()),
                name=f"veh"
            )

            # constraint: each request is either rejected or served by a single trip 
            # active requests are handled with extra care
            for request_no, request in enumerate(requests):
                trip_no = self.single_trip_map[request.id]
                cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]

                model.addConstr(x_r[request_no]+gp.quicksum(x_t[i] for i in cost_map_indices) == 1, "req_{0}".format(request.id))
                
                # all the previously assigned requests must be picked up
                if request.id in active_requests and keep_active:
                    model.addConstr(x_r[request_no] == 0, "active_req_{0}".format(request.id))

            model.setParam('TimeLimit', self.config.ILP_TIMEOUT)
            model.optimize()

            return model, x_t, x_r