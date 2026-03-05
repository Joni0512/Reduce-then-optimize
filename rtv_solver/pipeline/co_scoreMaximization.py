import gurobipy as gp
from gurobipy import GRB
import numpy as np

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
    def __init__(self, config: Config):
        super().__init__(config)    
    
    def run(self, feature_scores, requests: list[Request], active_requests: dict[int, Request]) -> AssignmentResult:
        model, x_t, x_r, _x_reject = self.solve_ilp(
            feature_scores, requests, active_requests, keep_active=self.config.KEEP_ACTIVE
        )
        assignment_result = self.transform_solution_to_assignment(model, x_t, x_r, requests)
        return assignment_result
        
    def solve_ilp(
        self,
        feature_scores,
        requests: list[Request],
        active_requests: dict[int, Request],
        penalty: int = 100_000,
        keep_active: bool = True,
        reject_action_scores: np.ndarray | None = None,
        reject_vehicle_ids: list[int] | None = None,
    ):
        # TODO add docstring
        trip_count = len(self.trip_costs)
        request_count = len(requests)

        assert trip_count == feature_scores.shape[0]
        if reject_action_scores is not None:
            if reject_vehicle_ids is None:
                raise ValueError("reject_vehicle_ids are required when reject_action_scores are provided.")
            if len(reject_action_scores) != len(reject_vehicle_ids):
                raise ValueError("reject_action_scores length must match reject_vehicle_ids length.")
        else:
            # Backward-compatible path: solve with trip variables only.
            reject_vehicle_ids = []

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
            x_reject = {}
            if reject_action_scores is not None and len(reject_vehicle_ids) > 0:
                for idx, vehicle_id in enumerate(reject_vehicle_ids):
                    # One explicit reject-action variable per vehicle.
                    x_reject[vehicle_id] = model.addVar(
                        lb=0,
                        ub=1,
                        obj=float(reject_action_scores[idx]),
                        name=f"rej_{vehicle_id}",
                        vtype=GRB.BINARY,
                    )

            model.ModelSense = GRB.MAXIMIZE  # maximize service quality
            
            # constraint: each vehicle has at most on trip
            for vehicle_id in self.vehicle_to_trips_cost_map.keys():
                vehicle_trip_sum = gp.quicksum(
                    x_t[i] for i in self.vehicle_to_trips_cost_map[vehicle_id]
                )
                if vehicle_id in x_reject:
                    # With reject-action enabled, force one explicit action per vehicle.
                    # This lets the model trade off "assign a trip" vs "reject for this vehicle".
                    model.addConstr(vehicle_trip_sum + x_reject[vehicle_id] == 1, name=f"veh_{vehicle_id}")
                else:
                    # Legacy behavior if no reject action is provided.
                    model.addConstr(vehicle_trip_sum <= 1, name=f"veh_{vehicle_id}")

            # constraint: each request is either rejected or served by a single trip 
            # active requests are handled with extra care
            for request_no, request in enumerate(requests):
                trip_no = self.single_trip_map[request.id]
                cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]

                model.addConstr(x_r[request_no] + gp.quicksum(x_t[i] for i in cost_map_indices) == 1, "req_{0}".format(request.id))
                
                # all the previously assigned requests must be picked up
                if request.id in active_requests and keep_active:
                    model.addConstr(x_r[request_no] == 0, "active_req_{0}".format(request.id))

            model.setParam('TimeLimit', self.config.ILP_TIMEOUT)
            model.optimize()

            return model, x_t, x_r, x_reject