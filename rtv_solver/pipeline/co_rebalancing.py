import numpy as np
import gurobipy as gp
from gurobipy import GRB

from rtv_solver.pipeline import CO

from rtv_solver.handlers.vehicle_handler import VehicleHandler

from rtv_solver.structure.request import Request
from rtv_solver.structure.vehicle import Vehicle
from rtv_solver.structure.assignment_result import AssignmentResult

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)


class CO_RebalancingCoverage(CO):
    # NOT TESTED and uncertain whether VehicleHandler can still calculate the rebalancing-distances, but in the refactoring (10.02.2026) wanted to clean up the tripHandler class in order to facilitate COAML integration
    def __init__(self, config):
        super().__init__(config)

    def run(self, result: AssignmentResult, vehicles: dict[int, Vehicle], requests: list[Request]) -> AssignmentResult:
        """ 
        Idling, empty vehicles can be reallocated to the origins of current requests that have not been covered
        
        Assumption: rejected requests are in underserved areas, so we need to send additional vehicles there. 
        """
        # initiliaze prior information
        empty_vehicles = []
        for vehicle_id in vehicles:
            if vehicle_id not in result.vehicle_assignment:
                vehicle = vehicles[vehicle_id]
                if not (vehicle.rebalancing or len(vehicle.stop_sequence)>0):
                    empty_vehicles.append(vehicle_id)
                
        unassigned_requests = []
        for request in requests:
            if request.id not in result.request_assignment:
                unassigned_requests.append(request)

        number_of_vehicles = len(empty_vehicles)
        number_of_requests = len(unassigned_requests)
        max_rebalancing_count = min(number_of_vehicles, number_of_requests) 

        rebalancing_assignment = {}

        if max_rebalancing_count > 0:
            model = gp.Model('Rebalancing')
            rebalancing_costs = np.zeros((number_of_vehicles,number_of_requests))
            for i in range(number_of_vehicles):
                vehicle = vehicles[empty_vehicles[i]]
                for j in range(number_of_requests):
                    # get origin for unassigned requests and costs for idling vehicles to get to that position
                    origin = unassigned_requests[j].origin
                    rebalancing_costs[i][j] = VehicleHandler.cost_of_rebalancing(vehicle, origin) # if this does not work, instantiate
            y_vr = model.addVars(number_of_vehicles, number_of_requests, lb=0, ub=1, obj=rebalancing_costs, name="y_vr", vtype=GRB.BINARY)

            model.addConstrs((y_vr.sum(i,'*') <= 1 for i in range(number_of_vehicles)), "veh")
            model.addConstrs((y_vr.sum('*',j) <= 1 for j in range(number_of_requests)), "req")
            model.addConstr((y_vr.sum() <= max_rebalancing_count), "total_assignment")
            model.optimize()

            # process optimization result
            if model.Status == GRB.OPTIMAL:
                for i in range(number_of_vehicles):
                    for j in range(number_of_requests):
                        if y_vr[i,j].X == 1:
                            vehicle_id = empty_vehicles[i]
                            origin = unassigned_requests[j].origin
                            rebalancing_assignment[vehicle_id] = origin
                            break

        # update assignmentResult
        result.rebalancing_assignment = rebalancing_assignment
        return result
