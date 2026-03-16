import copy
import json
import math
from pathlib import Path
import numpy as np

from rtv_solver.parser.base_parser import BaseParser


class LiLimParser(BaseParser):
    """
    Parser for Li & Lim PDPTW benchmark instances.
    https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/

    Format:
    Line 1: K (vehicles), Q (capacity), S (speed - not used)
    Line 2+: TASK_NO, X, Y, DEMAND, EARLIEST_TIME, LATEST_TIME, SERVICE_TIME, PICKUP_INDEX, DELIVERY_INDEX
    Example Line: 3	42	66	10	65	146	90	0	75
    Interpretation: Task 3 is a pickup at (42, 66) with demand 10, earliest time 65, latest time 146, service time 90, pickup index 0, delivery index 75.

    Task 0 is the depot.
    For pickups: PICKUP_INDEX=0, DELIVERY_INDEX=corresponding delivery task index
    For deliveries: PICKUP_INDEX=corresponding pickup task index, DELIVERY_INDEX=0
    """
    @staticmethod
    def parse_file(filepath: str):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header = lines[0].strip().split('\t')
        num_vehicles = int(header[0])
        vehicle_capacity = int(header[1])
        depot_info = lines[1].strip().split('\t')
        depot_start_time = int(depot_info[4])
        depot_end_time = int(depot_info[5])
        depot_loc = {'lon': float(depot_info[1]), 'lat': float(depot_info[2]), 'node_id': 0}

        driver_runs = []
        for i in range(num_vehicles):
            driver_runs.append({
                'state': {
                    'run_id': i,
                    'start_time': depot_start_time,
                    'end_time': depot_end_time,
                    'am_capacity': vehicle_capacity,
                    'wc_capacity': 0, # we do not do wheelchair requests in this solver
                    'locations_already_serviced': 0,
                    'location_dt_seconds': 0,
                    'loc': copy.deepcopy(depot_loc),
                    'total_locations': 0
                },
                'manifest': [
                ]
            })
    
        # Parse tasks
        tasks = {}
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            task_no = int(parts[0])
            tasks[task_no] = {
                'task_no': task_no,
                'x': float(parts[1]),
                'y': float(parts[2]),
                'demand': int(parts[3]),
                'earliest': int(parts[4]),
                'latest': int(parts[5]),
                'service_time': int(parts[6]),
                'pickup_idx': int(parts[7]),
                'delivery_idx': int(parts[8])
            }
        
        # Task 0 is the depot
        depot_task = tasks[0]
        depot = {'pt': {'lon': depot_task['x'], 'lat': depot_task['y']}, 'node_id': 0}
        
        # Find pickup-delivery pairs
        # Pickups have pickup_idx=0 and delivery_idx>0
        # Deliveries have pickup_idx>0 and delivery_idx=0
        requests = []
        processed = set()
        
        for task_no, task in tasks.items():
            if task_no == 0:  # Skip depot
                continue
            if task_no in processed:
                continue
                
            # Check if this is a pickup (pickup_idx=0, delivery_idx>0)
            if task['pickup_idx'] == 0 and task['delivery_idx'] > 0:
                pickup_task = task
                delivery_task_no = task['delivery_idx']
                delivery_task = tasks[delivery_task_no]
                
                # Mark both as processed
                processed.add(task_no)
                processed.add(delivery_task_no)
                
                # Create request
                request = {
                    'booking_id': task_no,
                    'pickup_pt': {
                        'lon': pickup_task['x'],
                        'lat': pickup_task['y'],
                        'node_id': pickup_task['task_no']
                    },
                    'dropoff_pt': {
                        'lon': delivery_task['x'],
                        'lat': delivery_task['y'],
                        'node_id': delivery_task['task_no']
                    },
                    'pickup_time_window_start': pickup_task['earliest'],
                    'pickup_time_window_end': pickup_task['latest'],
                    'dropoff_time_window_start': delivery_task['earliest'],
                    'dropoff_time_window_end': delivery_task['latest'],
                    'am': abs(pickup_task['demand']),  # Use absolute value of demand
                    'wc': 0,
                    # Additional fields that might be useful
                    'pickup_service_time': pickup_task['service_time'],
                    'dropoff_service_time': delivery_task['service_time']
                }
                requests.append(request)
        
        # order requests by pickup_time_window_start
        requests = sorted(requests, key=lambda r: r['pickup_time_window_start'])

        # build the travel time matrix (using Euclidean distance as a proxy for travel time)
        num_tasks = len(tasks)
        travel_time_matrix = np.zeros((num_tasks, num_tasks))

        for i, task_i in tasks.items():
            for j, task_j in tasks.items():
                if i == j:
                    travel_time_matrix[i][j] = 0
                else:
                    # travel time matrix is just the euclidean distance between two tasks
                    distance = np.sqrt((task_i['x'] - task_j['x'])**2 + (task_i['y'] - task_j['y'])**2)
                    travel_time_matrix[i][j] = distance

        # Normalize travel time matrix by vehicle speed (if needed)
        # Assuming speed is 1 unit per time for now
        travel_time_matrix = travel_time_matrix.tolist()
        return {
            "requests": requests, 
            "depot": depot, 
            "driver_runs": driver_runs, 
            "travel_time_matrix": travel_time_matrix
        }

    @staticmethod
    def parse_solution(original_filepath, solution_filepath):
        """
        We want to parse the solution file in order to have 'optimal' solution for the entire period.

        Args:
            original_filepath: Path to the original file
            solution_filepath: Path to the solution file

        Returns:
            dict with keys: requests, depot, driver_runs, travel_time_matrix

        Format: The tasks are considered to be completed in the order of the solution file. Each route is one vehicle with a sequence of tasks and no dwell time.
        

        Example File:
            Instance name :	lc101
            Authors       :	Geir Hasle Oddvar Kloster
            Date          : 11-mar-03
            Reference     :	Chapter in Hasle G. K-A Lie E. Quak (eds): Geometric Modelling Numerical Simulation and Optimization. ISBN 978-3-540-68782-5 Springer 2007.
            Solution
            Route 1 : 81 78 104 76 71 70 73 77 79 80
            Route 2 : 57 55 54 53 56 58 60 59
            Route 3 : 98 96 95 94 92 93 97 106 100 99
            Route 4 : 13 17 18 19 15 16 14 12
            Route 5 : 32 33 31 35 37 38 39 36 105 34
            Route 6 : 90 87 86 83 82 84 85 88 89 91
            Route 7 : 43 42 41 40 44 46 45 48 51 101 50 52 49 47
            Route 8 : 67 65 63 62 74 72 61 64 102 68 66 69
            Route 9 : 5 3 7 8 10 11 9 6 4 2 1 75
            Route 10 : 20 24 25 27 29 30 28 26 23 103 22 21
        """
        # Step 1: Parse the original instance to reuse:
        # - request definitions (pickup/dropoff mapping, windows, demand)
        # - depot information
        # - travel time matrix
        original_payload = LiLimParser.parse_file(original_filepath)
        requests = original_payload["requests"]
        depot = original_payload["depot"]
        travel_time_matrix = original_payload["travel_time_matrix"]

        # Step 2: Build a fast lookup from task id -> stop metadata.
        # The Li&Lim solution file routes list task ids directly, so we need to resolve
        # each task into either a pickup or dropoff stop plus all manifest attributes.
        stop_by_task_id = {}
        for request in requests:
            booking_id = request["booking_id"]
            pickup_task_id = request["pickup_pt"]["node_id"]
            dropoff_task_id = request["dropoff_pt"]["node_id"]
            am = request["am"]
            pickup_service_time = float(request.get("pickup_service_time", 0))
            dropoff_service_time = float(request.get("dropoff_service_time", 0))

            stop_by_task_id[pickup_task_id] = {
                "booking_id": booking_id,
                "action": "pickup",
                "loc": copy.deepcopy(request["pickup_pt"]),
                "am": am,
                "wc": 0,
                "time_window_start": request["pickup_time_window_start"],
                "time_window_end": request["pickup_time_window_end"],
                "dwell": pickup_service_time,
            }
            stop_by_task_id[dropoff_task_id] = {
                "booking_id": booking_id,
                "action": "dropoff",
                "loc": copy.deepcopy(request["dropoff_pt"]),
                "am": am,
                "wc": 0,
                "time_window_start": request["dropoff_time_window_start"],
                "time_window_end": request["dropoff_time_window_end"],
                "dwell": dropoff_service_time,
            }

        # Step 3: Read the solution file and extract the explicit routes.
        with open(solution_filepath, "r") as f:
            solution_lines = f.readlines()

        routes = []
        for line in solution_lines:
            stripped = line.strip()
            if not stripped.startswith("Route"):
                continue
            if ":" not in stripped:
                continue
            route_part = stripped.split(":", 1)[1].strip()
            if route_part:
                task_sequence = [int(token) for token in route_part.split()]
            else:
                task_sequence = []
            routes.append(task_sequence)

        if len(routes) == 0:
            raise ValueError(f"No routes found in solution file: {solution_filepath}")

        # Step 4: Use the provided matrix directly so no backend/network handler is required.
        def matrix_travel_time(from_node_id: int, to_node_id: int) -> float:
            try:
                return float(travel_time_matrix[from_node_id][to_node_id])
            except (TypeError, IndexError) as exc:
                raise ValueError(
                    f"Invalid node indices for travel_time_matrix lookup: "
                    f"from={from_node_id}, to={to_node_id}"
                ) from exc

        # Step 5: Build driver_runs from solution routes. Number of vehicles is the number of routes in the solution, not the input header.
        depot_pt = depot["pt"]
        if original_payload["driver_runs"]:
            vehicle_state_template = original_payload["driver_runs"][0]["state"]
            start_time = vehicle_state_template["start_time"]
            end_time = vehicle_state_template["end_time"]
            am_capacity = vehicle_state_template["am_capacity"]
        else:
            start_time = 0
            end_time = 24 * 3600
            am_capacity = 0

        driver_runs = []
        assigned_tasks = []
        for run_id, route in enumerate(routes):
            current_node_id = depot["node_id"]
            current_time = float(start_time)
            order = 1
            manifest = []

            for task_id in route:
                if task_id not in stop_by_task_id:
                    raise ValueError(
                        f"Task id {task_id} from solution route is unknown for instance {original_filepath}"
                    )

                stop_template = stop_by_task_id[task_id]

                # Schedule this stop:
                # 1) travel from current node
                # 2) scheduled_time is the service start (cannot be before TW start)
                # 3) dwell is applied after service start to get service_end_time
                travel_time = matrix_travel_time(current_node_id, task_id)
                arrival_time = current_time + float(travel_time)
                service_start_time = max(arrival_time, float(stop_template["time_window_start"]))
                scheduled_time = service_start_time
                service_end_time = service_start_time + float(stop_template["dwell"])

                stop = {
                    "run_id": run_id,
                    "booking_id": stop_template["booking_id"],
                    "order": order,
                    "action": stop_template["action"],
                    "loc": copy.deepcopy(stop_template["loc"]),
                    "scheduled_time": scheduled_time,
                    "arrival_time": arrival_time,
                    "service_start_time": service_start_time,
                    "service_end_time": service_end_time,
                    "am": stop_template["am"],
                    "wc": stop_template["wc"],
                    "time_window_start": stop_template["time_window_start"],
                    "time_window_end": stop_template["time_window_end"],
                    "dwell": stop_template["dwell"],
                }
                manifest.append(stop)
                assigned_tasks.append(task_id)

                # Advance simulation clock after service is completed.
                current_time = service_end_time
                current_node_id = task_id
                order += 1

            # Finalize non-empty routes by returning each vehicle to the depot.
            if manifest and current_node_id != depot["node_id"]:
                depot_travel_time = matrix_travel_time(current_node_id, depot["node_id"])
                depot_arrival_time = current_time + float(depot_travel_time)
                depot_booking_id = -(run_id + 1)
                manifest.append({
                    "run_id": run_id,
                    "booking_id": depot_booking_id,
                    "order": order,
                    "action": "depot",
                    "loc": {
                        "lon": depot_pt["lon"],
                        "lat": depot_pt["lat"],
                        "node_id": depot["node_id"],
                    },
                    "scheduled_time": depot_arrival_time,
                    "arrival_time": depot_arrival_time,
                    "service_start_time": depot_arrival_time,
                    "service_end_time": depot_arrival_time,
                    "am": 0,
                    "wc": 0,
                    "time_window_start": depot_arrival_time - 10,
                    "time_window_end": depot_arrival_time + 10,
                    "dwell": 0.0,
                })

            driver_runs.append({
                "state": {
                    "run_id": run_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "am_capacity": am_capacity,
                    "wc_capacity": 0,
                    "locations_already_serviced": 0,
                    "location_dt_seconds": start_time,
                    "loc": {
                        "lon": depot_pt["lon"],
                        "lat": depot_pt["lat"],
                        "node_id": depot["node_id"],
                    },
                    "total_locations": len(manifest),
                },
                "manifest": manifest,
            })

        # Step 6: Validate that each non-depot task appears exactly once across routes.
        expected_tasks = set(stop_by_task_id.keys())
        assigned_task_set = set(assigned_tasks)
        if assigned_task_set != expected_tasks:
            missing = sorted(list(expected_tasks - assigned_task_set))
            unexpected = sorted(list(assigned_task_set - expected_tasks))
            raise ValueError(
                f"Routes do not cover all tasks exactly once. Missing: {missing}, Unexpected: {unexpected}"
            )
        if len(assigned_tasks) != len(expected_tasks):
            raise ValueError(
                "Duplicate task ids found in routes. Each task must appear exactly once."
            )

        solution_dict = {
            "requests": requests,
            "depot": depot,
            "driver_runs": driver_runs,
            "travel_time_matrix": travel_time_matrix,
        }
        return solution_dict # solution dict should have the same format as an original payload 'wilson'

    @staticmethod
    def build_payloads_from_solution_folder(
        input_folder: str | Path,
        solution_folder: str | Path,
        score_file: str | Path,
        *,
        abs_tol: float = 1e-2,
        rel_tol: float = 1e-6,
        export_payloads: bool = False,
        export_folder: str | Path | None = None,
    ) -> dict[str, dict]:
        """
        Parse every Li&Lim solution in a folder into complete payloads and validate score alignment.

        Returns a mapping by instance key (e.g. "lc101") containing payload, feasibility and
        score comparison metadata.
        """
        from rtv_solver.handlers.stats_parser import StatsParser
        from rtv_solver.structure.config import Config

        input_path = Path(input_folder)
        solution_path = Path(solution_folder)
        score_path = Path(score_file)

        with open(score_path, "r") as f:
            expected_scores: dict[str, float] = json.load(f)

        if export_payloads:
            if export_folder is None:
                export_path = solution_path / "optimal_manifests"
            else:
                export_path = Path(export_folder)
            export_path.mkdir(parents=True, exist_ok=True)
        else:
            export_path = None

        results: dict[str, dict] = {}
        solution_files = sorted(solution_path.glob("*.txt"))
        for solution_file in solution_files:
            instance_name = solution_file.stem
            original_file = input_path / f"{instance_name}.txt"
            if not original_file.exists():
                raise FileNotFoundError(f"Input file not found for {instance_name}: {original_file}")

            solution_payload = LiLimParser.parse_solution(
                original_filepath=str(original_file),
                solution_filepath=str(solution_file),
            )

            config = Config()
            # Keep dwell explicit to mirror the benchmark setup.
            config.DWELL_PICKUP = 90
            config.DWELL_DROPOFF = 90

            stats_parser = StatsParser(config=config, payload=solution_payload)
            feasible, stats, violations = stats_parser.evaluate(payload=solution_payload)

            computed_score = float(stats.vmt)
            expected_score = expected_scores.get(instance_name)
            if expected_score is None:
                raise KeyError(f"Missing expected score for '{instance_name}' in {score_path}")

            score_matches = math.isclose(
                computed_score,
                float(expected_score),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            if not score_matches:
                raise ValueError(
                    f"Score mismatch for {instance_name}: computed={computed_score}, "
                    f"expected={float(expected_score)}, feasible={feasible}, "
                    f"violations={len(violations)}"
                )

            # Keep only the final payload; score/feasibility can be recalculated when needed.
            results[instance_name] = solution_payload

            if export_path is not None:
                with open(export_path / f"{instance_name}.json", "w") as f:
                    json.dump(solution_payload, f, indent=2)

        return results

if __name__ == "__main__":
    parsed = LiLimParser.build_payloads_from_solution_folder(
        input_folder="inputs/li_lim/pdp_100",
        solution_folder="solutions/li_lim/txt_files",
        score_file="solutions/li_lim/liLim_solution_scores.json",
        abs_tol=5e-2,
        rel_tol=1e-6,
        export_payloads=True,
        export_folder="solutions/li_lim/manifests",
    )

    total = len(parsed)
    print(f"Parsed and stored payloads: {total}")

