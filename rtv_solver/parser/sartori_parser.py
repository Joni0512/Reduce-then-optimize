import copy

class SartoriParser:
    """
    Parser for Sartori & Buriol PDPTW benchmark instances.
    https://github.com/cssartori/pdptw-instances/tree/master

    Format (see inputs/sartori/README.txt):
    - First 10 lines: metadata (SIZE, ROUTE-TIME, CAPACITY, etc.)
    - NODES: SIZE lines with <id> <lat> <lon> <dem> <etw> <ltw> <sd> <p> <d>
    - Node 0 is depot
    - dem > 0 for pickup, dem < 0 for delivery
    - Pickup id pairs with delivery id = id + (SIZE-1)/2
    - EDGES: SIZE lines of SIZE integers (travel times in minutes, OSRM-based)
    """
    # TODO exchange relevant keys to PayloadParser keys, but first need a file to check correct outcome

    @staticmethod
    def parse_file(filepath, num_vehicles=None):
        """
        Parse a Sartori instance file and return data in the same format as LiLimParser.

        Args:
            filepath: Path to the Sartori instance file
            num_vehicles: Optional number of vehicles. If None, defaults to min(50, num_requests).

        Returns:
            dict with keys: requests, depot, driver_runs, travel_time_matrix
        """
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Parse header (first 10 lines)
        header = {}
        for line in lines[:10]:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()

        size = int(header["SIZE"])
        route_time = int(header["ROUTE-TIME"])
        capacity = int(header["CAPACITY"])

        # Find NODES and EDGES sections
        nodes_start = None
        edges_start = None
        for i, line in enumerate(lines):
            if line.strip() == "NODES":
                nodes_start = i + 1
            elif line.strip() == "EDGES":
                edges_start = i + 1
                break

        if nodes_start is None or edges_start is None:
            raise ValueError("Invalid Sartori file: missing NODES or EDGES section")

        # Parse nodes
        tasks = {}
        for i in range(size):
            line = lines[nodes_start + i]
            parts = line.strip().split()
            if len(parts) < 9:
                raise ValueError(f"Invalid node line {i + 1}: {line}")

            node_id = int(parts[0])
            lat = float(parts[1])
            lon = float(parts[2])
            demand = int(parts[3])
            etw = int(parts[4])
            ltw = int(parts[5])
            service_time = int(parts[6])

            tasks[node_id] = {
                "task_no": node_id,
                "x": lon,  # Use lon as x for consistency with LiLimParser
                "y": lat,  # Use lat as y
                "demand": demand,
                "earliest": etw,
                "latest": ltw,
                "service_time": service_time,
            }

        # Depot is node 0
        depot_task = tasks[0]
        depot = {"pt": {"lon": depot_task["x"], "lat": depot_task["y"]}}
        depot_start_time = depot_task["earliest"]
        depot_end_time = depot_task["latest"]
        depot_loc = {"lon": depot_task["x"], "lat": depot_task["y"]}

        # Build pickup-delivery pairs: pickup id -> delivery id = id + (SIZE-1)/2
        num_pickups = (size - 1) // 2
        requests = []
        for pickup_id in range(1, num_pickups + 1):
            delivery_id = pickup_id + num_pickups
            pickup_task = tasks[pickup_id]
            delivery_task = tasks[delivery_id]

            if pickup_task["demand"] <= 0 or delivery_task["demand"] >= 0:
                raise ValueError(
                    f"Invalid pair: pickup {pickup_id} (dem={pickup_task['demand']}), "
                    f"delivery {delivery_id} (dem={delivery_task['demand']})"
                )

            request = {
                "booking_id": str(pickup_id),
                "pickup_pt": {
                    "lon": pickup_task["x"],
                    "lat": pickup_task["y"],
                    "node_id": pickup_task["task_no"],
                },
                "dropoff_pt": {
                    "lon": delivery_task["x"],
                    "lat": delivery_task["y"],
                    "node_id": delivery_task["task_no"],
                },
                "pickup_time_window_start": pickup_task["earliest"],
                "pickup_time_window_end": pickup_task["latest"],
                "dropoff_time_window_start": delivery_task["earliest"],
                "dropoff_time_window_end": delivery_task["latest"],
                "am": abs(pickup_task["demand"]),
                "wc": 0,
                "pickup_service_time": pickup_task["service_time"],
                "dropoff_service_time": delivery_task["service_time"],
            }
            requests.append(request)

        requests = sorted(requests, key=lambda r: r["pickup_time_window_start"])

        # Parse travel time matrix from EDGES
        travel_time_matrix = []
        for i in range(size):
            line = lines[edges_start + i]
            row = [int(x) for x in line.strip().split()]
            if len(row) != size:
                raise ValueError(
                    f"EDGES row {i + 1}: expected {size} values, got {len(row)}"
                )
            travel_time_matrix.append(row)

        # Build driver_runs
        if num_vehicles is None:
            num_vehicles = min(50, len(requests))

        driver_runs = []
        for i in range(num_vehicles):
            driver_runs.append(
                {
                    "state": {
                        "run_id": i,
                        "start_time": depot_start_time,
                        "end_time": depot_end_time,
                        "am_capacity": capacity,
                        "wc_capacity": 0,
                        "locations_already_serviced": 0,
                        "location_dt_seconds": depot_start_time,
                        "loc": copy.deepcopy(depot_loc),
                        "total_locations": 0,
                    },
                    "manifest": [],
                }
            )

        return {
            "requests": requests,
            "depot": depot,
            "driver_runs": driver_runs,
            "travel_time_matrix": travel_time_matrix,
        }