import math

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.vehicle_handler import VehicleHandler


class RequestGraphVehicleFeatureBuilder:
    """
    Computes additional node features describing the local vehicle supply
    and request density around each transportation request.

    These features provide the GNN with information about local fleet
    availability and supply-demand balance.
    """

    VEHICLE_CONTEXT_NODE_FEATURES = [
        # Vehicle supply around request pickup
        "nearby_vehicle_count_10min",
        "nearby_am_capacity_10min",
        "nearby_wc_capacity_10min",
        "nearest_vehicle_time_to_pickup",

        # Local demand / supply area around request pickup
        "local_request_count_10min",
        "local_vehicle_count_10min",
        "local_supply_demand_ratio_10min",
        "local_am_capacity_per_request_10min",
    ]
    @staticmethod
    def _ensure_node_id(node):
        """
        Assign a network node ID if it has not been created yet.

        Travel-time queries require valid network node IDs for routing.
        """
        if node.node_id is None:
            node.node_id = NetworkHandler.get_next_node_id(
                node.lat,
                node.lon,
            )
        return node

    @staticmethod
    def add_vehicle_and_local_context_features(
        G,
        vehicles: dict,
        max_travel_time_seconds: float = 600.0,
    ):
        """
        Fügt pro Request-Node Vehicle-Supply und Local-Area  Features hinzu.

        Radius:
            Default 600 Sekunden = 10 Minuten.

        Idee:
            Für jeden Request schauen wir:
                - Welche Vehicles sind innerhalb von 10 Minuten am Pickup?
                - Wie viel Kapazität ist dort verfügbar?
                - Wie viele andere Requests liegen lokal in der Nähe?
                - Wie angespannt ist Supply vs Demand?
        """

        nodes = list(G.nodes)

        for node_id in nodes:
            request = G.nodes[node_id]["request"]
            pickup_node = request.origin

            # 1. Estimate vehicle availability around the pickup location.

            nearby_vehicle_count = 0.0
            nearby_am_capacity = 0.0
            nearby_wc_capacity = 0.0
            nearest_vehicle_time = math.inf

            for vehicle in vehicles.values():
                vehicle_node, _ = VehicleHandler.get_current_location_time(vehicle)

                travel_time = NetworkHandler.travel_time(
                    vehicle_node,
                    pickup_node,
                )

                nearest_vehicle_time = min(nearest_vehicle_time, travel_time)
                # Only vehicles within the travel-time radius contribute to the local supply features.

                if travel_time <= max_travel_time_seconds:
                    nearby_vehicle_count += 1.0
                    nearby_am_capacity += float(vehicle.am_capacity)
                    nearby_wc_capacity += float(vehicle.wc_capacity)

            if math.isinf(nearest_vehicle_time):
                nearest_vehicle_time = max_travel_time_seconds * 10.0


            # 2. Local request demand around this request
        

            local_request_count = 0.0

            for other_node_id in nodes:
                if other_node_id == node_id:
                    continue

                other_request = G.nodes[other_node_id]["request"]
                other_pickup_node = other_request.origin

                request_to_request_time = NetworkHandler.travel_time(
                    pickup_node,
                    other_pickup_node,
                )

                if request_to_request_time <= max_travel_time_seconds:
                    local_request_count += 1.0

            # 3. Estimate the local supply-demand balance.

            # Add one to avoid division by zero for isolated requests.
            demand_denominator = local_request_count + 1.0

            # Approximate the local supply-demand balance.
            local_supply_demand_ratio = (
                nearby_vehicle_count / demand_denominator
            )

            local_am_capacity_per_request = (
                nearby_am_capacity / demand_denominator
            )
            # 4. Store the computed features as node attributes.
        

            G.nodes[node_id]["nearby_vehicle_count_10min"] = nearby_vehicle_count
            G.nodes[node_id]["nearby_am_capacity_10min"] = nearby_am_capacity
            G.nodes[node_id]["nearby_wc_capacity_10min"] = nearby_wc_capacity
            G.nodes[node_id]["nearest_vehicle_time_to_pickup"] = float(nearest_vehicle_time)

            G.nodes[node_id]["local_request_count_10min"] = local_request_count
            G.nodes[node_id]["local_vehicle_count_10min"] = nearby_vehicle_count
            G.nodes[node_id]["local_supply_demand_ratio_10min"] = local_supply_demand_ratio
            G.nodes[node_id]["local_am_capacity_per_request_10min"] = local_am_capacity_per_request

        return G