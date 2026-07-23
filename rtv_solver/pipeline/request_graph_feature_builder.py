import math
import numpy as np


class RequestGraphFeatureBuilder:
    NODE_FEATURES = [
        "pickup_lat",
        "pickup_lon",
        "dropoff_lat",
        "dropoff_lon",
        "earliest_pickup_time",
        "pickup_window_length",
        "trip_distance_euclidean",
        "wheelchair",
        #"nearby_vehicle_count_10min",
        #"nearby_am_capacity_10min",
        #"nearby_wc_capacity_10min",
        #"nearest_vehicle_time_to_pickup",
        "local_request_count_10min",
        #"local_vehicle_count_10min",
        #"local_supply_demand_ratio_10min",
        #"local_am_capacity_per_request_10min",
    ]

    EDGE_FEATURES = [
        "pickup_distance",
        "dropoff_distance",
        "pickup_time_difference",
        "pickup_window_overlap_seconds",
        "pickup_window_overlap_ratio",
        # 2026-07-21: dropoff_window_overlap_seconds/ratio added (see
        # add_edge_features below, computation kept) but temporarily removed
        # from this list again -- v1_final/v2_final were trained with 10 edge
        # features, so adding these 2 breaks state_dict loading for the
        # existing checkpoints (edge_encoder expects [64,10], not [64,12]).
        # Re-enable once v1_final/v2_final are retrained with 12 features.
        "direction_similarity",
        "cross_pickup1_dropoff2_distance",
        "cross_pickup2_dropoff1_distance",
        "trip_distance_difference",
        "latest_pickup_time_difference",
    ]

    @staticmethod
    def add_features(G):
        RequestGraphFeatureBuilder.add_node_features(G)
        RequestGraphFeatureBuilder.add_edge_features(G)
        return G
    @staticmethod
    def add_node_features(G):
        for node_id, data in G.nodes(data=True):
            req = data["request"]

            pickup_lat = req.origin.lat
            pickup_lon = req.origin.lon
            dropoff_lat = req.destination.lat
            dropoff_lon = req.destination.lon

            data["pickup_lat"] = pickup_lat
            data["pickup_lon"] = pickup_lon
            data["dropoff_lat"] = dropoff_lat
            data["dropoff_lon"] = dropoff_lon

            data["earliest_pickup_time"] = req.earliest_pickup_time
            data["pickup_window_length"] = (
                req.latest_pickup_time - req.earliest_pickup_time
            )

            data["trip_distance_euclidean"] = RequestGraphFeatureBuilder._euclidean_distance(
                pickup_lat,
                pickup_lon,
                dropoff_lat,
                dropoff_lon,
            )

            data["wheelchair"] = float(req.wc_capacity)
            local_request_count = 0

            for other_id, other_data in G.nodes(data=True):
                if other_id == node_id:
                    continue

                other_req = other_data["request"]

                spatial_dist = RequestGraphFeatureBuilder._euclidean_distance(
                    req.origin.lat,
                    req.origin.lon,
                    other_req.origin.lat,
                    other_req.origin.lon,
                )

                time_diff = abs(
                    req.earliest_pickup_time - other_req.earliest_pickup_time
                )

                if spatial_dist <= 10.0 and time_diff <= 600:
                    local_request_count += 1

            data["local_request_count_10min"] = float(local_request_count)
        return G
    @staticmethod
    def add_edge_features(G):
        for u, v, data in G.edges(data=True):
            r1 = G.nodes[u]["request"]
            r2 = G.nodes[v]["request"]

            data["pickup_distance"] = RequestGraphFeatureBuilder._euclidean_distance(
                r1.origin.lat,
                r1.origin.lon,
                r2.origin.lat,
                r2.origin.lon,
            )

            data["dropoff_distance"] = RequestGraphFeatureBuilder._euclidean_distance(
                r1.destination.lat,
                r1.destination.lon,
                r2.destination.lat,
                r2.destination.lon,
            )

            data["pickup_time_difference"] = abs(
                r1.earliest_pickup_time - r2.earliest_pickup_time
            )

            overlap_seconds, overlap_ratio = RequestGraphFeatureBuilder._window_overlap(
                r1.earliest_pickup_time,
                r1.latest_pickup_time,
                r2.earliest_pickup_time,
                r2.latest_pickup_time,
            )

            data["pickup_window_overlap_seconds"] = overlap_seconds
            data["pickup_window_overlap_ratio"] = overlap_ratio

            dropoff_overlap_seconds, dropoff_overlap_ratio = RequestGraphFeatureBuilder._window_overlap(
                r1.earliest_arrival_time,
                r1.latest_arrival_time,
                r2.earliest_arrival_time,
                r2.latest_arrival_time,
            )

            data["dropoff_window_overlap_seconds"] = dropoff_overlap_seconds
            data["dropoff_window_overlap_ratio"] = dropoff_overlap_ratio

            data["direction_similarity"] = RequestGraphFeatureBuilder._direction_similarity(
                r1,
                r2,
            )

            trip_distance_1 = RequestGraphFeatureBuilder._euclidean_distance(
                r1.origin.lat,
                r1.origin.lon,
                r1.destination.lat,
                r1.destination.lon,
            )

            trip_distance_2 = RequestGraphFeatureBuilder._euclidean_distance(
                r2.origin.lat,
                r2.origin.lon,
                r2.destination.lat,
                r2.destination.lon,
            )

            data["cross_pickup1_dropoff2_distance"] = RequestGraphFeatureBuilder._euclidean_distance(
                r1.origin.lat,
                r1.origin.lon,
                r2.destination.lat,
                r2.destination.lon,
            )

            data["cross_pickup2_dropoff1_distance"] = RequestGraphFeatureBuilder._euclidean_distance(
                r2.origin.lat,
                r2.origin.lon,
                r1.destination.lat,
                r1.destination.lon,
            )

            data["trip_distance_difference"] = abs(
                trip_distance_1 - trip_distance_2
            )

            data["latest_pickup_time_difference"] = abs(
                r1.latest_pickup_time - r2.latest_pickup_time
            )

        return G   

    @staticmethod
    def to_numpy(G):
        node_matrix = []
        node_ids = []

        for node_id, data in G.nodes(data=True):
            node_ids.append(node_id)
            node_matrix.append([
                float(data[name])
                for name in RequestGraphFeatureBuilder.NODE_FEATURES
            ])

        edge_matrix = []
        edge_index = []

        for u, v, data in G.edges(data=True):
            edge_index.append([u, v])
            edge_matrix.append([
                float(data[name])
                for name in RequestGraphFeatureBuilder.EDGE_FEATURES
            ])

        return (
            np.array(node_matrix, dtype=np.float32),
            np.array(edge_index, dtype=np.int64),
            np.array(edge_matrix, dtype=np.float32),
            node_ids,
        )

    @staticmethod
    def _euclidean_distance(lat1, lon1, lat2, lon2):
        return math.sqrt(
            (lat1 - lat2) ** 2
            + (lon1 - lon2) ** 2
        )

    @staticmethod
    def _window_overlap(start1, end1, start2, end2):
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        overlap = max(0.0, overlap_end - overlap_start)

        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union = max(1.0, union_end - union_start)

        return overlap, overlap / union

    @staticmethod
    def _direction_similarity(r1, r2):
        v1 = np.array([
            r1.destination.lat - r1.origin.lat,
            r1.destination.lon - r1.origin.lon,
        ])

        v2 = np.array([
            r2.destination.lat - r2.origin.lat,
            r2.destination.lon - r2.origin.lon,
        ])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))