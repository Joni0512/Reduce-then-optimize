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
    def add_features(G, geographic: bool = False):
        # 2026-08-02: `geographic` added because NYC's lat/lon are real WGS84
        # degrees (curved-earth), while Li&Lim's lat/lon fields actually hold
        # the benchmark's abstract flat-grid X/Y coordinates (see
        # li_lim_parser.py) - plain Euclidean distance is correct for Li&Lim
        # but distorts NYC distances direction-dependently (verified: ~24%
        # under-weights east-west vs. north-south at NYC's latitude).
        # Default False keeps Li&Lim's behavior byte-identical.
        RequestGraphFeatureBuilder.add_node_features(G, geographic=geographic)
        RequestGraphFeatureBuilder.add_edge_features(G, geographic=geographic)
        return G
    @staticmethod
    def add_node_features(G, geographic: bool = False):
        distance_fn = (
            RequestGraphFeatureBuilder._haversine_distance_m
            if geographic
            else RequestGraphFeatureBuilder._euclidean_distance
        )
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

            # 2026-08-02: uses haversine (meters) when geographic=True; name
            # kept as "trip_distance_euclidean" so NODE_FEATURES/checkpoint
            # column order is unaffected - only the underlying metric changes.
            data["trip_distance_euclidean"] = distance_fn(
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

                # NOTE 2026-08-02: still plain Euclidean with a fixed 10.0
                # threshold regardless of `geographic` - that threshold is
                # tuned for Li&Lim's grid units and is not yet meaningful for
                # NYC's real coordinates (would need a meter-based cutoff).
                # Known open issue, intentionally not touched here - no
                # agreed-on meter value yet.
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
    def add_edge_features(G, geographic: bool = False):
        distance_fn = (
            RequestGraphFeatureBuilder._haversine_distance_m
            if geographic
            else RequestGraphFeatureBuilder._euclidean_distance
        )
        for u, v, data in G.edges(data=True):
            r1 = G.nodes[u]["request"]
            r2 = G.nodes[v]["request"]

            data["pickup_distance"] = distance_fn(
                r1.origin.lat,
                r1.origin.lon,
                r2.origin.lat,
                r2.origin.lon,
            )

            data["dropoff_distance"] = distance_fn(
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

            trip_distance_1 = distance_fn(
                r1.origin.lat,
                r1.origin.lon,
                r1.destination.lat,
                r1.destination.lon,
            )

            trip_distance_2 = distance_fn(
                r2.origin.lat,
                r2.origin.lon,
                r2.destination.lat,
                r2.destination.lon,
            )

            data["cross_pickup1_dropoff2_distance"] = distance_fn(
                r1.origin.lat,
                r1.origin.lon,
                r2.destination.lat,
                r2.destination.lon,
            )

            data["cross_pickup2_dropoff1_distance"] = distance_fn(
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
    # used for Li&Lim
    @staticmethod
    def _euclidean_distance(lat1, lon1, lat2, lon2):
        return math.sqrt(
            (lat1 - lat2) ** 2
            + (lon1 - lon2) ** 2
        )

    @staticmethod
    def _haversine_distance_m(lat1, lon1, lat2, lon2):
        # 2026-08-02: added for NYC (geographic=True) - real-distance
        # counterpart to _euclidean_distance, same formula already used
        # elsewhere in the pipeline (build_nyc_manifest.py, feat_builder.py's
        # _calc_geo_distance_meter) so the pruner's notion of distance matches
        # the solver's. Not used for Li&Lim, whose lat/lon fields hold flat
        # grid coordinates, not real degrees.
        R = 6371000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
        )
        return 2 * R * math.asin(math.sqrt(a))

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