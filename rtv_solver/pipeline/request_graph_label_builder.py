import numpy as np
import pandas as pd


class RequestGraphLabelBuilder:
    @staticmethod
    def build_heuristic_labels(
        edge_df: pd.DataFrame,
        max_pickup_time_difference: float = 3600.0,
        min_direction_similarity: float = 0.0,
        require_pickup_overlap: bool = True,
    ) -> np.ndarray:
        time_ok = edge_df["pickup_time_difference"] <= max_pickup_time_difference
        direction_ok = edge_df["direction_similarity"] > min_direction_similarity

        if require_pickup_overlap:
            overlap_ok = edge_df["pickup_window_overlap_seconds"] > 0
        else:
            overlap_ok = True

        labels = time_ok & direction_ok & overlap_ok
        return labels.astype(float).to_numpy()

    @staticmethod
    def build_positive_pairs_from_solution_payload(
        solution_payload: dict,
    ) -> set[tuple[int, int]]:
        positive_pairs = set()

        for driver_run in solution_payload["driver_runs"]:
            request_ids = set()

            for stop in driver_run["manifest"]:
                booking_id = int(stop["booking_id"])

                # Depot-Stops haben künstliche negative booking_ids.
                if booking_id < 0:
                    continue

                request_ids.add(booking_id)

            request_ids = sorted(request_ids)

            for i in range(len(request_ids)):
                for j in range(i + 1, len(request_ids)):
                    positive_pairs.add((request_ids[i], request_ids[j]))

        return positive_pairs

    @staticmethod
    def build_positive_pairs_overlap_from_solution_payload(
        solution_payload: dict,
    ) -> set[tuple[int, int]]:
        # 2026-11-03: stricter pair label requested for the NYC pair pruner - a pair is positive
        # only if both requests are ON BOARD THE SAME VEHICLE AT THE SAME TIME, i.e. their
        # [pickup, dropoff] intervals (stop `scheduled_time`) overlap strictly. The original
        # label above marks every pair of requests that ever appears in one vehicle's whole-day
        # manifest; on the NYC train day (RHO batch_interval=3600/step_size=1800) only 825 of
        # 9,159 such pairs overlap in time (9.01%), the rest are sequential rides.
        positive_pairs = set()

        for driver_run in solution_payload["driver_runs"]:
            intervals = {}

            for stop in driver_run["manifest"]:
                booking_id = int(stop["booking_id"])

                if booking_id < 0:
                    continue

                intervals.setdefault(booking_id, {})[stop["action"]] = stop["scheduled_time"]

            # Only requests with both a pickup and a dropoff stop in the manifest.
            complete = sorted(
                b for b, t in intervals.items() if "pickup" in t and "dropoff" in t
            )

            for i in range(len(complete)):
                a = complete[i]
                for j in range(i + 1, len(complete)):
                    b = complete[j]
                    if (
                        intervals[a]["pickup"] < intervals[b]["dropoff"]
                        and intervals[b]["pickup"] < intervals[a]["dropoff"]
                    ):
                        positive_pairs.add((a, b))

        return positive_pairs

    @staticmethod
    def build_expert_labels_from_solution_payload(
        edge_index,
        node_ids,
        solution_payload: dict,
        label_mode: str = "vehicle",
    ) -> np.ndarray:
        # 2026-11-03: label_mode added - "vehicle" (default, unchanged behaviour) or "overlap"
        # (simultaneously on board the same vehicle, see the overlap builder above).
        if label_mode == "overlap":
            positive_pairs = RequestGraphLabelBuilder.build_positive_pairs_overlap_from_solution_payload(
                solution_payload
            )
        elif label_mode == "vehicle":
            positive_pairs = RequestGraphLabelBuilder.build_positive_pairs_from_solution_payload(
                solution_payload
            )
        else:
            raise ValueError(f"unknown label_mode: {label_mode}")

        labels = []

        for u, v in edge_index:
            pair = tuple(sorted((int(u), int(v))))
            labels.append(1.0 if pair in positive_pairs else 0.0)

        return np.array(labels, dtype=np.float32)