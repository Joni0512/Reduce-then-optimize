import json
import itertools
import numpy as np
import torch
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rtv_solver.structure.trip_cost import TripCost

from rtv_solver.structure.config import Config
from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.request import Request
from rtv_solver.structure.driver_run import DriverRun, ManifestEntry
from rtv_solver.structure.trip_cost import TripCost

from rtv_solver.schema.payload_keys import PayloadKeys

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

TYPE_BEST_ORDERED_MATCH = "best_ordered_match"
TYPE_BEST_UNORDERED_MATCH = "best_unordered_match"

class ImitationHandler:
    """
    Build imitation scores and y* targets for the CO layer.

    The key invariant across all methods is index stability:
    - `trip_costs[i]` always maps to `imitation_scores[i]` and `y_star[i]`
    - reject-action scores (if enabled) are appended after trip scores in the
      exact order of `reject_vehicle_ids`

    This class supports both:
    - legacy single-vehicle scoring (`get_optimal_request_solution_for_batch`)
    - vehicle-aware multi-vehicle scoring/y* construction
    """
    # The current ImitationHandler cannot handle multiple vehicles in many methods or at least we do not have the methods that handle multiple vehicles. We can generate the optimal solution and have a dictionary of vehicles {vehicle_id: optimal_solution}.
    # However, turning the current trips into solutions requires new methods that are not yet implemented.
    # The data that we get from the tripHandler clearly has the direct vehicle association in the TripCost object but also in the vehicle_to_trips_cost_map and trip_to_vehicle_cost_map dictionaries. We can use this information to get the optimal solutions for each vehicle. We still need to make sure that the order of the trips is preserved.
    # Instructions: we need new methods to handle the optimal solutions for multiple vehicles. We need to make sure that the final order of our imitated solution matches the corresponding order of the generated trips. The code should be highly testable and robust.
    # The order is very important as the order of TripCosts is deeply integrated into the optimization process. The final outcome must make sure that the order remains, even if we temporarily deduce solutions per vehicle. The number of vehicles is known beforehand and can be identified by a clear vehicleID. We need to add testing to check the final outcome of both the imitation_scores and the y_star tensor.
    def __init__(self, config: Config):
        self.config = config
        self.optimal_solution = self._load_complete_optimal_solution()

    @staticmethod
    def _validate_unique_request_ids(request_batch: list[Request]) -> list[int]:
        """
        Extract request IDs and reject duplicate IDs in the same batch.

        Duplicate IDs would make subsequence matching ambiguous during scoring.
        """
        request_ids = [request.id for request in request_batch]
        assert len(request_ids) == len(set(request_ids)), (
            f"Duplicate request IDs in batch are not supported: {request_ids}"
        )
        return request_ids

    def get_optimal_request_solution_for_batch(self, request_batch: list[Request]) -> list[int]:
        """
        Legacy single-vehicle helper.

        For backward compatibility this selects one vehicle sequence and filters it to the current batch. Multi-vehicle paths should use  `get_optimal_request_solution_for_batch_by_vehicle`.
        """
        request_ids = self._validate_unique_request_ids(request_batch)
        if not self.optimal_solution:
            raise ValueError("No optimal solution loaded.")

        # Keep historical behavior where vehicle 0 is preferred if present.
        # Fallback to the smallest vehicle id to keep this deterministic.
        selected_vehicle_id = 0 if 0 in self.optimal_solution else min(self.optimal_solution.keys())
        optimal_sequence = self.optimal_solution[selected_vehicle_id]
        request_ids_set = set(request_ids)

        # Keep order by iterating over the full optimal sequence and filtering for active requests.
        part_optimal_sequence = [
            request_id for request_id in optimal_sequence if request_id in request_ids_set
        ]

        # Sanity check: the result must be an ordered subsequence of the optimal sequence.
        optimal_positions = {request_id: idx for idx, request_id in enumerate(optimal_sequence)}
        part_positions = [optimal_positions[request_id] for request_id in part_optimal_sequence]
        assert part_positions == sorted(part_positions), (
            "Internal error: partial optimal sequence is not ordered as a subsequence "
            "of the optimal solution."
        )
        return part_optimal_sequence

    def get_optimal_request_solution_for_batch_by_vehicle(
        self,
        request_batch: list[Request],
        vehicle_ids: list[int],
    ) -> dict[int, list[int]]:
        """
        Return per-vehicle ordered optimal pickup subsequences for the batch.

        The resulting dictionary maps each vehicle ID to the list of request IDs
        that are active in the current batch, preserving each vehicle's manifest
        pickup order from `self.optimal_solution`.
        """
        request_ids = self._validate_unique_request_ids(request_batch)
        request_ids_set = set(request_ids)
        result: dict[int, list[int]] = {}
        for vehicle_id in vehicle_ids:
            if vehicle_id not in self.optimal_solution:
                raise KeyError(
                    f"Vehicle {vehicle_id} has no entry in loaded optimal solution."
                )
            optimal_sequence = self.optimal_solution[vehicle_id]
            subsequence = [
                request_id for request_id in optimal_sequence if request_id in request_ids_set
            ]
            result[vehicle_id] = subsequence
        return result

    def get_optimal_stop_sequence_for_batch(self, request_batch: list[Request]) -> list[int]:
        pass

    def _load_complete_optimal_solution(self):
        """
        Load the complete optimal solution from the payload for the entire period.

        Assumption: all vehicles start in the same position (singledepot).
        """
        if self.config.IMITATION_SOLUTION_FILE is None:
            raise ValueError("No imitation solution file provided.")
        
        with open(Path(__file__).resolve().parent.parent.parent / self.config.IMITATION_SOLUTION_FILE, "r") as f:
            payload_data = json.load(f)
        
        # driverID: solution list of request IDs
        optimal_solution = {}
        for driver_run in payload_data[PayloadKeys.DRIVERS]:
            run = DriverRun.from_dict(driver_run)
            manifest = run.manifest
            driver_id = run.state.run_id

            complete_solution = self._manifest_to_request_solution(manifest)
            optimal_solution[driver_id] = complete_solution

        return optimal_solution

    def _manifest_to_request_solution(self, manifest: list[ManifestEntry]) -> list[dict]:
        """
        When reading the manifest of a vehicle, we need to convert it to a solution that can be used by the imitation solver.

        First, we consider this to only solve the manifest of a single vehicle (collect the order of pickups)
        We want to get all requests in the right order in a specific list. We consider the trip as the complete solution.
        """
        solution = []
        for stop_entry in manifest:
            if stop_entry.action == VehicleStop.ACT_PICKUP:
                request_id = stop_entry.booking_id
                solution.append(request_id)
        return solution

    def _manifest_to_stop_sequence(self, manifest: list[dict]) -> list[dict]:
        """
        TODO
        """
        solution = []
        for stop in manifest:
            if stop[PayloadKeys.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                stop_id = stop[PayloadKeys.NIFEST_BOOKING_ID]
                solution.append(stop_id)
            if stop[PayloadKeys.MANIFEST_ACTION] == VehicleStop.ACT_DROPOFF:
                stop_id = stop[PayloadKeys.MANIFEST_BOOKING_ID]
                solution.append(-1 * stop_id)
        return solution

    @staticmethod
    def _find_subsequence(sequence: list[dict], subsequence: list[dict]) -> int:
        """
        Finds the exact, same-ordered subsequence in the total sequence.

        Returns the index of the subsequence in the sequence.
        """
        n = len(subsequence)
        for i in range(len(sequence) - n + 1):
            if sequence[i:i+n] == subsequence:
                return i
        return -1

    @staticmethod
    def find_permuted_subsequence(sequence: list[dict], subsequence: list[dict]) -> int:
        """
        Finds the existence of the permuted subsequence in the total sequence.

        Returns the index of the subsequence in the sequence.
        """
        n = len(subsequence)
        target = sorted(subsequence)

        for i in range(len(sequence) - n + 1):
            if sorted(sequence[i:i+n]) == target:
                return i  # start index
        return None

    @staticmethod
    def get_trip_index_combinations(
        trips: list[int], min_cardinality: int = 1, max_cardinality: int | None = None
    ) -> tuple[list[tuple[int, ...]], dict[int, int]]:
        """
        Get all possible combinations of trip values of a given cardinality.
        Mostly for testing purposes.

        NOTE: Despite the historical method name, this returns combinations
        of values from `trips`, not positional indices.

        Returns:
            - combinations: list of all possible combinations of trips
            - counts_by_cardinality: dictionary of the number of combinations for each cardinality
        """
        n_trips = len(trips)
        if max_cardinality is None:
            max_cardinality = n_trips
        if min_cardinality < 0 or max_cardinality < 0:
            raise ValueError("Cardinality bounds must be non-negative.")
        if min_cardinality > max_cardinality:
            raise ValueError("min_cardinality must be <= max_cardinality.")
        if max_cardinality > n_trips:
            raise ValueError("max_cardinality cannot exceed number of trips.")

        combinations: list[tuple[int, ...]] = []
        counts_by_cardinality: dict[int, int] = {}
        for cardinality in range(min_cardinality, max_cardinality + 1):
            cardinality_combinations = list(itertools.combinations(trips, cardinality))
            counts_by_cardinality[cardinality] = len(cardinality_combinations)
            combinations.extend(cardinality_combinations)

        return combinations, counts_by_cardinality

    @staticmethod
    def score_combinations_against_solution(
        combinations: list[tuple[int, ...]], optimal_solution: list[int]
    ) -> torch.Tensor:
        """
        Score each combination against an optimal solution sequence.

        Parameters:
            - combinations: list of combinations of request IDs
            - optimal_solution: list of request IDs of the optimal solution sequence

        Scoring rules:
            - 1000: perfect full match with the complete optimal solution and same order
            - ordered full match for the combination length:
                - 100 + 10 * len(combination), if it starts at index 0
            - 0: no match

        NOTE This could be an interesting experiment, which assignment leads to the best results. We would probably take the the max or min score for further evaluation.
        """
        scores: list[int] = []
        optimal_tuple = tuple(optimal_solution)
        
        debug_dict = {}
        debug_dict_key = -1

        for combination in combinations:
            debug_dict_key += 1
            # Perfect match with the complete optimal solution.
            if combination == optimal_tuple:
                score_full_ordered_exact_match = 1000
                scores.append(score_full_ordered_exact_match)
                debug_dict[debug_dict_key] =  score_full_ordered_exact_match, combination
                continue

            combination_list = list(combination)
            ordered_start_index = ImitationHandler._find_subsequence(
                optimal_solution, combination_list
            )
            permuted_start_index = ImitationHandler.find_permuted_subsequence(
                optimal_solution, combination_list
            )

            # Full ordered match for this combination length.
            if ordered_start_index != -1:
                if ordered_start_index == 0:
                    score_full_ordered_early_index = 100 + 10 * len(combination)
                    scores.append(score_full_ordered_early_index)
                    debug_dict[debug_dict_key] =  score_full_ordered_early_index, combination
                else:
                    # Keep vector alignment: each combination must produce exactly one score.
                    scores.append(0)
                    debug_dict[debug_dict_key] = 0, combination
                # else:
                    # DISCARDED option for score making
                    # - -10 * len(combination), if it starts at a later index
                    # score_partial_ordered_late_index = 0 + 10 * len(combination)
                    # scores.append(score_partial_ordered_late_index)
                    # debug_dict[debug_dict_key] =  score_partial_ordered_late_index, combination
                continue
            
            # DISCARDED option for score making
            # - unordered full match for the combination length:
            #    - -100 - 10 * len(combination)
            # Full coverage for this combination length but wrong order.
            # if permuted_start_index is not None:
            #     score_full_unordered = - 100 - 10 * len(combination)
            #     scores.append(score_full_unordered)
            #     debug_dict[debug_dict_key] =  score_full_unordered, combination
            #     continue

            # alternative score for all other cases
            scores.append(0)
            debug_dict[debug_dict_key] =  0, combination

        score_tensor = torch.tensor(scores)
        # print(debug_dict)

        # assert that there is at least one value that is greater than 0 and another test that a value is smaller than 0
        if not torch.any(score_tensor > 0): 
            console_logger.info(f"IH: No positive scores found.")
        return score_tensor

    @staticmethod
    def tripCosts_to_request_combinations(tripCosts: list[TripCost]) -> list[tuple[int, ...]]:
        """
        Convert a list of trip costs to a list of request combinations.
        """
        combinations = []
        for tripCost in tripCosts:
            combinations.append(tripCost.get_ordered_request_ids())
        return combinations

    @staticmethod
    def build_trip_vehicle_index_map(trip_costs: list[TripCost]) -> dict[int, list[int]]:
        """
        Build a deterministic vehicle -> global tripCost-index mapping.

        This helper mirrors `TripHandler.vehicle_to_trips_cost_map` but can be reconstructed locally from `trip_costs` for isolated unit tests and defensive validation.
        """
        vehicle_to_indices: dict[int, list[int]] = {}
        for trip_idx, trip_cost in enumerate(trip_costs):
            vehicle_to_indices.setdefault(trip_cost.vehicle_id, []).append(trip_idx)
        return vehicle_to_indices

    @staticmethod
    def _validate_vehicle_index_map(
        trip_costs: list[TripCost],
        vehicle_to_trips_cost_map: dict[int, list[int]],
    ) -> None:
        """
        Validate that a vehicle->indices map is consistent with `trip_costs`.
        """
        expected_indices = set(range(len(trip_costs)))
        seen_indices: set[int] = set()
        for vehicle_id, indices in vehicle_to_trips_cost_map.items():
            for idx in indices:
                if idx < 0 or idx >= len(trip_costs):
                    raise ValueError(
                        f"Vehicle {vehicle_id} contains out-of-range trip index {idx}."
                    )
                if idx in seen_indices:
                    raise ValueError(
                        f"Trip index {idx} appears in multiple vehicle groups."
                    )
                seen_indices.add(idx)
                if trip_costs[idx].vehicle_id != vehicle_id:
                    raise ValueError(
                        "Vehicle index map disagrees with trip_costs vehicle IDs "
                        f"at index {idx}: map={vehicle_id}, trip={trip_costs[idx].vehicle_id}."
                    )
        if seen_indices != expected_indices:
            missing = sorted(expected_indices - seen_indices)
            raise ValueError(
                f"Vehicle index map does not cover all trip indices. Missing: {missing}"
            )

    def score_trip_costs_against_optimal_by_vehicle(
        self,
        trip_costs: list[TripCost],
        request_batch: list[Request],
        vehicle_to_trips_cost_map: dict[int, list[int]] | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Score each trip against the corresponding vehicle's optimal subsequence.

        Returns:
        - `trip_scores`: tensor of shape `(len(trip_costs),)` aligned 1:1 with global `trip_costs` order.
        - `trip_vehicle_ids`: list of vehicle IDs with the same global ordering.

        Design details:
        - Scoring is computed per vehicle to avoid cross-vehicle contamination.
        - Results are written back to the original global indices.
        - If no external map is provided, grouping is reconstructed locally via `build_trip_vehicle_index_map`.
        """
        if not trip_costs:
            return torch.empty(0, dtype=torch.int64), []

        trip_vehicle_ids = [trip_cost.vehicle_id for trip_cost in trip_costs]
        grouping = (
            vehicle_to_trips_cost_map
            if vehicle_to_trips_cost_map is not None
            else self.build_trip_vehicle_index_map(trip_costs)
        )
        self._validate_vehicle_index_map(trip_costs, grouping)

        vehicle_ids = [vehicle_id for vehicle_id, indices in grouping.items() if len(indices) > 0]
        optimal_by_vehicle = self.get_optimal_request_solution_for_batch_by_vehicle(
            request_batch=request_batch,
            vehicle_ids=vehicle_ids,
        )

        trip_scores = torch.zeros(len(trip_costs), dtype=torch.int64)
        for vehicle_id, global_indices in grouping.items():
            if len(global_indices) == 0:
                continue
            # Keep the external map order if provided so debugging can mirror TripHandler's index lists exactly.
            ordered_global_indices = list(global_indices)
            combinations = [
                tuple(trip_costs[global_idx].get_ordered_request_ids())
                for global_idx in ordered_global_indices
            ]
            local_scores = self.score_combinations_against_solution(
                combinations=combinations,
                optimal_solution=optimal_by_vehicle[vehicle_id],
            )
            for local_idx, global_idx in enumerate(ordered_global_indices):
                trip_scores[global_idx] = local_scores[local_idx]
        return trip_scores, trip_vehicle_ids

    @staticmethod
    def append_reject_action_scores(
        imitation_scores: torch.Tensor,
        reject_vehicle_ids: list[int],
        trip_vehicle_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Append one imitation score per reject-action (vehicle-specific).

        Logic:
        - For each vehicle reject action, check all trip scores belonging to
          that vehicle.
        - If the vehicle has no positive trip score, append reject score -10.
        - Otherwise append 0 for that vehicle.

        If `trip_vehicle_ids` is not provided, fallback to global behavior:
        if no positive score exists at all, assign -10 to all reject actions.
        """
        if len(reject_vehicle_ids) == 0:
            return imitation_scores

        reject_penalty_value = -10.0

        if trip_vehicle_ids is None:
            has_any_positive = bool(torch.any(imitation_scores > 0).item())
            reject_value = 0.0 if has_any_positive else reject_penalty_value
            reject_scores = torch.full(
                (len(reject_vehicle_ids),),
                reject_value,
                dtype=imitation_scores.dtype,
                device=imitation_scores.device,
            )
            return torch.cat([imitation_scores, reject_scores], dim=0)

        if len(trip_vehicle_ids) != imitation_scores.shape[0]:
            raise ValueError(
                "trip_vehicle_ids length must match imitation_scores length."
            )

        has_positive_by_vehicle: dict[int, bool] = {vehicle_id: False for vehicle_id in reject_vehicle_ids}
        for idx, vehicle_id in enumerate(trip_vehicle_ids):
            if vehicle_id in has_positive_by_vehicle and float(imitation_scores[idx].item()) > 0.0:
                has_positive_by_vehicle[vehicle_id] = True

        reject_scores = []
        for vehicle_id in reject_vehicle_ids:
            if has_positive_by_vehicle.get(vehicle_id, False):
                reject_scores.append(0.0)
            else:
                reject_scores.append(reject_penalty_value)
        reject_scores = torch.tensor(
            reject_scores,
            dtype=imitation_scores.dtype,
            device=imitation_scores.device,
        )
        return torch.cat([imitation_scores, reject_scores], dim=0)

    @staticmethod
    def get_y_star_best_ordered_match(imitation_scores: torch.Tensor) -> torch.Tensor:
        """
        Legacy global y* builder.

        Global maxima/minima are selected similarly to the historical behavior, but ties on non-zero maxima are treated as invalid and raise, because they indicate ambiguous supervision.

        Multi-vehicle flows should prefer `build_y_star_per_vehicle_from_scores`.
        """
        # NOTE this will not be able to handle multiple vehicles if the scores are not separated by vehicle.
        max_value = torch.max(imitation_scores)
        max_indices = torch.where(imitation_scores == max_value)[0]
        if float(max_value.item()) != 0.0 and len(max_indices) > 1:
            raise ValueError(
                "Ambiguous imitation scores: multiple non-zero global maxima found "
                f"(value={float(max_value.item())}, indices={max_indices.tolist()})."
            )
        # assert that there is only one maximum value
        # assert len(max_indices) == 1, f"There should be only one maximum value. {imitation_scores}"  
        # not required anymore as we have a negative value for the reject action per vehicle 
        # if all other values are zero or negative, set the minimum value to 1
        if torch.all(imitation_scores <= 0):
            min_value = torch.min(imitation_scores)
            min_indices = torch.where(imitation_scores == min_value)[0]
            y_star = torch.zeros_like(imitation_scores)
            y_star[min_indices] = 1
            return y_star
        y_star = torch.zeros_like(imitation_scores)
        y_star[max_indices] = 1
        return y_star

    @staticmethod
    def build_y_star_per_vehicle_from_imit_scores(
        imitation_scores_with_reject: torch.Tensor,
        trip_vehicle_ids: list[int],
        reject_vehicle_ids: list[int],
    ) -> torch.Tensor:
        """
        Build a binary y* with exactly one selected action per vehicle.

        Inputs follow the shared layout contract:
        - first `len(trip_vehicle_ids)` entries are trip scores
        - tail entries are reject scores ordered by `reject_vehicle_ids`

        Selection rule for each vehicle:
        - choose max candidate score among that vehicle's trip indices and its reject index (if available)
        - when all candidate scores are non-positive, choose minimum value (preserves existing reject-penalty semantics)
        - if there are multiple non-zero maxima imitation scores for the same vehicle, raise a ValueError because imitation learning would be ambiguous
        """
        trip_count = len(trip_vehicle_ids)
        if imitation_scores_with_reject.ndim != 1:
            raise ValueError(
                f"imitation_scores_with_reject must be a 1D tensor. Got {imitation_scores_with_reject.ndim} dimensions."
            )
        expected_length = trip_count + len(reject_vehicle_ids)
        if imitation_scores_with_reject.shape[0] != expected_length:
            raise ValueError(
                "Invalid score vector length for per-vehicle y* construction: "
                f"got {imitation_scores_with_reject.shape[0]}, "
                f"expected {expected_length}."
            )

        trip_indices_by_vehicle: dict[int, list[int]] = {}
        for trip_idx, vehicle_id in enumerate(trip_vehicle_ids):
            trip_indices_by_vehicle.setdefault(vehicle_id, []).append(trip_idx)

        reject_index_by_vehicle = {
            vehicle_id: trip_count + reject_idx
            for reject_idx, vehicle_id in enumerate(reject_vehicle_ids)
        }

        all_vehicle_ids = set(trip_indices_by_vehicle.keys()) | set(reject_vehicle_ids)
        y_star = torch.zeros_like(imitation_scores_with_reject)
        for vehicle_id in sorted(all_vehicle_ids):
            candidate_indices = list(trip_indices_by_vehicle.get(vehicle_id, []))
            if vehicle_id in reject_index_by_vehicle:
                candidate_indices.append(reject_index_by_vehicle[vehicle_id])
            if not candidate_indices:
                continue

            candidate_indices.sort()
            candidate_scores = imitation_scores_with_reject[candidate_indices]
            max_candidate_value = torch.max(candidate_scores)
            max_candidate_indices_local = torch.where(candidate_scores == max_candidate_value)[0]
            if float(max_candidate_value.item()) != 0.0 and len(max_candidate_indices_local) > 1:
                tied_global_indices = [candidate_indices[idx] for idx in max_candidate_indices_local.tolist()]
                raise ValueError(
                    "Ambiguous imitation scores for vehicle: multiple non-zero maxima found "
                    f"(vehicle_id={vehicle_id}, value={float(max_candidate_value.item())}, "
                    f"indices={tied_global_indices})."
                )
            if torch.all(candidate_scores <= 0):
                target_value = torch.min(candidate_scores)
            else:
                target_value = torch.max(candidate_scores)

            selected_index = None
            for global_idx in candidate_indices:
                if float(imitation_scores_with_reject[global_idx].item()) == float(target_value.item()):
                    selected_index = global_idx
                    break
            if selected_index is None:
                raise RuntimeError(
                    f"Failed to select y* action for vehicle {vehicle_id}."
                )
            y_star[selected_index] = 1
        return y_star


if __name__ == "__main__":
    solution = [1, 3, 2]
    trips = [0, 1, 2, 3, 4, 5, 6, 7]
    min_cardinality = 1
    max_cardinality = len(trips)

    combinations, counts = ImitationHandler.get_trip_index_combinations(
        trips=trips,
        # min_cardinality=,
        # max_cardinality=4,
    )
    print(counts)
    scores = ImitationHandler.score_combinations_against_solution(combinations, solution)

    # for score, combination in zip(scores, combinations):
    #     print(score, combination)

    np_scores = np.array(scores)

    # give me all indices without value 0
    non_zero_indices = np.where(np_scores != 0)[0]
    for idx in non_zero_indices:
        print(idx, combinations[idx], scores[idx])
    # max_idx = np.argmax(np_scores)
    # print(combinations[np.argmax(np_scores)], scores)
    # np.argmin(np_scores)
    # print(combinations[np.argmin(np_scores)])



    


    # i need a function that adds a bit of logic: create a list that accounts for the final result; it goes through all the combinations if the combinations perfectly matches including the order, I want to get a value of 100 in the final list; if it is a perfect match but is not ordered correctly, I want to get a value of 10 in the final list; if it is only a partial match of the optimal solution but the entire combination is part of the optimal solution but does not cover the full solution, I want to get a value of the length of that combination. If there is no match, I want to have a value of 0. iterate over it and use existing functions.