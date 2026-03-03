import json
import itertools
import numpy as np

from rtv_solver.structure.config import Config
from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.request import Request
from rtv_solver.structure.driver_run import DriverRun, ManifestEntry

from rtv_solver.handlers.payload_parser import PayloadParser

class ImitationHandler:
    """
    Handles the generation of the y* list of the correct solution for the CO-layer.
    """
    def __init__(self, config: Config):
        self.config = config
        self.optimal_solution = self._load_complete_optimal_solution()

    def get_optimal_request_solution_for_batch(self, request_batch: list[Request]) -> list[int]:
        """
        Get the optimal solution for a given batch of requests.
        """
        # TODO fix this
        # find the longest subsequence of requests in the optimal solution that is contained in the request batch
        len_longest_subsequence = 0
        longest_subsequence = []
        best_sequence_driver_id = None
        
        if len(request_batch) == 0:
            return longest_subsequence, best_sequence_driver_id
        
        request_ids = [request.id for request in request_batch]
        
        for driver_id, solution in self.optimal_solution.items():
            subsequence = [x for x in solution if x in request_ids]
            if len(subsequence) > len_longest_subsequence:
                len_longest_subsequence = len(subsequence)
                longest_subsequence = subsequence
                best_sequence_driver_id = driver_id
                
        return longest_subsequence, best_sequence_driver_id

    def get_optimal_stop_sequence_for_batch(self, request_batch: list[Request]) -> list[int]:
        pass

    def _load_complete_optimal_solution(self):
        """
        Load the complete optimal solution from the payload for the entire period.
        """
        if self.config.IMITATION_SOLUTION_FILE is None:
            raise ValueError("No imitation solution file provided.")
        
        with open(self.config.IMITATION_SOLUTION_FILE, "r") as f:
            payload_data = json.load(f)
        
        # driverID: solution list of request IDs
        optimal_solution = {}
        for driver_run in payload_data[PayloadParser.DRIVERS]:
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
            if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                stop_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
                solution.append(stop_id)
            if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_DROPOFF:
                stop_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
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
        Get all possible combinations of trips of a given cardinality.
        Mostly for testing purposes.

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
            cardinality_combinations = list(itertools.combinations(range(n_trips), cardinality))
            counts_by_cardinality[cardinality] = len(cardinality_combinations)
            combinations.extend(cardinality_combinations)

        return combinations, counts_by_cardinality

    @staticmethod
    def score_combinations_against_solution(
        combinations: list[tuple[int, ...]], optimal_solution: list[int]
    ) -> list[int]:
        """
        Score each combination against an optimal solution sequence.

        Parameters:
            - combinations: list of combinations of request indices
            - optimal_solution: list of request indices of the optimal solution sequence

        Scoring rules:
            - 1000: perfect full match with the complete optimal solution and same order
            - ordered full match for the combination length:
                - 100 + 10 * len(combination), if it starts at index 0
                - -10 * len(combination), if it starts at a later index
            - unordered full match for the combination length:
                - -100 - 10 * len(combination)
            - 0: no match

        NOTE This could be an interesting experiment, which assignment leads to the best results. We would probably take the the max or min score for further evaluation.
        """
        scores: list[int] = []
        optimal_tuple = tuple(optimal_solution)

        for combination in combinations:
            # Perfect match with the complete optimal solution.
            if combination == optimal_tuple:
                scores.append(1000)
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
                    scores.append(100 + 10 * len(combination))
                else:
                    scores.append(0 - 10 * len(combination)) # - (ordered_start_index * 10 * len(combination))
                continue

            # Full coverage for this combination length but wrong order.
            if permuted_start_index is not None:
                scores.append( - 100 - 10 * len(combination) )
                continue

            scores.append(0)
        return scores


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