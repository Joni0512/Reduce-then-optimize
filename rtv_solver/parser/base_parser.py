from abc import ABC, abstractmethod


class BaseParser(ABC):
    
    @staticmethod
    @abstractmethod
    def parse_file(filepath, **kwargs) -> dict:
        """
        Parse an instance file so it can be used by the PayloadParser based on the 'wilson' format. Some of the dicts might have additional keys that are not part of the 'wilson' format.

        Returns:
            dict with keys: requests, depot, driver_runs, travel_time_matrix
        """
        # TODO update this in order to only pass the dict instead of actually loading the data, that should be separate
        raise NotImplementedError("Subclasses must implement this method")
