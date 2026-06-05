import numpy as np
import pandas as pd


class RequestGraphLabelBuilder:
    """
    Baut Labels für Request-Request-Edges.

    Heute nutzen wir erstmal Heuristik-Labels:
        1 = Edge sieht sinnvoll aus
        0 = Edge sieht schlecht aus

    Später ersetzen wir diese Heuristik durch echte Labels aus der ILP-/Simulation-Lösung.
    """

    @staticmethod
    def build_heuristic_labels(
        edge_df: pd.DataFrame,
        max_pickup_time_difference: float = 3600.0,
        min_direction_similarity: float = 0.0,
        require_pickup_overlap: bool = True,
    ) -> np.ndarray:
        """
        Erzeugt ein Binary-Label pro Edge.

        Input:
            edge_df:
                DataFrame mit Edge-Features, z.B.
                pickup_time_difference
                pickup_window_overlap_seconds
                direction_similarity

        Output:
            labels:
                np.ndarray mit Shape [num_edges]
                1.0 = gute Kante
                0.0 = schlechte Kante
        """

        # Regel 1:
        # Requests sollen zeitlich nicht zu weit auseinander liegen.
        time_ok = (
            edge_df["pickup_time_difference"]
            <= max_pickup_time_difference
        )

        # Regel 2:
        # Requests sollen ungefähr in ähnliche Richtung gehen.
        direction_ok = (
            edge_df["direction_similarity"]
            > min_direction_similarity
        )

        # Regel 3:
        # Pickup-Zeitfenster sollen sich überschneiden.
        if require_pickup_overlap:
            overlap_ok = (
                edge_df["pickup_window_overlap_seconds"] > 0
            )
        else:
            overlap_ok = True

        labels = time_ok & direction_ok & overlap_ok

        return labels.astype(float).to_numpy()