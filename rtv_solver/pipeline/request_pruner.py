"""
request_pruner.py
==================
2026-07-14: Phase 3 (runtime pruner) of the request-pruner design discussion.

Runtime counterpart to RequestGraphPruner (request_graph_pruner.py) - plays
the same role for individual REQUESTS that RequestGraphPruner plays for
REQUEST PAIRS: given the currently active requests for a rolling-horizon
window, decide which ones go on to trip generation this round and which get
deferred to a later window (NOT rejected - see filter_requests docstring).

API deliberately mirrors RequestGraphPruner:
    RequestGraphPruner.build_allowed_pairs(requests) -> (allowed_pairs, stats)
    RequestPruner.filter_requests(requests, current_time, num_vehicles) -> (kept_requests, stats)

Why the force-keep rules exist (the actual question that led to this class,
design discussion 2026-07-14): pruning a whole REQUEST is a much bigger
consequence than pruning one PAIR - a wrongly-dropped request loses ALL its
options this window, not just one combination with one other request. So on
top of the model score, two categories of request are force-kept
unconditionally, regardless of what the model predicts:

1. URGENT: latest_pickup_time is coming up within `urgent_slack_seconds`.
   Model scoring only ever applies to requests that still have enough slack
   for a wrong call to be recoverable next window.
2. ACTIVE (added 2026-07-16, see below): already committed to a vehicle in a
   prior rolling-horizon iteration. Pruning one of these isn't just a missed
   opportunity like an urgent request - it breaks a hard invariant
   downstream (KEEP_ACTIVE / _check_consistency_of_manifests expects every
   active request to still be present) and previously crashed with
   ManifestConsistencyError the first time it was actually exercised (found
   running the COAML/OPT path against lc208).

2026-07-16 correction: an earlier version of this docstring claimed
"already-boarded requests never reach this class at all... 'already
committed to a vehicle' is handled elsewhere and by construction never shows
up here." That conflated two different things. It's true for BOARDED
requests (physically already picked up) - those ARE excluded from
request_batch upstream, in both online_rtv_solver.py and
coaml_pipeline.py's solve_iteration (RequestHandler.get_request_batches
puts them in a separate boarded_requests dict, never in request_batch).
It is NOT true for ACTIVE requests (accepted in a prior iteration but not
yet boarded) - get_request_batches puts those in request_batch too (see its
loop: the `else` branch appends to request_batch regardless of whether the
id is also active), specifically so the KEEP_ACTIVE ILP constraint can find
them. That means active requests were reaching RequestPruner all along, with
no protection beyond happening to also be urgent - filter_requests() now
takes active_request_ids explicitly instead of assuming they can't appear.

Feature computation deliberately duplicates the small amount of logic
(~30 lines: node-only graph builder, per-request feature assembly) already
written in request_pruner_dataset_builder.py and request_pruner_signal_check.py,
rather than importing it from either. This is runtime-critical code, called
every rolling-horizon iteration from inside TripHandler - kept self-
contained rather than depending on an offline dataset-building script or a
throwaway analysis script.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.request_pruner_mlp import RequestPrunerMLP


def _build_node_only_graph(requests) -> nx.Graph:
    """
    Nodes only, no edges - the request-only pruner never needs request-
    request edges, so building them (RequestGraphFullBuilder.build()'s
    O(n^2) itertools.combinations step) would be wasted work on every single
    rolling-horizon iteration. RequestGraphFeatureBuilder.add_node_features()
    only reads G.nodes, so this is a safe drop-in.
    """
    G = nx.Graph()
    for req in requests:
        G.add_node(req.id, request=req)
    return G


class RequestPruner:
    """
    Filters active requests down to the subset that goes on to trip
    generation this rolling-horizon window.

    Usage (mirrors RequestGraphPruner):
        pruner = RequestPruner(model_path=..., threshold=0.3)
        kept_requests, stats = pruner.filter_requests(
            requests=active_requests,
            current_time=current_time,
            num_vehicles=len(vehicle_handler.vehicles),
        )

    Geprunte Requests werden hier nur zurückgehalten, nicht gelöscht: der
    Aufrufer entscheidet, was mit den nicht zurückgegebenen Requests
    passiert (im TripHandler-Kontext: sie werden diese Iteration einfach
    nicht an generate_ondemand_only_trips/generate_shared_trips
    weitergereicht, bleiben aber im RequestHandler-Zustand für das nächste
    Fenster aktiv - siehe TripHandler-Integration).
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.3,
        urgent_slack_seconds: float = 100.0,
        min_retention_fraction: float = 0.3,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        # 2026-07-15: hard safety floor, added after the lr211/lrc208
        # investigation showed that neither more training data nor
        # window-config-specific retraining fixed a specific failure mode:
        # on rare, underrepresented "early-horizon backlog" windows (many
        # candidates, unusually high requests_per_vehicle), the model scored
        # almost every candidate below threshold and pruned nearly the whole
        # window (e.g. 0/47 kept) - a miscalibration that neither more data
        # nor a window-matched dataset removed, because that scenario stays
        # rare in the training population regardless of window config.
        # min_retention_fraction guarantees AT LEAST this fraction of
        # CANDIDATES (force-kept requests are separate and unaffected) is
        # kept regardless of model confidence, by topping up with the
        # highest-scoring pruned candidates if the threshold-based selection
        # falls short - see the floor logic in filter_requests(). This
        # doesn't fix the model's calibration, it just bounds the damage a
        # miscalibrated window can do.
        self.min_retention_fraction = min_retention_fraction
        # 2026-07-14: default matches li_lim's STEP_SIZE (100s), NOT
        # BATCH_INTERVAL (400s) - tried BATCH_INTERVAL first and it was far
        # too conservative: verified empirically on lc108 that it force-keeps
        # 70-94% of requests every window (0-1 actually prunable), because
        # li_lim's pickup windows are already tight relative to a full
        # BATCH_INTERVAL. The correct safety margin is STEP_SIZE: that's how
        # far current_time advances before this request would be reconsidered
        # at all (see offline_rtv_solver.py's `current_time += step_size`) -
        # if latest_pickup_time is closer than that, there may be no
        # meaningful next window left to retry in, so it's unsafe to leave
        # that call to a model that was never told "you are about to make
        # this request infeasible". With STEP_SIZE=100 on lc108, force-keeps
        # dropped to 0-6/window while pruning 0-4/window, with zero recall
        # loss on served requests in that test run. Not tied to whatever
        # STEP_SIZE the caller is actually using - pass a matching value
        # explicitly if it differs from li_lim's default.
        self.urgent_slack_seconds = urgent_slack_seconds
        self.device = torch.device(device)

        self.model = None
        self.feature_names: list[str] | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def filter_requests(
        self,
        requests: list,
        current_time: float,
        num_vehicles: int,
        active_request_ids: set | None = None,
    ) -> tuple[list, dict]:
        """
        Main function.

        Input:
            requests:
                Active Request objects for the current rolling-horizon
                window (same population TripHandler.generate_ondemand_only_trips
                would otherwise see unfiltered).
            current_time:
                Current simulation time, same clock as request.earliest_pickup_time
                / request.latest_pickup_time.
            num_vehicles:
                Current fleet size - used for the "requests_per_vehicle"
                context feature, same definition used at training time
                (len(payload["driver_runs"]); at runtime, len(vehicle_handler.vehicles)).
            active_request_ids:
                2026-07-16: request ids already committed to a vehicle in a
                prior rolling-horizon iteration (accepted, not yet boarded).
                Force-kept unconditionally, same as urgent requests - see
                module docstring for why this parameter exists now instead
                of being assumed away. Pass the caller's `active_requests`
                keys (e.g. TripHandler.active_requests); None/empty means
                "no active-request protection", NOT "there are none" - only
                pass None if the caller genuinely has no such concept.

        Output:
            kept_requests:
                Subset of `requests` to pass on to trip generation this
                window - force-kept (active or urgent) requests plus
                whichever candidates scored >= threshold.
            stats:
                dict with pruning statistics, same shape/spirit as
                RequestGraphPruner.build_allowed_pairs's stats dict.
        """
        start_time = time.time()
        active_request_ids = active_request_ids or set()

        if len(requests) == 0:
            return [], {
                "num_requests_total": 0,
                "num_force_kept_active": 0,
                "num_force_kept_urgent": 0,
                "num_candidates": 0,
                "num_kept_by_model": 0,
                "num_floor_kept": 0,
                "num_kept_total": 0,
                "num_pruned": 0,
                "retention": 0.0,
                "threshold": self.threshold,
                "min_retention_fraction": self.min_retention_fraction,
                "urgent_slack_seconds": self.urgent_slack_seconds,
                "inference_time": time.time() - start_time,
            }

        # Two force-keep rules, checked in this order (active first: it's a
        # hard correctness requirement, not just a safety margin like
        # urgency is). See module docstring for why both exist.
        force_kept = []
        candidates = []
        num_force_kept_active = 0
        num_force_kept_urgent = 0
        for req in requests:
            if req.id in active_request_ids:
                force_kept.append(req)
                num_force_kept_active += 1
            elif req.latest_pickup_time - current_time <= self.urgent_slack_seconds:
                force_kept.append(req)
                num_force_kept_urgent += 1
            else:
                candidates.append(req)

        if len(candidates) == 0:
            return force_kept, {
                "num_requests_total": len(requests),
                "num_force_kept_active": num_force_kept_active,
                "num_force_kept_urgent": num_force_kept_urgent,
                "num_candidates": 0,
                "num_kept_by_model": 0,
                "num_floor_kept": 0,
                "num_kept_total": len(force_kept),
                "num_pruned": 0,
                "retention": 1.0,
                "threshold": self.threshold,
                "min_retention_fraction": self.min_retention_fraction,
                "urgent_slack_seconds": self.urgent_slack_seconds,
                "inference_time": time.time() - start_time,
            }

        self._load_model_if_needed()

        # Node/context features are built from ALL active requests
        # (force-kept ones included), not just candidates - at training time
        # "active requests in this window" always meant the whole window
        # population (the training data never knew about a force-keep
        # split), so context features like num_active_requests and
        # requests_per_vehicle, and the neighborhood-based
        # local_request_count_10min base feature, must be computed the same
        # way here or they'd silently mismatch what the model learned.
        # Candidates are simply the only rows actually fed into the model
        # afterwards - force-kept requests are never scored at all.
        G = _build_node_only_graph(requests)
        G = RequestGraphFeatureBuilder.add_node_features(G)
        node_matrix, _edge_index, _edge_matrix, node_ids = RequestGraphFeatureBuilder.to_numpy(G)
        node_id_to_base_features = {
            node_id: node_matrix[i] for i, node_id in enumerate(node_ids)
        }

        num_active_requests = float(len(requests))
        requests_per_vehicle = len(requests) / num_vehicles

        feature_rows = []
        for req in candidates:
            base_values = node_id_to_base_features[req.id].tolist()
            urgency_values = [
                req.latest_pickup_time - current_time,   # time_until_latest_pickup
                current_time - req.earliest_pickup_time,  # time_since_earliest_pickup
            ]
            context_values = [num_active_requests, requests_per_vehicle]
            feature_rows.append(base_values + urgency_values + context_values)

        features = np.array(feature_rows, dtype=np.float32)
        features_norm = (features - self.feature_mean) / self.feature_std

        x = torch.tensor(features_norm, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(self.model(x)).cpu().numpy()

        kept_mask = scores >= self.threshold

        # Hard safety floor: if the threshold-based selection keeps fewer
        # than min_retention_fraction of the candidates, top up with the
        # highest-scoring PRUNED candidates (least-confidently-pruned first)
        # until the floor is met. Triggers only on the rare miscalibrated
        # windows this was built for - on a normal window, the model already
        # keeps more than the floor and this is a no-op.
        min_keep_count = math.ceil(self.min_retention_fraction * len(candidates))
        num_floor_kept = 0
        if int(kept_mask.sum()) < min_keep_count:
            order = np.argsort(-scores)  # highest score first
            floor_mask = np.zeros(len(candidates), dtype=bool)
            floor_mask[order[:min_keep_count]] = True
            num_floor_kept = int((floor_mask & ~kept_mask).sum())
            kept_mask = kept_mask | floor_mask

        kept_by_model = [req for req, keep in zip(candidates, kept_mask) if keep]
        kept_requests = force_kept + kept_by_model

        stats = {
            "num_requests_total": len(requests),
            "num_force_kept_active": num_force_kept_active,
            "num_force_kept_urgent": num_force_kept_urgent,
            "num_candidates": len(candidates),
            "num_kept_by_model": len(kept_by_model),
            "num_floor_kept": num_floor_kept,
            "num_kept_total": len(kept_requests),
            "num_pruned": len(candidates) - len(kept_by_model),
            "retention": len(kept_requests) / len(requests),
            "threshold": self.threshold,
            "min_retention_fraction": self.min_retention_fraction,
            "urgent_slack_seconds": self.urgent_slack_seconds,
            "inference_time": time.time() - start_time,
        }
        return kept_requests, stats

    def _load_model_if_needed(self) -> None:
        if self.model is not None:
            return

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.feature_names = checkpoint["feature_names"]
        self.feature_mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.array(checkpoint["feature_std"], dtype=np.float32)

        self.model = RequestPrunerMLP(
            feature_dim=len(self.feature_names),
            hidden_dim=checkpoint["hidden_dim"],
            num_hidden_layers=checkpoint["num_hidden_layers"],
            dropout=checkpoint["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
