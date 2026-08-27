"""
request_pruner_dataset_builder.py
==================================
2026-07-14: Phase 0 (step 2) of the request-pruner design discussion.

Turns the per-window rolling-horizon baseline logs (assignment_data.jsonl,
one per li_lim instance under outputs/eval_rh_no_learning/{instance}/rh_no_learning/)
into a fixed, reusable dataset on disk - so that later scripts (feature
experiments, the eventual RequestPrunerMLP training script) don't need to
re-parse jsonl logs and rebuild request graphs from scratch every time they
run, the way request_pruner_signal_check.py currently does.

Relationship to request_pruner_signal_check.py (the earlier throwaway
analysis script that validated there IS signal here - see that file's
docstring for the full history of how the label definition was found):
    - Same window-replay idea: read assignment_data.jsonl, figure out which
      requests were active in each rolling-horizon iteration, compute the
      same feature set for them.
    - Different output. The signal-check script z-scores every feature
      immediately (per-window for the 9 base+urgency features, train-set-
      fixed for the 4 context features) because it only ever trains one
      throwaway linear probe in memory and then discards everything. This
      script stores RAW, UNNORMALIZED values instead: normalization is a
      training-time decision (which statistics to fix, whether validation
      windows may influence them, etc. - see the "why ddof=0" and "why
      context features can't be per-window normalized" comments in the
      signal-check script for how easy it is to get this wrong) and baking
      one specific normalization choice into the dataset file would lock
      every future consumer into it, silently.
    - Deliberately duplicates the ~40 lines of logic that differ (node-only
      graph construction, per-window feature computation) instead of
      importing the private helpers from request_pruner_signal_check.py.
      That script is already validated and its printed results are
      referenced in the design discussion - refactoring it into a shared
      module now, just to serve this one additional caller, risks silently
      changing behavior in code whose output has already been checked
      line-by-line. Two call sites with different needs (normalize-and-
      discard vs. persist-raw) is exactly the case where a bit of
      duplication beats a shared abstraction.

Output layout:
    dataset/request_pruning/{train,val,test}/{instance}.npz
        One .npz per instance, containing (all arrays same length = number
        of (window, request) rows for that instance):
            node_ids     int64   [n_rows]   request id per row
            window_index int64   [n_rows]   which rolling-horizon iteration
            timestamp    float32 [n_rows]   that window's current_time
            label        float32 [n_rows]   1.0 = assigned this window, 0.0 = unserved
            features     float32 [n_rows, n_features]  RAW feature values,
                                             column order given by FEATURE_NAMES
                                             below (also written to manifest.json)
    dataset/request_pruning/manifest.json
        Build metadata: feature column names/groups, per-instance/per-split
        row and window counts, git-independent build timestamp. Read this
        first when writing a training script against this dataset - it's
        the single source of truth for what column i in `features` means.

Split: identical to the pair pruner's "mixed_all" config
(train_models_v2.MODEL_CONFIGS), so both pruners are trained/evaluated on
the same instances and results are comparable.

Run:
    python3 -m rtv_solver.pipeline.request_pruner_dataset_builder
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.train_models_v2 import MODEL_CONFIGS
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config


MANIFEST_DIR = Path("solutions/li_lim/manifests")  # static request data (lat/lon, time windows) per instance
RH_BASELINE_DIR = Path("outputs/eval_rh_no_learning")  # per-instance rolling-horizon baseline logs, USE_REQUEST_GRAPH_PRUNER=false
OUTPUT_DIR = Path("dataset/request_pruning")

# Real train/val/test split across all 56 li_lim instances - identical to
# what the pair pruner trains/evaluates on (see train_models_v2.py:71-166).
REQUEST_PRUNER_SPLIT = MODEL_CONFIGS["mixed_all"]

# Fixed column order for the "features" array in every .npz file. Grouped
# the same way the signal-check script grouped them, kept here as a single
# ordered list because the .npz files store raw numpy arrays with no column
# names attached - this list (also dumped into manifest.json) is the only
# record of which column is which.
BASE_FEATURES = RequestGraphFeatureBuilder.NODE_FEATURES  # 9 features, see that class
URGENCY_FEATURES = ["time_until_latest_pickup", "time_since_earliest_pickup"]
CONTEXT_FEATURES_STATIC = ["num_active_requests", "requests_per_vehicle"]
# Kept in the STORED dataset even though the signal check found this proxy
# WORSE than the static one (test AUC 0.768 vs 0.832, see
# request_pruner_signal_check.py's last run) - cheap to compute and store,
# useful for future feature-selection experiments without re-parsing jsonl
# logs again.
#
# 2026-07-14: MUST NOT be used to train the deployable model, though - see
# DEPLOYABLE_FEATURE_NAMES below. Both of these are computed from
# status["assigned_requests"].values() in build_instance_arrays() further
# down, i.e. from the SOLVER'S OWN OUTPUT for that rolling-horizon window
# (which vehicle ended up serving which request). The request pruner is
# meant to run BEFORE trip generation/the ILP solve, to shrink its input -
# at that point "which vehicles got used this window" doesn't exist yet, it
# IS the thing being solved for. Training on these two columns would produce
# a model whose inputs can't actually be computed at inference time. Caught
# while designing the runtime RequestPruner class, which is also why the
# distinction lives here as a second named list rather than a comment only -
# train_request_pruner.py's load_split() slices down to
# DEPLOYABLE_FEATURE_NAMES so this mistake can't silently resurface.
CONTEXT_FEATURES_IDLE_PROXY = ["num_idle_vehicles_proxy", "requests_per_idle_vehicle"]
FEATURE_NAMES = BASE_FEATURES + URGENCY_FEATURES + CONTEXT_FEATURES_STATIC + CONTEXT_FEATURES_IDLE_PROXY

# The subset of FEATURE_NAMES that's actually computable BEFORE the ILP
# solve runs, i.e. safe to use for the real, deployed model. Relies on
# CONTEXT_FEATURES_IDLE_PROXY being the last two entries of FEATURE_NAMES -
# asserted below so a future reordering can't silently break this slice.
DEPLOYABLE_FEATURE_NAMES = BASE_FEATURES + URGENCY_FEATURES + CONTEXT_FEATURES_STATIC
assert FEATURE_NAMES[: len(DEPLOYABLE_FEATURE_NAMES)] == DEPLOYABLE_FEATURE_NAMES


def _build_node_only_graph(requests) -> nx.Graph:
    """
    Nodes only, no edges - the request-only pruner never needs request-
    request edges, so building them (RequestGraphFullBuilder.build()'s
    O(n^2) itertools.combinations step) would be wasted work here, same
    reasoning as in request_pruner_signal_check.py.
    """
    G = nx.Graph()
    for req in requests:
        G.add_node(req.id, request=req)
    return G


def _load_instance_data(instance_name: str, config: Config) -> tuple[dict[int, object], int]:
    """
    Returns (request_lookup, num_vehicles) for one instance - full Request
    objects keyed by id (static per request, reused across all of that
    instance's windows) plus the instance's total fleet size (static count
    of driver_runs entries, used for the "requests_per_vehicle" feature).
    """
    payload_path = MANIFEST_DIR / f"{instance_name}.json"
    payload = PayloadParser.load_input_data(payload_path)

    # Required before RequestHandler can build Request objects: it resolves
    # pickup/dropoff node_ids via NetworkHandler internally.
    NetworkHandler.init_from_payload(payload=payload, server_url=config.SERVER_URL)

    request_handler = RequestHandler(payload[PayloadKeys.REQUESTS], config)
    request_lookup = {req.id: req for req in request_handler.get_all_requests()}
    num_vehicles = len(payload[PayloadKeys.DRIVERS])
    return request_lookup, num_vehicles


def build_instance_arrays(
    instance_name: str,
    config: Config,
    rh_baseline_dir: Path = RH_BASELINE_DIR,
    geographic: bool = False,
) -> dict | None:
    """
    Replays one instance's rolling-horizon log window by window and returns
    the raw (unnormalized) arrays ready to be written to an .npz file.
    Returns None if the instance has no log yet (shouldn't happen after
    Phase 0, but build_dataset() below skips gracefully rather than crash a
    56-instance batch over one missing file).

    2026-07-15: rh_baseline_dir made a parameter (default unchanged, still
    the original 400/100 baseline dir) so this can build a dataset for a
    DIFFERENT window config's logs (e.g. outputs/eval_rh_no_learning_bi200_ss100/)
    without touching the 400/100 dataset - see build_window_specific_request_pruners.py.
    Window-size robustness is handled by training one specialized model per
    window config instead of merging configs into one dataset (see design
    discussion 2026-07-15: models are always evaluated against one known
    window config at a time, so a merged/generalist dataset wasn't needed).
    """
    log_path = rh_baseline_dir / instance_name / "rh_no_learning" / "assignment_data.jsonl"
    if not log_path.exists():
        print(f"  [skip] no rolling-horizon log for {instance_name}: {log_path}")
        return None

    request_lookup, num_vehicles = _load_instance_data(instance_name, config)

    all_node_ids: list[int] = []
    all_window_index: list[int] = []
    all_timestamp: list[float] = []
    all_labels: list[float] = []
    all_feature_rows: list[list[float]] = []

    with open(log_path) as f:
        for window_index, line in enumerate(f):
            entry = json.loads(line)
            status = entry["extra"]["status"]
            assigned_ids = {int(rid) for rid in status["assigned_requests"].keys()}
            unserved_ids = {int(rid) for rid in status["unserved_requests"]}
            active_ids = assigned_ids | unserved_ids

            if len(active_ids) == 0:
                continue

            active_requests = [request_lookup[rid] for rid in active_ids]
            timestamp = entry["extra"]["timestamp"]

            G = _build_node_only_graph(active_requests)
            # 2026-08-21: geographic threaded through (default False keeps
            # Li&Lim behavior byte-identical) - see RequestPruner.__init__
            # for why NYC needs geographic=True (haversine vs. Li&Lim's
            # flat-grid Euclidean coordinates).
            G = RequestGraphFeatureBuilder.add_node_features(G, geographic=geographic)
            node_matrix, _edge_index, _edge_matrix, node_ids = RequestGraphFeatureBuilder.to_numpy(G)

            # Sanity check: every node we built a feature row for must be
            # either assigned or unserved this window, and vice versa - see
            # request_pruner_signal_check.py for why this can catch a
            # mismatch between the static manifest file and the jsonl log.
            assert set(node_ids) == active_ids, (
                f"{instance_name} window {window_index}: node ids don't match "
                f"active ids from the log"
            )

            # Idle-vehicle proxy inputs, computed once per window (same
            # value for every request in it) - see CONTEXT_FEATURES_IDLE_PROXY
            # comment above for why this is stored despite being the weaker
            # of the two context-feature variants.
            vehicles_used_this_window = {int(vid) for vid in status["assigned_requests"].values()}
            num_idle_vehicles_proxy = max(num_vehicles - len(vehicles_used_this_window), 0)

            for row_index, node_id in enumerate(node_ids):
                req = request_lookup[node_id]

                base_values = node_matrix[row_index].tolist()  # 9 base features, already in BASE_FEATURES order
                urgency_values = [
                    req.latest_pickup_time - timestamp,       # time_until_latest_pickup
                    timestamp - req.earliest_pickup_time,     # time_since_earliest_pickup
                ]
                context_static_values = [
                    float(len(active_ids)),                   # num_active_requests
                    len(active_ids) / num_vehicles,            # requests_per_vehicle
                ]
                context_idle_values = [
                    float(num_idle_vehicles_proxy),                              # num_idle_vehicles_proxy
                    len(active_ids) / (num_idle_vehicles_proxy + 1),             # requests_per_idle_vehicle (+1 guard, see signal-check script)
                ]

                all_feature_rows.append(
                    base_values + urgency_values + context_static_values + context_idle_values
                )
                all_node_ids.append(int(node_id))
                all_window_index.append(window_index)
                all_timestamp.append(float(timestamp))
                all_labels.append(1.0 if node_id in assigned_ids else 0.0)

    if len(all_node_ids) == 0:
        return None

    return {
        "node_ids": np.array(all_node_ids, dtype=np.int64),
        "window_index": np.array(all_window_index, dtype=np.int64),
        "timestamp": np.array(all_timestamp, dtype=np.float32),
        "label": np.array(all_labels, dtype=np.float32),
        "features": np.array(all_feature_rows, dtype=np.float32),
        "num_windows": int(max(all_window_index) + 1),
    }


def build_dataset(rh_baseline_dir: Path = RH_BASELINE_DIR, output_dir: Path = OUTPUT_DIR) -> None:
    """
    2026-07-15: rh_baseline_dir/output_dir made parameters (defaults
    unchanged) so the same builder can produce a SEPARATE, window-config-
    specific dataset (e.g. for BATCH_INTERVAL=200/STEP_SIZE=100 logs) without
    touching the original 400/100 dataset under dataset/request_pruning/.
    See build_window_specific_request_pruners.py, which calls this once per
    additional window config.
    """
    config = Config()
    _SPLIT_LOOKUP = {
        name: split for split, names in REQUEST_PRUNER_SPLIT.items() for name in names
    }

    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rh_baseline_dir": str(rh_baseline_dir),
        "feature_names": FEATURE_NAMES,
        "feature_groups": {
            "base": BASE_FEATURES,
            "urgency": URGENCY_FEATURES,
            "context_static": CONTEXT_FEATURES_STATIC,
            "context_idle_proxy": CONTEXT_FEATURES_IDLE_PROXY,
        },
        "label_definition": (
            "1.0 if the request appears in assigned_requests for that "
            "rolling-horizon window (got a vehicle THIS window), else 0.0 "
            "(unserved this window, rolls over to the next one). Source: "
            f"{rh_baseline_dir}/<instance>/rh_no_learning/assignment_data.jsonl, "
            "the USE_REQUEST_GRAPH_PRUNER=false baseline run."
        ),
        "normalization": (
            "NOT applied - features are raw. Normalize at train time using "
            "TRAIN-split statistics only (see request_pruner_signal_check.py "
            "for the per-window vs. train-set-fixed distinction that matters "
            "for the context features)."
        ),
        "split_source": "train_models_v2.MODEL_CONFIGS['mixed_all']",
        "instances": {},
    }

    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for instance_name in REQUEST_PRUNER_SPLIT[split]:
            print(f"  building {split}/{instance_name} ...")
            arrays = build_instance_arrays(instance_name, config, rh_baseline_dir=rh_baseline_dir)
            if arrays is None:
                continue

            out_path = split_dir / f"{instance_name}.npz"
            np.savez(
                out_path,
                node_ids=arrays["node_ids"],
                window_index=arrays["window_index"],
                timestamp=arrays["timestamp"],
                label=arrays["label"],
                features=arrays["features"],
            )

            num_rows = len(arrays["node_ids"])
            manifest["instances"][instance_name] = {
                "split": split,
                "num_windows": arrays["num_windows"],
                "num_rows": num_rows,
                "positive_rate": float(arrays["label"].mean()),
            }
            print(f"    -> {out_path} ({arrays['num_windows']} windows, {num_rows} rows)")

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_rows = sum(v["num_rows"] for v in manifest["instances"].values())
    total_windows = sum(v["num_windows"] for v in manifest["instances"].values())
    print(f"\nDONE: {len(manifest['instances'])} instances, {total_windows} windows, {total_rows} rows")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    build_dataset()
