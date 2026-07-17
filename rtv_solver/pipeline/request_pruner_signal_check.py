"""
request_pruner_signal_check.py
===============================
2026-07-13: one-off analysis script, written before any request-pruner
infrastructure exists. Question it answers: does the *existing* per-request
node-feature set (RequestGraphFeatureBuilder.NODE_FEATURES - the same 8
features already used by the request-request pair pruner) carry any signal
for predicting whether a request gets assigned to a vehicle in the CURRENT
rolling-horizon window, vs. staying unserved and rolling over to the next
window?

This is deliberately NOT the request pruner. It does not save a model, does
not touch RequestHandler/TripHandler/Config, and does not create a dataset
folder. It is a throwaway signal check to decide whether it's worth building
the real pipeline (RequestPrunerLabelBuilder file, RequestPrunerMLP,
TripHandler integration, Main flags) described in the design discussion.

--- Version history of this script (why it looks the way it does) ---
v1 (first attempt) used the final, fully-converged solutions in
solutions/li_lim/manifests/*.json and labelled a request "served" if its
booking_id appeared ANYWHERE in the final driver manifests. That gave
positive_rate == 1.0 for every single instance: those files are the merged
END STATE of an entire rolling-horizon run (see offline_rtv_solver.py:19-82),
so by the end of the simulation basically every request has been picked up
by some vehicle eventually - the "not served in THIS window, rolls over to
the next one" case (the actual thing the request pruner is supposed to
predict) is invisible in that file.

v2 (this version) fixes that by using real per-window rolling-horizon logs
that already exist on disk from earlier pair-pruner evaluation runs:
    outputs/eval_rh_no_learning/{instance}/rh_no_learning/assignment_data.jsonl
Each line is one rolling-horizon iteration with the exact per-window ground
truth we need (see online_rtv_solver.py:157-183 for how these are computed):
    "assigned_requests": {request_id: vehicle_id, ...}   -> label 1 (kept this window)
    "unserved_requests": [request_id, ...]                -> label 0 (deferred to next window)
config.json in the same folder confirms USE_REQUEST_GRAPH_PRUNER=false, i.e.
this is the unpruned baseline - the correct "expert" ground truth to learn
from, analogous to how the pair pruner learns from unpruned RTV combos.

v3 (2026-07-14, Phase 0 of the design discussion): the coverage caveat from
v2 is now resolved. v2 only had per-window logs for the 12 li_lim instances
that happen to be the pair-pruner's "test" split (leftovers of prior
pair-pruner evaluation sweeps), so this script used a small, PROVISIONAL 6/6
instance split just to get a first read. The other 44 instances have since
been rolling-horizon-simulated with the identical baseline config
(run_request_pruner_missing_instances.sh, see that file for exactly how) and
now have assignment_data.jsonl too - all 56 li_lim instances are covered.
This version switches from the provisional PROBE_SPLIT to the REAL
train/val/test split already used for the pair pruner
(train_models_v2.MODEL_CONFIGS["mixed_all"]), so the two pruners are now
directly comparable on identical instances, and the signal-check result
rests on ~5x more windows than before.

Run:
    python3 -m rtv_solver.pipeline.request_pruner_signal_check
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
# 2026-07-14: reuse the pair-pruner's REAL instance split ("mixed_all" - all
# 6 li_lim families combined, see train_models_v2.py:71-166) instead of the
# provisional 6/6 split this script used before Phase 0 finished. Same
# instances, same train/val/test assignment as the pair pruner, so results
# from the two pruners are directly comparable and not an apples-to-oranges
# split mismatch.
from rtv_solver.pipeline.train_models_v2 import MODEL_CONFIGS
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config


MANIFEST_DIR = Path("solutions/li_lim/manifests")  # static request data (lat/lon, time windows) per instance
RH_BASELINE_DIR = Path("outputs/eval_rh_no_learning")  # per-instance rolling-horizon baseline logs, USE_REQUEST_GRAPH_PRUNER=false

# Real train/val/test split across all 56 li_lim instances (lc1/lr1/lrc1/
# lc2/lr2/lrc2 combined), identical to what the pair pruner trains/evaluates
# on. _SPLIT_LOOKUP maps each instance name -> "train"/"val"/"test" so
# build_window_samples can tag every window with its split membership.
REQUEST_PRUNER_SPLIT = MODEL_CONFIGS["mixed_all"]
_SPLIT_LOOKUP = {
    name: split for split, names in REQUEST_PRUNER_SPLIT.items() for name in names
}


def _build_node_only_graph(requests) -> nx.Graph:
    """
    Nodes only, no edges.

    Unlike RequestGraphFullBuilder.build(), this deliberately skips the
    O(n^2) itertools.combinations edge construction: the request-only pruner
    never uses request-request edges, so building them here would just be
    wasted work repeated for every single rolling-horizon window. This
    matters more here than it did in v1 (single build per instance) because
    this version builds one graph PER WINDOW (dozens of windows x 12
    instances). RequestGraphFeatureBuilder.add_node_features() only reads
    G.nodes, so a node-only graph is a safe drop-in.
    """
    G = nx.Graph()
    for req in requests:
        G.add_node(req.id, request=req)
    return G


# 2026-07-13 (step 5 of the design discussion): extra feature groups on top
# of the original 8 pair-pruner features, added to check whether
# request-only signal (AUC 0.697 on held-out instances, see script history
# below) can be improved before investing in the full MLP pipeline.
#   - "urgency" features vary PER REQUEST within a window (each request has
#     its own pickup window), so they're treated exactly like the existing 8
#     node features (see build_window_samples): per-window z-scored together
#     with them.
#   - "context" features are constant ACROSS all requests within one window
#     (there's only one "how many requests are active right now" value per
#     window) - these need different handling, see the normalization note in
#     main().
URGENCY_FEATURES = ["time_until_latest_pickup", "time_since_earliest_pickup"]

# CONTEXT_FEATURES_STATIC uses the instance's total fleet size (constant for
# the whole instance) as the "how loaded is the fleet" denominator.
CONTEXT_FEATURES_STATIC = ["num_active_requests", "requests_per_vehicle"]

# 2026-07-13: refined version using a proxy for CURRENTLY IDLE vehicles
# instead of total fleet size, derived for free from data we already have -
# no new simulation needed. assignment_data.jsonl's "assigned_requests" dict
# maps request_id -> vehicle_id for this window; the number of DISTINCT
# vehicle_ids appearing there is how many vehicles got a new pickup this
# round, so (total fleet - that count) is a rough "not currently busy with a
# new assignment" proxy. NOT the same as true remaining am/wc capacity per
# vehicle (that lives on the Vehicle objects inside VehicleHandler and isn't
# persisted anywhere per-iteration today - see design discussion) but free
# to compute from the existing logs, so worth checking before investing in
# instrumenting VehicleHandler and re-simulating everything (Option B from
# that discussion).
CONTEXT_FEATURES_IDLE_PROXY = ["num_idle_vehicles_proxy", "requests_per_idle_vehicle"]

CONTEXT_FEATURES = CONTEXT_FEATURES_STATIC + CONTEXT_FEATURES_IDLE_PROXY
BASE_FEATURES = RequestGraphFeatureBuilder.NODE_FEATURES
ALL_FEATURES = BASE_FEATURES + URGENCY_FEATURES + CONTEXT_FEATURES


def _load_instance_data(instance_name: str, config: Config) -> tuple[dict[int, object], int]:
    """
    Returns (request_lookup, num_vehicles) for one instance.

    request_lookup: full Request objects keyed by request id, built once and
    reused across all of that instance's rolling-horizon windows. Feature
    values (lat/lon, pickup window, ...) are static per request; only WHICH
    requests are active differs per window, and that's what the
    assignment_data.jsonl log tells us.

    num_vehicles: static fleet size for this instance (len(payload["driver_runs"])),
    used below to build the "requests_per_vehicle" context feature. This is
    NOT "currently idle vehicles" - getting that would require replaying the
    full vehicle-busy state through the whole rolling-horizon run, which
    isn't in the jsonl log. Total fleet size is a cheap, static proxy for
    fleet pressure instead.
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


def build_window_samples(instance_name: str, config: Config) -> list[pd.DataFrame]:
    """
    One row per (rolling-horizon window, active request) for one li_lim
    instance, labelled from the real baseline simulation log:
        assigned this window -> label 1
        unserved this window -> label 0
    """
    log_path = RH_BASELINE_DIR / instance_name / "rh_no_learning" / "assignment_data.jsonl"
    if not log_path.exists():
        print(f"  [skip] no rolling-horizon log for {instance_name}: {log_path}")
        return []

    request_lookup, num_vehicles = _load_instance_data(instance_name, config)
    split = _SPLIT_LOOKUP.get(instance_name, "unused")

    window_frames: list[pd.DataFrame] = []
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
            G = RequestGraphFeatureBuilder.add_node_features(G)
            node_matrix, _edge_index, _edge_matrix, node_ids = RequestGraphFeatureBuilder.to_numpy(G)

            # Sanity check: every node we built a feature row for must be
            # either assigned or unserved this window, and vice versa - if
            # this ever fails it means the request_lookup (built from the
            # static manifest file) and the jsonl log (from a possibly
            # different run/config) disagree about which ids exist.
            assert set(node_ids) == active_ids, (
                f"{instance_name} window {window_index}: node ids don't match "
                f"active ids from the log (lookup missing some ids or vice versa)"
            )

            node_df = pd.DataFrame(node_matrix, columns=RequestGraphFeatureBuilder.NODE_FEATURES)

            # Urgency features vary PER REQUEST within this window (each
            # request has its own pickup window), so they get computed here
            # and normalized together with the base features below - same
            # treatment, no special case needed.
            node_df["time_until_latest_pickup"] = [
                request_lookup[node_id].latest_pickup_time - timestamp for node_id in node_ids
            ]
            node_df["time_since_earliest_pickup"] = [
                timestamp - request_lookup[node_id].earliest_pickup_time for node_id in node_ids
            ]

            # Per-window z-score normalization, matching how RequestGraphPruner
            # normalizes at inference time today. IMPORTANT: ddof=0 (population
            # std), not pandas' default ddof=1 (sample std). Some rolling-horizon
            # windows here have exactly 1 active request (e.g. lr111 window 2) -
            # with ddof=1 that means std() divides by (n-1)=0 and returns NaN for
            # every column, which silently poisons the whole linear-probe
            # training a few lines down (all weights end up NaN after one full-
            # batch gradient step). ddof=0 gives 0 instead of NaN for n=1, which
            # the existing "+ 1e-8" guard then safely turns into a normalized
            # value of 0. Found by inspecting probe weights after training kept
            # coming back all-NaN (2026-07-13).
            node_df_norm = (node_df - node_df.mean()) / (node_df.std(ddof=0) + 1e-8)

            # Context features are constant ACROSS ALL REQUESTS of this one
            # window (there's only one "how many requests active right now"
            # value per window). Deliberately left RAW here instead of
            # per-window z-scored: subtracting the window's own mean from a
            # within-window-constant column always yields exactly 0 for every
            # row (the value equals its own window's mean by definition),
            # which would silently destroy exactly the cross-window signal
            # we're trying to test. These get standardized once, globally,
            # using train-set-only statistics in main() instead - "production
            # style" fixed normalization, not per-window/per-instance.
            node_df_norm["num_active_requests"] = float(len(active_ids))
            node_df_norm["requests_per_vehicle"] = len(active_ids) / num_vehicles

            # Idle-vehicle proxy: distinct vehicle_ids that received an
            # assignment THIS window, from the same "assigned_requests" dict
            # we already use for labels - no extra data needed. Capped at 0
            # so a (shouldn't-happen-but-be-safe) count exceeding the fleet
            # size doesn't go negative.
            vehicles_used_this_window = {int(vid) for vid in status["assigned_requests"].values()}
            num_idle_vehicles_proxy = max(num_vehicles - len(vehicles_used_this_window), 0)
            node_df_norm["num_idle_vehicles_proxy"] = float(num_idle_vehicles_proxy)
            # +1 guard: a window where every vehicle already got an
            # assignment (idle proxy = 0) shouldn't produce a division by
            # zero - it should instead read as "maximally loaded", i.e. a
            # large ratio, which len(active_ids) / 1 approximates reasonably.
            node_df_norm["requests_per_idle_vehicle"] = len(active_ids) / (num_idle_vehicles_proxy + 1)

            labels = np.array(
                [1.0 if node_id in assigned_ids else 0.0 for node_id in node_ids],
                dtype=np.float32,
            )
            node_df_norm["label"] = labels
            node_df_norm["instance"] = instance_name
            node_df_norm["window_index"] = window_index
            node_df_norm["timestamp"] = timestamp
            node_df_norm["split"] = split
            window_frames.append(node_df_norm)

    return window_frames


def load_all_samples(config: Config) -> pd.DataFrame:
    # Now iterates all 56 instances (train+val+test), not just the 12 that
    # used to have logs - build_window_samples() already skips gracefully
    # (with a printed [skip] line) if a log is somehow still missing, so
    # this stays safe even if Phase 0's simulation run didn't fully finish.
    all_instances = (
        REQUEST_PRUNER_SPLIT["train"] + REQUEST_PRUNER_SPLIT["val"] + REQUEST_PRUNER_SPLIT["test"]
    )
    frames: list[pd.DataFrame] = []
    for name in all_instances:
        print(f"  building window samples: {name} ({_SPLIT_LOOKUP[name]})")
        frames.extend(build_window_samples(name, config))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics - implemented by hand since scikit-learn is not installed in this
# project's venv (checked 2026-07-13). Small, well-known formulas, no need
# to add a new dependency for a throwaway script.
# ---------------------------------------------------------------------------


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U statistic). Handles ties via average ranks."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Standard average-precision formula: precision at each positive example,
    in score-descending order, averaged over all positives. Equivalent to
    sklearn's average_precision_score for the non-interpolated case.
    """
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp_cumsum = np.cumsum(labels_sorted)
    precision_at_k = tp_cumsum / (np.arange(len(labels_sorted)) + 1)
    return float((precision_at_k * labels_sorted).sum() / n_pos)


def recall_and_retention_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    """Recall = fraction of served (label=1) requests kept; retention = fraction of ALL requests kept."""
    kept = scores >= threshold
    n_pos = int(labels.sum())
    recall = float((kept & (labels == 1)).sum() / n_pos) if n_pos > 0 else float("nan")
    retention = float(kept.mean())
    return recall, retention


# ---------------------------------------------------------------------------
# Linear probe (logistic regression) - deliberately the simplest possible
# model. If this cannot beat the positive-rate baseline on held-out
# instances, a bigger MLP is unlikely to find signal that isn't there in
# these 8 features; the fix would be better (context) features, not a
# bigger model.
# ---------------------------------------------------------------------------


class LogisticProbe(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)  # raw logits; BCEWithLogitsLoss applies sigmoid internally


def train_linear_probe(train_df: pd.DataFrame, feature_cols: list[str], epochs: int = 300, lr: float = 0.05) -> LogisticProbe:
    x = torch.tensor(train_df[feature_cols].to_numpy(), dtype=torch.float32)
    y = torch.tensor(train_df["label"].to_numpy(), dtype=torch.float32)

    model = LogisticProbe(num_features=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Class imbalance varies per window; weight positives by the pooled
    # train-set imbalance ratio so the probe doesn't just learn "predict
    # unserved always" if unserved happens to dominate.
    n_pos = y.sum().clamp(min=1.0)
    n_neg = (len(y) - y.sum()).clamp(min=1.0)
    pos_weight = n_neg / n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    return model


def evaluate_probe(model: LogisticProbe, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32)
        scores = torch.sigmoid(model(x)).numpy()
    labels = df["label"].to_numpy()

    result = {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": average_precision(labels, scores),
        "positive_rate": float(labels.mean()),
    }
    for threshold in (0.1, 0.2, 0.3, 0.5):
        recall, retention = recall_and_retention_at_threshold(labels, scores, threshold)
        result[f"recall@{threshold}"] = recall
        result[f"retention@{threshold}"] = retention
    return result


def main() -> None:
    config = Config()

    print(f"Loading per-window samples from {RH_BASELINE_DIR} (56 instances, real train/val/test split)")
    all_df = load_all_samples(config)

    print(f"\nTotal (window, request) rows: {len(all_df)}")
    print(f"Windows: {all_df.groupby('instance')['window_index'].nunique().sum()}")
    print(f"Instances: {all_df['instance'].nunique()}")
    # Quick per-split row counts, mainly to sanity-check that "test" here
    # still has roughly the same number of rows as the old provisional-split
    # run (~1171 rows across the same 12 instances) - if not, something
    # changed about how those 12 instances' logs are being read.
    print(all_df.groupby("split")["instance"].nunique().rename("num_instances").to_string())

    # Standardize the two context columns now, using TRAIN rows only, and
    # apply that fixed transform to every row (train AND val/test). This is
    # the train-set-fixed normalization deferred from build_window_samples
    # (see the comment there for why per-window normalization doesn't work
    # for these two columns). Using only "train" (not train+val) keeps this
    # consistent with how a real deployed pruner would be built: val/test
    # must never influence the normalization statistics either, or they stop
    # being a fair held-out read.
    train_mask = all_df["split"] == "train"
    print("\n=== Context feature normalization (fit on train split only) ===")
    for col in CONTEXT_FEATURES:
        mean = all_df.loc[train_mask, col].mean()
        std = all_df.loc[train_mask, col].std(ddof=0)
        print(f"  {col:<24s} train mean={mean:.2f} train std={std:.2f}")
        all_df[col] = (all_df[col] - mean) / (std + 1e-8)

    # --- 1. Positive rate per instance (pooled across its windows) -----
    print("\n=== Positive rate per instance (pooled across windows) ===")
    print(all_df.groupby("instance")["label"].mean().to_string())

    # --- 2. Positive rate per window, to see within-instance variance ---
    print("\n=== Positive rate per window (first instance only, as a spot check) ===")
    first_instance = all_df["instance"].iloc[0]
    per_window = all_df[all_df["instance"] == first_instance].groupby("window_index")["label"].agg(["mean", "count"])
    print(per_window.to_string())

    # --- 3. Per-feature mean by label (now including the 4 new features) --
    print("\n=== Feature means by label (0 = unserved this window, 1 = assigned this window) ===")
    print(all_df.groupby("label")[ALL_FEATURES].mean().to_string())

    # --- 4. Per-feature univariate AUC (now including the 4 new features) -
    print("\n=== Univariate ROC-AUC per feature (no model, raw feature vs. label) ===")
    labels_np = all_df["label"].to_numpy()
    for col in ALL_FEATURES:
        auc = roc_auc(labels_np, all_df[col].to_numpy())
        marker = "  <- new" if col in URGENCY_FEATURES + CONTEXT_FEATURES else ""
        print(f"  {col:<28s} AUC={auc:.3f}{marker}")

    # --- 5. Linear probes: BASELINE (original 8) vs FULL variants ----------
    # Now uses the REAL train/val/test split (38 train / 6 val / 12 test
    # instances, see REQUEST_PRUNER_SPLIT above) instead of the old 6/6
    # probe_train/probe_eval split. "val" is reported for reference (it's
    # what you'd use to pick a variant/threshold in a real training run);
    # "test" is the actual held-out read the earlier 0.697/0.834 AUC numbers
    # should be compared against.
    train_df = all_df[all_df["split"] == "train"]
    val_df = all_df[all_df["split"] == "val"]
    test_df = all_df[all_df["split"] == "test"]

    print(
        f"\nTraining linear probes on {len(train_df)} rows "
        f"({train_df['instance'].nunique()} instances)..."
    )

    # Three variants, same split, same training routine - isolates exactly
    # what each feature group adds:
    #   BASELINE     - the original 8 pair-pruner features only
    #   STATIC       - + urgency + context using the STATIC fleet-size proxy
    #                  (requests_per_vehicle) - this is last run's "FULL"
    #   IDLE_PROXY   - + urgency + context using the IDLE-vehicle proxy
    #                  instead (num_idle_vehicles_proxy, requests_per_idle_vehicle)
    static_features = BASE_FEATURES + URGENCY_FEATURES + CONTEXT_FEATURES_STATIC
    idle_features = BASE_FEATURES + URGENCY_FEATURES + CONTEXT_FEATURES_IDLE_PROXY

    test_results = {}
    for label, feature_cols in (
        ("BASELINE (8 features)", BASE_FEATURES),
        ("STATIC (8 + urgency + static-fleet context = 12)", static_features),
        ("IDLE_PROXY (8 + urgency + idle-vehicle context = 12)", idle_features),
        ("ALL (8 + urgency + both context groups = 14)", ALL_FEATURES),
    ):
        probe = train_linear_probe(train_df, feature_cols)
        print(f"\n=== Linear probe: {label} ===")
        for name, df in (
            ("train (sanity check, not held-out)", train_df),
            ("val", val_df),
            ("test (held-out instances)", test_df),
        ):
            metrics = evaluate_probe(probe, df, feature_cols)
            print(f"\n  -- {name} --")
            for key, value in metrics.items():
                print(f"    {key:<14s} {value:.3f}")
        test_results[label] = evaluate_probe(probe, test_df, feature_cols)

    print("\n=== test-split summary (held-out instances only) ===")
    print(f"  {'variant':<55s} {'roc_auc':>8s} {'pr_auc':>8s}")
    for label, metrics in test_results.items():
        print(f"  {label:<55s} {metrics['roc_auc']:>8.3f} {metrics['pr_auc']:>8.3f}")

    print(
        "\nHow to read this: compare these test-split numbers against the "
        "earlier provisional-split run (BASELINE 0.697 / STATIC 0.834 / "
        "IDLE_PROXY 0.767 AUC on the old 6-instance probe_eval). If they "
        "land in a similar range, the earlier signal read holds up on ~5x "
        "more data and wasn't a small-sample artifact - safe to move on to "
        "building the real MLP pipeline. If STATIC drops a lot here, the "
        "earlier result was likely too optimistic (the 6 old probe_eval "
        "instances happened to be easy/similar to the 6 probe_train ones) "
        "and the features may need more work first."
    )


if __name__ == "__main__":
    main()
