"""
Runs a "stress" NYC scenario through RollingHorizon's solver with real vehicle
scarcity (few vehicles relative to requests, forcing genuine ride-pooling
pressure), sweeping the trained NYC GNN pruner across thresholds, and reports
solver-level + classifier-level metrics side by side in one table + plots.

Why a stress scenario: our first end-to-end test (1000 vehicles for ~470
requests) showed pruning barely mattered - vehicle supply was so abundant that
the solver rarely needed to pool requests at all, so cutting "unlikely to be
pooled" edges changed almost nothing. Here VEHICLE_LIMIT is deliberately small
so vehicles are genuinely scarce and pooling (and therefore pruning of the
pooling candidate graph) actually has work to do.

Metrics measured, per threshold:

Solver-level (from actually running RollingHorizon's ./prog on the stress
scenario):
    - runtime_s: WALL-CLOCK seconds for the ./prog subprocess to finish the
      whole simulated 3h window. This is real compute time (ILP + trip
      generation + simulation), not simulated trip/ride time - it answers
      "does pruning make each iteration's ILP faster to solve in practice".
    - service_rate: RollingHorizon's own "Service rate is X" log line
      (percentage of requests actually picked up by the end of the run).
    - edge_reduction_operational: mean(1 - after/before) across every RTV
      iteration's LOCAL R-R candidate graph (from "GNN pruning R-R edges:
      X -> Y" log lines). This is the live, per-iteration candidate-graph
      shrinkage the solver actually experienced - usually much smaller in
      absolute edge count than edge_reduction_global below, since the local
      candidate graph per iteration only covers currently-pending requests.

Classifier-level (scoring the trained checkpoint once against expert labels -
no solver run needed, same request graph regardless of vehicle count):
    - recall: of the pairs that were genuinely pooled together in a real solve
      (strict temporally-overlapping expert_pairs_overlap.csv), how many does
      the pruner keep at this threshold? Losing one directly risks losing a
      pooling opportunity the solver could have used.
    - f2, f3: precision/recall F-beta scores (beta=2, 3) - weight recall over
      precision, matching the training objective's pos_weight=5 preference for
      not dropping true positive (should-be-pooled) edges over keeping a few
      extra false positives.
    - edge_reduction_global: fraction of the COMPLETE 473-request pair graph
      (111,628 unordered request-pair edges - RequestGraphFullBuilder builds an
      undirected graph, one edge per pair, no direction duplication) pruned
      away at this threshold -
      the theoretical maximum candidate-space reduction, independent of which
      requests happen to be locally active at any one simulated moment.

Usage:
    python3 -m rtv_solver.pipeline.nyc_stress_sweep \
        --manifest solutions/nyc/manifests/nyc_dense900_40min.json \
        --checkpoint outputs/models_nyc/rgnn_nyc_morning500_mc3_fixed_pw5_v2/rgnn_nyc_morning500_mc3_fixed_pw5_v2_best_val_f3.pt \
        --expert-pairs-csv results/nyc_dense900_40min/expert_pairs_overlap.csv \
        --request-data-file requests_dense900_40min.csv \
        --initial-time 070000 --final-time 074000 \
        --vehicle-limit 1000 \
        --out-subdir nyc_dense900_sweep
"""

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config

from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.export_gnn_pruned_edges import load_model

ROOT = Path(__file__).resolve().parents[2]
RH_ROOT = Path("/Users/joni/Desktop/Masterarbeit/external_repos/RollingHorizon")
MOSEK_LIB = "/Users/joni/software/mosek/8/tools/platform/osx64x86/bin"

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_prog(
    results_dir: str, use_gnn_pruning: bool, gnn_pruning_file: str | None,
    request_data_file: str, initial_time: str, final_time: str, vehicle_limit: int,
    interval: int | None = None,
) -> tuple[float, str]:
    args = [
        "./prog", "4",
        "RESULTS_DIRECTORY", results_dir,
        "REQUEST_DATA_FILE", request_data_file,
        "CARSIZE", "3",
        "INITIAL_TIME", initial_time,
        "FINAL_TIME", final_time,
        "VEHICLE_LIMIT", str(vehicle_limit),
        "USE_GNN_PRUNING", "true" if use_gnn_pruning else "false",
    ]
    # INTERVAL = seconds between reoptimization epochs (RollingHorizon default: 60).
    # Left unset (None) reproduces the exact prior behaviour/results.
    if interval is not None:
        args += ["INTERVAL", str(interval)]
    if use_gnn_pruning:
        args += ["GNN_PRUNING_FILE", gnn_pruning_file]

    (RH_ROOT / results_dir).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "DYLD_LIBRARY_PATH": MOSEK_LIB}

    start = time.time()
    result = subprocess.run(args, cwd=RH_ROOT, env=env, capture_output=True, text=True)
    runtime_s = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"./prog failed (exit {result.returncode}):\n{result.stderr[-2000:]}")

    return runtime_s, result.stdout + result.stderr


def parse_service_rate(log: str) -> float:
    # "Service rate is 48.202960." - the trailing "." is a sentence-ending period, not
    # part of the number, so match digits.digits explicitly rather than a greedy [\d.]+
    # (which previously captured the period too and crashed float()).
    matches = re.findall(r"Service rate is (\d+\.\d+)", log)
    return float(matches[-1]) if matches else float("nan")


def parse_operational_edge_reduction(log: str) -> float:
    matches = re.findall(r"GNN pruning R-R edges: (\d+) -> (\d+)", log)
    reductions = [1 - int(after) / int(before) for before, after in matches if int(before) > 0]
    return sum(reductions) / len(reductions) if reductions else float("nan")


def fbeta(precision: float, recall: float, beta: float) -> float:
    denom = (beta ** 2) * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + beta ** 2) * precision * recall / denom


def build_request_graph(manifest: Path):
    config = Config()
    payload = PayloadParser.load_input_data(manifest)
    NetworkHandler.init_from_payload(payload=payload, server_url=config.SERVER_URL)
    requests = RequestHandler(payload[PayloadKeys.REQUESTS], config).get_all_requests()

    G = RequestGraphFullBuilder.build(requests)
    G = RequestGraphFeatureBuilder.add_features(G)
    node_matrix, edge_index, edge_matrix, node_ids = RequestGraphFeatureBuilder.to_numpy(G)

    node_df = pd.DataFrame(node_matrix, columns=RequestGraphFeatureBuilder.NODE_FEATURES)
    edge_df = pd.DataFrame(edge_matrix, columns=RequestGraphFeatureBuilder.EDGE_FEATURES)
    node_matrix_norm = ((node_df - node_df.mean()) / (node_df.std() + 1e-8)).to_numpy()
    edge_matrix_norm = ((edge_df - edge_df.mean()) / (edge_df.std() + 1e-8)).to_numpy()

    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edge_index_reindexed = [[id_to_idx[u], id_to_idx[v]] for u, v in edge_index]

    x = torch.tensor(node_matrix_norm, dtype=torch.float32)
    edge_attr = torch.tensor(edge_matrix_norm, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index_reindexed, dtype=torch.long).t().contiguous()
    return x, edge_attr, edge_index_t, edge_index


def score_edges(x, edge_attr, edge_index_t, checkpoint: Path):
    model, version = load_model(checkpoint, x.shape[1], edge_attr.shape[1])
    print(f"Loaded {version} checkpoint: {checkpoint}")
    with torch.no_grad():
        return torch.sigmoid(model(x=x, edge_index=edge_index_t, edge_attr=edge_attr)).numpy()


def export_pruned_csv(edge_index, scores, threshold: float, out_path: Path) -> set[tuple[int, int]]:
    kept_pairs = set()
    for (u, v), score in zip(edge_index, scores):
        if score >= threshold:
            kept_pairs.add(tuple(sorted((int(u), int(v)))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("request_i,request_j\n")
        for a, b in sorted(kept_pairs):
            f.write(f"{a},{b}\n")
    return kept_pairs


def classifier_metrics(kept_pairs: set[tuple[int, int]], expert_pairs: set[tuple[int, int]], total_pairs: int) -> dict:
    tp = len(kept_pairs & expert_pairs)
    fn = len(expert_pairs - kept_pairs)
    precision = tp / len(kept_pairs) if kept_pairs else 0.0
    recall = tp / len(expert_pairs) if expert_pairs else 0.0
    return {
        "recall": recall,
        "f2": fbeta(precision, recall, 2.0),
        "f3": fbeta(precision, recall, 3.0),
        "edge_reduction_global": 1 - len(kept_pairs) / total_pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="relative to Reduce-then-optimize repo root")
    parser.add_argument("--checkpoint", required=True, help="relative to Reduce-then-optimize repo root")
    parser.add_argument("--expert-pairs-csv", required=True, help="relative to RollingHorizon repo root")
    parser.add_argument("--request-data-file", required=True, help="filename under RollingHorizon/data/requests/")
    parser.add_argument("--initial-time", default="070000")
    parser.add_argument("--final-time", required=True)
    parser.add_argument("--vehicle-limit", type=int, default=1000)
    parser.add_argument("--interval", type=int, default=None,
                         help="Seconds between reoptimization epochs (RollingHorizon default: 60). "
                              "Left unset reproduces the exact prior behaviour.")
    parser.add_argument("--out-subdir", required=True, help="results subdir under RollingHorizon/results/")
    parser.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    args = parser.parse_args()

    manifest = ROOT / args.manifest
    checkpoint = ROOT / args.checkpoint
    expert_pairs_csv = RH_ROOT / args.expert_pairs_csv
    out_dir = RH_ROOT / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    expert_df = pd.read_csv(expert_pairs_csv)
    expert_pairs = set(
        tuple(sorted((int(a), int(b))))
        for a, b in zip(expert_df["request_id_a"], expert_df["request_id_b"])
    )

    x, edge_attr, edge_index_t, edge_index = build_request_graph(manifest)
    scores = score_edges(x, edge_attr, edge_index_t, checkpoint)
    total_pairs = len(set(tuple(sorted((int(u), int(v)))) for u, v in edge_index))
    print(f"Request graph: {total_pairs} unordered edges, "
          f"{len(expert_pairs)} expert (genuinely pooled) pairs")

    def _run(results_subdir, use_gnn_pruning, gnn_pruning_file):
        return run_prog(
            results_subdir, use_gnn_pruning, gnn_pruning_file,
            request_data_file=args.request_data_file,
            initial_time=args.initial_time,
            final_time=args.final_time,
            vehicle_limit=args.vehicle_limit,
            interval=args.interval,
        )

    rows = []

    print(f"\n=== Baseline (no pruning), VEHICLE_LIMIT={args.vehicle_limit} ===")
    runtime_s, log = _run(f"{args.out_subdir}/baseline", use_gnn_pruning=False, gnn_pruning_file=None)
    (out_dir / "baseline.log").write_text(log)
    rows.append({
        "threshold": None,
        "runtime_s": runtime_s,
        "service_rate": parse_service_rate(log),
        "edge_reduction_operational": 0.0,
        "recall": 1.0,
        "f2": None,
        "f3": None,
        "edge_reduction_global": 0.0,
    })
    print(f"  runtime={runtime_s:.1f}s  service_rate={rows[-1]['service_rate']}")

    for threshold in args.thresholds:
        print(f"\n=== Threshold {threshold} ===")
        csv_path = out_dir / f"pruned_keep_edges_{threshold:.2f}.csv"
        kept_pairs = export_pruned_csv(edge_index, scores, threshold, csv_path)
        cls_metrics = classifier_metrics(kept_pairs, expert_pairs, total_pairs)

        rel_csv_path = f"results/{args.out_subdir}/pruned_keep_edges_{threshold:.2f}.csv"
        runtime_s, log = _run(
            f"{args.out_subdir}/t{str(threshold).replace('.', '')}",
            use_gnn_pruning=True,
            gnn_pruning_file=rel_csv_path,
        )
        (out_dir / f"t{str(threshold).replace('.', '')}.log").write_text(log)

        row = {
            "threshold": threshold,
            "runtime_s": runtime_s,
            "service_rate": parse_service_rate(log),
            "edge_reduction_operational": parse_operational_edge_reduction(log),
            **cls_metrics,
        }
        rows.append(row)
        print(f"  runtime={row['runtime_s']:.1f}s  service_rate={row['service_rate']}  "
              f"edge_reduction(op/global)={row['edge_reduction_operational']:.3f}/{row['edge_reduction_global']:.3f}  "
              f"recall={row['recall']:.3f}  f2={row['f2']:.3f}  f3={row['f3']:.3f}")

    df = pd.DataFrame(rows)
    table_path = out_dir / f"{args.out_subdir}_results.csv"
    df.to_csv(table_path, index=False)
    print(f"\nSaved table: {table_path}")
    print(df.to_string(index=False))

    # --- plots ---
    plot_df = df[df["threshold"].notna()].sort_values("threshold")
    baseline_row = df[df["threshold"].isna()].iloc[0]

    # 2026-07-11: clarified titles - "operational" and "global" edge reduction are easy
    # to conflate. The GNN is scored once, statically, over the ENTIRE request graph
    # (that's "global" - includes pairs the solver would never even compare, e.g. two
    # requests 38 minutes apart). The solver ITSELF already narrows candidates down to
    # a small locally-active set per iteration via MAX_WAITING/MAX_DETOUR, independent
    # of the GNN; "operational" is how much smaller THAT already-prefiltered live set
    # gets once the GNN's precomputed keep-list is additionally applied on top of it.
    metrics_to_plot = [
        ("runtime_s", "Runtime (s, wall-clock)", None),
        ("service_rate", "Service rate (%)", None),
        ("edge_reduction_operational", "Edge reduction - OPERATIONAL",
         "GNN's extra cut on top of the solver's own\nlive time-window candidate list (small, per iteration)"),
        ("edge_reduction_global", "Edge reduction - GLOBAL",
         "GNN scored on the full static request graph\n(incl. pairs the solver would never compare anyway)"),
        ("recall", "Recall (vs. expert pooled pairs)", None),
        ("f2", "F2 score", None),
        ("f3", "F3 score", None),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(11, 15))
    axes = axes.flatten()
    for ax, (col, title, subtitle) in zip(axes, metrics_to_plot):
        ax.plot(plot_df["threshold"], plot_df[col], marker="o", color="tab:blue")
        if col in ("runtime_s", "service_rate"):
            ax.axhline(baseline_row[col], color="gray", linestyle="--", label="baseline (no pruning)")
            ax.legend(fontsize=8)
        ax.set_xlabel("Pruner threshold")
        ax.set_ylabel(title)
        if subtitle:
            ax.set_title(f"{title}\n{subtitle}", fontsize=9, style="italic")
        else:
            ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[-1].axis("off")
    interval_label = f"INTERVAL={args.interval}s" if args.interval is not None else "INTERVAL=60s (default)"
    fig.suptitle(f"NYC scenario ({args.out_subdir}, VEHICLE_LIMIT={args.vehicle_limit}, {interval_label}) - GNN pruning sweep")
    fig.tight_layout()
    plot_path = out_dir / f"{args.out_subdir}_plots.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plots: {plot_path}")


if __name__ == "__main__":
    main()
