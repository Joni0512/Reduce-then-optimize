"""
2026-07-12: aggregates the PrunerImitationDiagnostics records written by
COAMLPipeline._log_pruner_imitation_diagnostics (see coaml_pipeline.py) across
one or more runs, so pruner-pair-recall and imitation-y*-fallback-rate can be
compared over a full run (or a whole threshold sweep) instead of eyeballing
single console lines.

Why this exists: a manual smoke test on lc108 (--use_request_graph_pruner True,
threshold 0.3) showed pair recall = 1.000 (pruning dropped nothing) together
with 9/11 vehicles in the y* fallback branch in the same iteration -- i.e. a
high fallback rate does NOT by itself mean pruning damaged the imitation
target (idle vehicles with no optimal-solution work also land in that
branch). This script exists to check whether that holds up over many
iterations/instances, by correlating recall against fallback_rate per run.

Usage:
    python -m rtv_solver.pipeline.parse_pruner_imitation_diagnostics \\
        --root outputs/debug/pruner_sensitivity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_run_records(jsonl_path: Path, run_id: str) -> list[dict]:
    """
    Extract all PrunerImitationDiagnostics entries from one assignment_data.jsonl.

    Reads the file line-by-line with json.loads(), the same pattern already
    used by StatsParser._compute_request_development (stats_parser.py) to read
    this file -- kept consistent rather than introducing a second jsonl-reading
    convention.
    """
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("message") != "PrunerImitationDiagnostics":
                continue
            extra = entry.get("extra", {})
            stats = extra.get("stats", {})
            records.append({
                "run_id": run_id,
                "timestamp": extra.get("timestamp"),
                "pruner_optimal_pairs_total": stats.get("pruner_optimal_pairs_total"),
                "pruner_optimal_pairs_preserved": stats.get("pruner_optimal_pairs_preserved"),
                "pruner_pair_recall": stats.get("pruner_pair_recall"),
                "pruner_fallback_vehicles": stats.get("pruner_fallback_vehicles"),
                "pruner_vehicles_considered": stats.get("pruner_vehicles_considered"),
            })
    return records


def collect_long_dataframe(root: Path) -> pd.DataFrame:
    """
    Recursively find every assignment_data.jsonl under root and stack their
    PrunerImitationDiagnostics records into one long-format dataframe (one row
    per solver iteration). run_id is the run directory's path relative to
    root, so results from different sweep runs (e.g. different pruner
    thresholds or instances) stay distinguishable without a hand-maintained
    list of run folders (unlike parse_threshold_sweep_coaml.py's RUN_SOURCES,
    which was avoided here since every run already writes a discoverable
    assignment_data.jsonl).
    """
    rows: list[dict] = []
    for jsonl_path in sorted(root.rglob("assignment_data.jsonl")):
        run_id = str(jsonl_path.parent.relative_to(root))
        rows.extend(_load_run_records(jsonl_path, run_id))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Derived (not logged directly) so both metrics stay in [0, 1] for direct
    # scatter/correlation comparison against pruner_pair_recall.
    df["pruner_fallback_rate"] = df["pruner_fallback_vehicles"] / df["pruner_vehicles_considered"]
    return df


def summarize_by_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-run aggregate: mean/min recall, mean fallback rate, and their
    correlation within the run.

    A correlation near zero (or positive) means fallback is not explained by
    pruning recall in that run -- most fallback vehicles are probably idle
    ones with no optimal-solution work this batch, not casualties of pruning.
    A clearly negative correlation (low recall coinciding with high fallback)
    would support the "pruning is degrading the imitation target" hypothesis
    and justify one of the mitigation strategies discussed for the pruner
    (train-time pair whitelisting, pruner-off curriculum, etc.).
    """
    grouped = df.groupby("run_id")
    summary = grouped.agg(
        iterations=("pruner_fallback_rate", "size"),
        mean_pair_recall=("pruner_pair_recall", "mean"),
        min_pair_recall=("pruner_pair_recall", "min"),
        mean_fallback_rate=("pruner_fallback_rate", "mean"),
        mean_fallback_vehicles=("pruner_fallback_vehicles", "mean"),
        mean_vehicles_considered=("pruner_vehicles_considered", "mean"),
    ).reset_index()

    correlations = grouped.apply(
        lambda g: g["pruner_pair_recall"].corr(g["pruner_fallback_rate"])
        if g["pruner_pair_recall"].notna().sum() > 1
        else float("nan")
    ).rename("recall_fallback_corr").reset_index()

    return summary.merge(correlations, on="run_id")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate PrunerImitationDiagnostics records across one or more COAML runs."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Directory to search recursively for assignment_data.jsonl files "
             "(e.g. outputs/debug/pruner_sensitivity, or a single run's output_dir).",
    )
    parser.add_argument(
        "--out_long",
        type=str,
        default=None,
        help="Output path for the per-iteration long CSV. Defaults to <root>/pruner_imitation_diagnostics_long.csv.",
    )
    parser.add_argument(
        "--out_summary",
        type=str,
        default=None,
        help="Output path for the per-run summary CSV. Defaults to <root>/pruner_imitation_diagnostics_summary.csv.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_long = Path(args.out_long) if args.out_long else root / "pruner_imitation_diagnostics_long.csv"
    out_summary = Path(args.out_summary) if args.out_summary else root / "pruner_imitation_diagnostics_summary.csv"

    df = collect_long_dataframe(root)
    if df.empty:
        print(f"No PrunerImitationDiagnostics records found under {root}.")
        return

    df.to_csv(out_long, index=False)
    print(f"Saved long CSV ({len(df)} rows): {out_long}")

    summary = summarize_by_run(df)
    summary.to_csv(out_summary, index=False)
    print(f"Saved summary CSV ({len(summary)} runs): {out_summary}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
