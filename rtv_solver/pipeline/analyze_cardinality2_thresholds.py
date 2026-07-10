from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs" / "threshold_sweep_long.csv"
BASE_OUT = ROOT / "outputs" / "analysis_cardinality2"

metrics = [
    "service_rate",
    "total_time",
    "rtv_combos_iter0",
    "trip_costs_iter0",
]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]


def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)


def flatten_columns(df):
    df.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple)
        else str(col)
        for col in df.columns
    ]
    return df


def add_deltas(df):
    base = df[df["threshold"] == 0.0][
        ["pipeline", "instance"] + metrics
    ].rename(columns={m: f"{m}_baseline" for m in metrics})

    out = df.merge(base, on=["pipeline", "instance"], how="left")

    for m in metrics:
        out[f"{m}_delta"] = out[m] - out[f"{m}_baseline"]
        out[f"{m}_pct_change"] = (
            (out[m] / out[f"{m}_baseline"] - 1) * 100
        )

    return out


def analyze_pipeline(df_all, pipeline):
    df = df_all[df_all["pipeline"] == pipeline].copy()
    out_dir = BASE_OUT / pipeline
    out_dir.mkdir(parents=True, exist_ok=True)

    # Long with baseline deltas
    df_delta = add_deltas(df)
    df_delta.to_csv(out_dir / "threshold_sweep_long_with_deltas.csv", index=False)

    # Solver table: one row per instance
    solver = df.pivot_table(
        index=["pipeline", "class", "instance"],
        columns="threshold",
        values=metrics,
        aggfunc="mean",
    )

    solver.columns = [
        f"{metric}_t{str(th).replace('.', '')}"
        for metric, th in solver.columns
    ]
    solver = solver.reset_index()

    for metric in metrics:
        base_col = f"{metric}_t00"
        for th in thresholds:
            suffix = str(th).replace(".", "")
            th_col = f"{metric}_t{suffix}"

            if base_col in solver.columns and th_col in solver.columns:
                solver[f"{metric}_delta_t{suffix}"] = solver[th_col] - solver[base_col]
                solver[f"{metric}_pct_t{suffix}"] = (
                    (solver[th_col] / solver[base_col] - 1) * 100
                )

    solver.to_csv(out_dir / "table_solver_results.csv", index=False)

    # Class summary
    class_summary = (
        df.groupby(["pipeline", "class", "threshold"])[metrics]
        .agg(["mean", "median", "std", iqr])
        .reset_index()
    )
    class_summary = flatten_columns(class_summary)
    class_summary.to_csv(out_dir / "table_class_summary.csv", index=False)

    # Overall summary
    overall = (
        df.groupby(["pipeline", "threshold"])[metrics]
        .agg(["mean", "median", "std", iqr])
        .reset_index()
    )
    overall = flatten_columns(overall)
    overall.to_csv(out_dir / "table_overall_statistics.csv", index=False)

    # Correlations solver-side
    corr_cols = [
        "threshold",
        "service_rate",
        "total_time",
        "rtv_combos_iter0",
        "trip_costs_iter0",
        "service_rate_delta",
        "total_time_pct_change",
        "rtv_combos_iter0_pct_change",
        "trip_costs_iter0_pct_change",
    ]

    corr_df = df_delta[corr_cols].dropna()
    corr_df.corr(method="pearson").to_csv(out_dir / "correlation_solver_pearson.csv")
    corr_df.corr(method="spearman").to_csv(out_dir / "correlation_solver_spearman.csv")

    print(f"\n=== {pipeline.upper()} ===")
    print("Saved to:", out_dir)
    print()
    print("Rows by threshold:")
    print(df.groupby("threshold").size())
    print()
    print("Mean summary:")
    print(df.groupby("threshold")[metrics].mean().round(4))


def main():
    df = pd.read_csv(INPUT)

    # Optional: remove exact duplicate log rows if same run was parsed twice
    df = df.drop_duplicates(
        subset=["pipeline", "instance", "threshold", "log_path"],
        keep="last",
    )

    BASE_OUT.mkdir(parents=True, exist_ok=True)

    for pipeline in ["coaml", "rh"]:
        analyze_pipeline(df, pipeline)

    # Combined comparison summary
    comparison = (
        df.groupby(["pipeline", "threshold"])[metrics]
        .agg(["mean", "median", "std", iqr])
        .reset_index()
    )
    comparison = flatten_columns(comparison)
    comparison.to_csv(BASE_OUT / "comparison_overall_statistics.csv", index=False)

    print("\n=== COMBINED COMPARISON SAVED ===")
    print(BASE_OUT / "comparison_overall_statistics.csv")


if __name__ == "__main__":
    main()