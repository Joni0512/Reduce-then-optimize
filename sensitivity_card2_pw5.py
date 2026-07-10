from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "new_tests" / "card2_pw5" / "analysis"

RH_FILE = BASE / "rh_pruning_summary_long.csv"
COAML_FILE = BASE / "coaml_pruning_summary_long.csv"

OUT = BASE / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = [
    "service_rate",
    "total_time",
    "service_rate_delta",
    "service_rate_pct_change",
    "total_time_delta",
    "total_time_pct_change",
]

def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

def add_quality_labels(df):
    df = df.copy()
    df["runtime_reduction_pct"] = -df["total_time_pct_change"]
    df["service_rate_change_pp"] = df["service_rate_delta"] * 100
    return df

def summarize(df, group_cols):
    return (
        df.groupby(group_cols)[
            ["service_rate", "total_time", "runtime_reduction_pct", "service_rate_change_pp"]
        ]
        .agg(["mean", "median", "std", iqr])
        .reset_index()
    )

def plot_threshold(df, pipeline):
    overall = (
        df.groupby("threshold")[["service_rate", "total_time", "runtime_reduction_pct"]]
        .mean()
        .reset_index()
    )

    # Runtime
    plt.figure()
    plt.plot(overall["threshold"], overall["total_time"], marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Mean runtime (s)")
    plt.title(f"{pipeline.upper()} mean runtime by threshold")
    plt.grid(True)
    plt.savefig(OUT / f"{pipeline}_runtime_by_threshold.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Service rate
    plt.figure()
    plt.plot(overall["threshold"], overall["service_rate"], marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Mean service rate")
    plt.title(f"{pipeline.upper()} mean service rate by threshold")
    plt.grid(True)
    plt.savefig(OUT / f"{pipeline}_service_rate_by_threshold.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Runtime reduction
    pruned = overall[overall["threshold"] > 0]
    plt.figure()
    plt.plot(pruned["threshold"], pruned["runtime_reduction_pct"], marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Mean runtime reduction (%)")
    plt.title(f"{pipeline.upper()} runtime reduction by threshold")
    plt.grid(True)
    plt.savefig(OUT / f"{pipeline}_runtime_reduction_by_threshold.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_boxplots(df, pipeline):
    pruned = df[df["threshold"] > 0].copy()

    plt.figure()
    pruned.boxplot(column="runtime_reduction_pct", by="threshold")
    plt.title(f"{pipeline.upper()} runtime reduction distribution")
    plt.suptitle("")
    plt.xlabel("Threshold")
    plt.ylabel("Runtime reduction (%)")
    plt.savefig(OUT / f"{pipeline}_runtime_reduction_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    pruned.boxplot(column="service_rate_change_pp", by="threshold")
    plt.title(f"{pipeline.upper()} service rate change distribution")
    plt.suptitle("")
    plt.xlabel("Threshold")
    plt.ylabel("Service rate change (pp)")
    plt.savefig(OUT / f"{pipeline}_service_rate_change_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close()

def analyze_solver(path, pipeline):
    df = pd.read_csv(path)
    df = add_quality_labels(df)

    df.to_csv(OUT / f"{pipeline}_sensitivity_long.csv", index=False)

    # Overall threshold stats
    overall = summarize(df, ["pipeline", "threshold"])
    overall.to_csv(OUT / f"{pipeline}_threshold_summary_stats.csv", index=False)

    # By class and threshold
    by_class = summarize(df, ["pipeline", "class", "threshold"])
    by_class.to_csv(OUT / f"{pipeline}_class_threshold_summary_stats.csv", index=False)

    # By instance and threshold
    by_instance = df[
        [
            "pipeline", "class", "instance", "threshold",
            "service_rate", "service_rate_change_pp",
            "total_time", "runtime_reduction_pct",
            "serviced", "total_requests"
        ]
    ].sort_values(["class", "instance", "threshold"])
    by_instance.to_csv(OUT / f"{pipeline}_instance_threshold_table.csv", index=False)

    plot_threshold(df, pipeline)
    plot_boxplots(df, pipeline)

    print(f"\n=== {pipeline.upper()} ===")
    print(
        df.groupby("threshold")[
            ["service_rate", "total_time", "runtime_reduction_pct", "service_rate_change_pp"]
        ].mean().round(4)
    )

    return df

def compare_pipelines(rh, coaml):
    combined = pd.concat([rh, coaml], ignore_index=True)

    comparison = summarize(combined, ["pipeline", "threshold"])
    comparison.to_csv(OUT / "pipeline_threshold_comparison_stats.csv", index=False)

    plt.figure()
    for pipeline, sub in combined.groupby("pipeline"):
        avg = sub.groupby("threshold")["runtime_reduction_pct"].mean().reset_index()
        avg = avg[avg["threshold"] > 0]
        plt.plot(avg["threshold"], avg["runtime_reduction_pct"], marker="o", label=pipeline)
    plt.xlabel("Threshold")
    plt.ylabel("Mean runtime reduction (%)")
    plt.title("Runtime reduction: RHO vs SIL/COAML")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT / "pipeline_runtime_reduction_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    for pipeline, sub in combined.groupby("pipeline"):
        avg = sub.groupby("threshold")["service_rate_change_pp"].mean().reset_index()
        plt.plot(avg["threshold"], avg["service_rate_change_pp"], marker="o", label=pipeline)
    plt.xlabel("Threshold")
    plt.ylabel("Mean service rate change (pp)")
    plt.title("Service rate change: RHO vs SIL/COAML")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT / "pipeline_service_rate_change_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

def main():
    rh = analyze_solver(RH_FILE, "rho")
    coaml = analyze_solver(COAML_FILE, "sil")
    compare_pipelines(rh, coaml)

    print("\nSaved all tables and plots to:")
    print(OUT)

if __name__ == "__main__":
    main()
