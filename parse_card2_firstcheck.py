from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
# 2026-07-09: the actual runs used --output_dir "outputs/new_tests/card2_firstcheck/...",
# and main.py prepends its own "outputs/" on top of that (see Config.create_output_dir),
# so results really did land under outputs/outputs/... - pointing BASE there to match
# reality instead of re-running the (already valid, already has final/) solver jobs.
BASE = ROOT / "outputs" / "outputs" / "new_tests" / "card2_firstcheck"
OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

ALL_INSTANCES = [
    "lc108", "lc109", "lc207", "lc208",
    "lr111", "lr112", "lr210", "lr211",
    "lrc107", "lrc108", "lrc207", "lrc208",
]
# 2026-07-09: mc3 sweep started with LC instances only (lr/lrc mc3 not run yet) -
# scoping the mc3 rows to just these avoids spamming "no results.json" warnings
# for instances that haven't been run at cardinality 3.
LC_INSTANCES = ["lc108", "lc109", "lc207", "lc208"]

# 2026-07-09: added "rh" (offline mode) alongside "coaml" so both solver pipelines
# can be compared in one pass instead of needing a second, near-duplicate script.
# Each run is (pipeline, method, threshold, cardinality, folder_name, instances).
# threshold=None for baseline (no pruner, so no threshold applies). Existing
# gnnv1/gnnv2 folders without a "_tXX"/"_mcN" suffix predate the multi-threshold/
# cardinality sweep and are all threshold=0.5, cardinality=2 runs.
RUNS = [
    ("coaml", "baseline", None, 2, "coaml_baseline", ALL_INSTANCES),
    ("coaml", "gnn_v1", 0.5, 2, "coaml_gnnv1", ALL_INSTANCES),
    ("coaml", "gnn_v2", 0.5, 2, "coaml_gnnv2", ALL_INSTANCES),
    ("coaml", "gnn_v1", 0.4, 2, "coaml_gnnv1_t04", ALL_INSTANCES),
    ("coaml", "gnn_v1", 0.3, 2, "coaml_gnnv1_t03", ALL_INSTANCES),
    ("coaml", "gnn_v2", 0.3, 2, "coaml_gnnv2_t03", ALL_INSTANCES),
    ("coaml", "gnn_v1", 0.2, 2, "coaml_gnnv1_t02", ALL_INSTANCES),
    ("coaml", "gnn_v2", 0.2, 2, "coaml_gnnv2_t02", ALL_INSTANCES),
    ("coaml", "gnn_v1", 0.1, 2, "coaml_gnnv1_t01", ALL_INSTANCES),
    ("coaml", "gnn_v2", 0.1, 2, "coaml_gnnv2_t01", ALL_INSTANCES),
    ("rh", "baseline", None, 2, "rh_baseline", ALL_INSTANCES),
    ("rh", "gnn_v1", 0.5, 2, "rh_gnnv1", ALL_INSTANCES),
    ("rh", "gnn_v2", 0.5, 2, "rh_gnnv2", ALL_INSTANCES),
    ("rh", "gnn_v1", 0.4, 2, "rh_gnnv1_t04", ALL_INSTANCES),
    ("rh", "gnn_v2", 0.4, 2, "rh_gnnv2_t04", ALL_INSTANCES),
    ("rh", "gnn_v1", 0.3, 2, "rh_gnnv1_t03", ALL_INSTANCES),
    ("rh", "gnn_v2", 0.3, 2, "rh_gnnv2_t03", ALL_INSTANCES),
    ("rh", "gnn_v1", 0.2, 2, "rh_gnnv1_t02", ALL_INSTANCES),
    ("rh", "gnn_v2", 0.2, 2, "rh_gnnv2_t02", ALL_INSTANCES),
    ("rh", "gnn_v1", 0.1, 2, "rh_gnnv1_t01", ALL_INSTANCES),
    ("rh", "gnn_v2", 0.1, 2, "rh_gnnv2_t01", ALL_INSTANCES),
    ("rh", "baseline", None, 3, "rh_baseline_mc3", LC_INSTANCES),
    ("rh", "gnn_v1", 0.4, 3, "rh_gnnv1_t04_mc3", LC_INSTANCES),
    ("rh", "gnn_v2", 0.4, 3, "rh_gnnv2_t04_mc3", LC_INSTANCES),
]


def cls(x):
    if x.startswith("lrc"):
        return "LRC"
    if x.startswith("lc"):
        return "LC"
    if x.startswith("lr"):
        return "LR"
    return "unknown"


def latest(folder):
    # Only match final/results.json (the single-instance solve result written at
    # main.py:252), not any of the many per-epoch results.json files COAML writes
    # for its own train_val/val/optimal_val instances - those belong to other
    # instances entirely and a plain rglob("results.json") would silently pick one
    # of those up as "the" result if it happened to be the most recently modified.
    files = list(folder.rglob("final/results.json"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def read(path):
    data = json.load(open(path))
    stats = data["stats"]
    serviced = stats["serviced"]
    total = stats["total_requests"]
    return {
        "serviced": serviced,
        "total_requests": total,
        "service_rate": serviced / total,
        "total_time": stats["total_time"],
        "vmt": stats.get("vmt"),
        "average_wait_time": stats.get("average_wait_time"),
        "average_detour": stats.get("average_detour"),
    }


rows = []

for pipeline, method, threshold, cardinality, folder_name, instances in RUNS:
    for inst in instances:
        p = latest(BASE / folder_name / inst)
        if p is None:
            print(f"WARNING: no final/results.json found for {pipeline}/{method}/t={threshold}/mc={cardinality}/{inst} under {BASE / folder_name / inst}")
            continue
        rows.append(
            {
                "pipeline": pipeline,
                "class": cls(inst),
                "instance": inst,
                "method": method,
                "threshold": threshold,
                "cardinality": cardinality,
                "result_path": str(p),
                **read(p),
            }
        )

if not rows:
    raise SystemExit(
        "No final/results.json found for any pipeline/instance/method - check BASE path "
        f"({BASE}) matches where the solver runs actually wrote their output."
    )

df = pd.DataFrame(rows)

metrics = [
    "service_rate", "total_time", "serviced", "total_requests",
    "vmt", "average_wait_time", "average_detour",
]

# baseline = no pruner, per (pipeline, instance, cardinality) -> used to compute
# delta/pct_change for gnn_v1/gnn_v2 rows. Merging on cardinality too keeps mc3 runs
# from being compared against the mc2 baseline (different search space -> not comparable).
base = df[df.method == "baseline"][["pipeline", "instance", "cardinality"] + metrics].rename(
    columns={m: f"{m}_baseline" for m in metrics}
)
df = df.merge(base, on=["pipeline", "instance", "cardinality"], how="left")

for m in metrics:
    df[f"{m}_delta"] = df[m] - df[f"{m}_baseline"]
    df[f"{m}_pct_change"] = (df[m] / df[f"{m}_baseline"] - 1) * 100

df.to_csv(OUT / "card2_firstcheck_long.csv", index=False)

# One row per (pipeline, instance, cardinality), columns per method+threshold - easier
# to eyeball v1 vs v2 (and threshold) vs baseline side by side.
df["method_t"] = df.apply(
    lambda r: r["method"] if pd.isna(r["threshold"]) else f"{r['method']}_t{str(r['threshold']).replace('.', '')}",
    axis=1,
)
wide = df.pivot_table(
    index=["pipeline", "class", "instance", "cardinality"],
    columns="method_t",
    values=metrics,
    aggfunc="first",
)
wide.columns = [f"{m}_{method_t}" for m, method_t in wide.columns]
wide = wide.reset_index()
wide.to_csv(OUT / "card2_firstcheck_wide.csv", index=False)

# 2026-07-09: mean alone hides how much individual instances scatter (e.g. the
# lc207 outlier at rh/gnn_v2/t0.5) - median/std/IQR make that spread visible instead
# of only ever showing the single-number average.
def q1(s):
    return s.quantile(0.25)


def q3(s):
    return s.quantile(0.75)


def iqr(s):
    return s.quantile(0.75) - s.quantile(0.25)


overall = (
    df.groupby(["pipeline", "method", "threshold", "cardinality"], dropna=False)
    .agg(
        n=("instance", "count"),
        service_rate_pct_change_mean=("service_rate_pct_change", "mean"),
        service_rate_pct_change_median=("service_rate_pct_change", "median"),
        service_rate_pct_change_std=("service_rate_pct_change", "std"),
        service_rate_pct_change_q1=("service_rate_pct_change", q1),
        service_rate_pct_change_q3=("service_rate_pct_change", q3),
        service_rate_pct_change_iqr=("service_rate_pct_change", iqr),
        total_time_pct_change_mean=("total_time_pct_change", "mean"),
        total_time_pct_change_median=("total_time_pct_change", "median"),
        total_time_pct_change_std=("total_time_pct_change", "std"),
        total_time_pct_change_q1=("total_time_pct_change", q1),
        total_time_pct_change_q3=("total_time_pct_change", q3),
        total_time_pct_change_iqr=("total_time_pct_change", iqr),
    )
    .reset_index()
)
overall.to_csv(OUT / "card2_firstcheck_overall.csv", index=False)

print("Rows by pipeline/method/threshold/cardinality:")
print(df.groupby(["pipeline", "method", "threshold", "cardinality"], dropna=False).size())
print()
print(
    df[
        [
            "pipeline", "instance", "method", "threshold", "cardinality", "service_rate", "total_time",
            "service_rate_pct_change", "total_time_pct_change",
        ]
    ].sort_values(["pipeline", "instance", "method", "threshold", "cardinality"]).to_string(index=False)
)
print()
print("Mean pct_change vs. baseline, by pipeline/method/threshold/cardinality:")
print(overall.to_string(index=False))
print()
print("Saved:")
print(OUT / "card2_firstcheck_long.csv")
print(OUT / "card2_firstcheck_wide.csv")
print(OUT / "card2_firstcheck_overall.csv")


# 2026-07-09: threshold-sensitivity plots (median line + IQR band per pipeline/method),
# cardinality=2 only - mc3 doesn't have enough thresholds/instances yet for a curve.
def plot_metric(metric_prefix, ylabel, filename):
    plot_df = overall[(overall["cardinality"] == 2) & overall["method"].str.startswith("gnn_")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, pipeline in zip(axes, ["coaml", "rh"]):
        sub = plot_df[plot_df["pipeline"] == pipeline]
        for method, color in [("gnn_v1", "tab:blue"), ("gnn_v2", "tab:orange")]:
            m = sub[sub["method"] == method].sort_values("threshold")
            if m.empty:
                continue
            ax.plot(m["threshold"], m[f"{metric_prefix}_median"], marker="o", color=color, label=method)
            ax.fill_between(
                m["threshold"], m[f"{metric_prefix}_q1"], m[f"{metric_prefix}_q3"],
                color=color, alpha=0.2,
            )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(pipeline.upper())
        ax.set_xlabel("Pruner threshold")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    fig.suptitle(f"{ylabel} vs. threshold (median + IQR band across instances)")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {OUT / filename}")


plot_metric("total_time_pct_change", "Runtime change vs. baseline (%)", "card2_firstcheck_runtime_by_threshold.png")
plot_metric("service_rate_pct_change", "Service-rate change vs. baseline (%)", "card2_firstcheck_service_rate_by_threshold.png")
