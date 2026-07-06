from pathlib import Path
import re
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2] / "outputs" / "outputs"
OUT_LONG = Path(__file__).resolve().parents[2] / "outputs" / "threshold_sweep_rh_long.csv"
OUT_WIDE = Path(__file__).resolve().parents[2] / "outputs" / "threshold_sweep_rh_wide.csv"

RUN_SOURCES = [
    ("rh", "eval_rh_pw5_t0.2", True, 0.2),
    ("rh", "eval_rh_pw5_t0.3", True, 0.3),
    ("rh", "eval_rh_pw5_t0.4", True, 0.4),
]
BASELINE_FILES = [
    ("rh", Path(__file__).resolve().parents[2] / "outputs" / "eval_rh_no_learning_baseline_summary.csv"),
]

def get_class(instance):
    if instance.startswith("lrc"):
        return "LRC"
    if instance.startswith("lr"):
        return "LR"
    if instance.startswith("lc"):
        return "LC"
    return "unknown"

def parse_log(log_path, pipeline, threshold, pruner):
    instance = log_path.relative_to(ROOT / log_path.parts[log_path.parts.index("outputs") + 1]).parts[0] if False else None
    # Easier: instance is first part below source directory
    text = log_path.read_text(errors="ignore")

    rtv_match = re.search(r"(\d+) RTV combos / (\d+) feasible trip costs generated", text)
    feat_match = re.search(r"84 features for (\d+) items created in ([\d.]+) s", text)
    card2_match = re.search(r"(\d+) cardinality 2 trips generated", text)
    allowed_match = re.search(r"Allowed request pairs: (\d+)", text)

    stats_match = re.search(r"Stats:\s*\n\s*(\{.*?\n\})", text, re.S)
    if not stats_match:
        print(f"WARNING: no stats found in {log_path}")
        return None

    try:
        stats = json.loads(stats_match.group(1))
    except Exception as e:
        print(f"WARNING: could not parse stats in {log_path}: {e}")
        return None

    total_requests = stats.get("total_requests")
    serviced = stats.get("serviced")

    return {
        "pipeline": pipeline,
        "pruner": pruner,
        "threshold": threshold,
        "serviced": serviced,
        "total_requests": total_requests,
        "service_rate": serviced / total_requests if total_requests else None,
        "vmt": stats.get("vmt"),
        "pmt": stats.get("pmt"),
        "vmt_over_pmt": stats.get("vmt_over_pmt"),
        "vmt_over_pmt_woDepot": stats.get("vmt_over_pmt_woDepot"),
        "average_wait_time": stats.get("average_wait_time"),
        "average_detour": stats.get("average_detour"),
        "average_dropoff_goal_lateness": stats.get("average_dropoff_goal_lateness"),
        "total_time": stats.get("total_time"),
        "allowed_request_pairs_iter0": int(allowed_match.group(1)) if allowed_match else None,
        "cardinality2_trips_iter0": int(card2_match.group(1)) if card2_match else None,
        "rtv_combos_iter0": int(rtv_match.group(1)) if rtv_match else None,
        "trip_costs_iter0": int(rtv_match.group(2)) if rtv_match else None,
        "feature_items_iter0": int(feat_match.group(1)) if feat_match else None,
        "feature_time_iter0": float(feat_match.group(2)) if feat_match else None,
        "log_path": str(log_path),
    }


rows = []

# Baselines from existing summary CSVs
for pipeline, csv_path in BASELINE_FILES:
    if not csv_path.exists():
        print(f"WARNING: missing baseline CSV: {csv_path}")
        continue

    df_base = pd.read_csv(csv_path)
    for _, r in df_base.iterrows():
        rows.append({
            "pipeline": pipeline,
            "instance": r["instance"],
            "class": get_class(r["instance"]),
            "pruner": False,
            "threshold": 0.0,
            "variant": f"{pipeline}_baseline",
            "serviced": r.get("serviced"),
            "total_requests": r.get("total_requests"),
            "service_rate": r.get("service_rate"),
            "vmt": r.get("vmt"),
            "pmt": r.get("pmt"),
            "vmt_over_pmt": r.get("vmt_over_pmt"),
            "vmt_over_pmt_woDepot": r.get("vmt_over_pmt_woDepot"),
            "average_wait_time": r.get("average_wait_time"),
            "average_detour": r.get("average_detour"),
            "average_dropoff_goal_lateness": r.get("average_dropoff_goal_lateness"),
            "total_time": r.get("total_time"),
            "allowed_request_pairs_iter0": r.get("allowed_request_pairs_iter0"),
            "cardinality2_trips_iter0": r.get("cardinality2_trips_iter0"),
            "rtv_combos_iter0": r.get("rtv_combos_iter0"),
            "trip_costs_iter0": r.get("trip_costs_iter0"),
            "feature_items_iter0": r.get("feature_items_iter0"),
            "feature_time_iter0": r.get("feature_time_iter0"),
            "log_path": r.get("log_path"),
        })

# Threshold runs from logs
for pipeline, folder, pruner, threshold in RUN_SOURCES:
    source_root = ROOT / folder

    if not source_root.exists():
        print(f"WARNING: missing run folder: {source_root}")
        continue

    for log_path in source_root.glob("*/*/*/main.log"):
        instance = log_path.relative_to(source_root).parts[0]
        parsed = parse_log(log_path, pipeline, threshold, pruner)
        if parsed is None:
            continue

        parsed["instance"] = instance
        parsed["class"] = get_class(instance)
        parsed["variant"] = f"{pipeline}_pw5_t{threshold}"
        rows.append(parsed)


df = pd.DataFrame(rows)

if df.empty:
    print("No runs parsed.")
else:
    df = df.sort_values(["pipeline", "class", "instance", "threshold"])
    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_LONG, index=False)

    # Wide table: one row per pipeline-instance, columns per threshold
    metrics = [
        "service_rate",
        "total_time",
        "rtv_combos_iter0",
        "trip_costs_iter0",
        "feature_time_iter0",
    ]

    wide = df.pivot_table(
        index=["class", "instance"],
        columns="threshold",
        values=metrics,
        aggfunc="first",
    )

    wide.columns = [f"{metric}_t{str(th).replace('.', '')}" for metric, th in wide.columns]
    wide = wide.reset_index()

    # Percentage changes vs baseline threshold 0.0
    for metric in metrics:
        base_col = f"{metric}_t00"
        for th in [0.2, 0.3, 0.4]:
            th_col = f"{metric}_t{str(th).replace('.', '')}"
            if base_col in wide.columns and th_col in wide.columns:
                wide[f"{metric}_change_t{str(th).replace('.', '')}"] = wide[th_col] - wide[base_col]
                wide[f"{metric}_pct_t{str(th).replace('.', '')}"] = (wide[th_col] / wide[base_col] - 1) * 100

    wide.to_csv(OUT_WIDE, index=False)

    print(f"Saved long CSV: {OUT_LONG}")
    print(f"Saved wide CSV: {OUT_WIDE}")
    print("Rows by threshold:")
    print(df.groupby("threshold").size())

    print()
    print("Mean summary:")
    print(df.groupby("threshold")[
        ["service_rate", "total_time", "rtv_combos_iter0", "trip_costs_iter0"]
    ].mean())
