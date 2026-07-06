from pathlib import Path
import re
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2] / "outputs" / "outputs" / "eval_baseline"
OUT = Path(__file__).resolve().parents[2] / "outputs" / "eval_mixed_all_baseline_summary.csv"

rows = []

def get_class(instance):
    if instance.startswith("lrc"):
        return "LRC"
    if instance.startswith("lr"):
        return "LR"
    if instance.startswith("lc"):
        return "LC"
    return "unknown"

for log_path in ROOT.glob("*/*/*/main.log"):
    instance = log_path.relative_to(ROOT).parts[0]
    run_name = f"{instance}_mixed_all_baseline"

    variant = "mixed_all_baseline"
    pruner = False
    threshold = None

    text = log_path.read_text(errors="ignore")

    rtv_match = re.search(r"(\d+) RTV combos / (\d+) feasible trip costs generated", text)
    feat_match = re.search(r"84 features for (\d+) items created in ([\d.]+) s", text)
    card2_match = re.search(r"(\d+) cardinality 2 trips generated", text)
    allowed_match = re.search(r"Allowed request pairs: (\d+)", text)

    stats_match = re.search(r"Stats:\s*\n\s*(\{.*?\n\})", text, re.S)
    if not stats_match:
        print(f"WARNING: no stats found in {log_path}")
        continue

    try:
        stats = json.loads(stats_match.group(1))
    except Exception as e:
        print(f"WARNING: could not parse stats in {log_path}: {e}")
        continue

    total_requests = stats.get("total_requests")
    serviced = stats.get("serviced")
    service_rate = serviced / total_requests if total_requests else None

    rows.append({
        "instance": instance,
        "class": get_class(instance),
        "variant": variant,
        "pruner": pruner,
        "threshold": threshold,
        "run_name": run_name,

        "serviced": serviced,
        "total_requests": total_requests,
        "service_rate": service_rate,
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
    })

df = pd.DataFrame(rows)

if df.empty:
    print("No baseline runs parsed. Check ROOT path:", ROOT.resolve())
else:
    df = df.sort_values(["class", "instance"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Saved {len(df)} rows to {OUT}")
    print()
    print("Runs by class:")
    print(df.groupby(["class"]).size())
    print()
    print("Quick summary:")
    print(df.groupby(["class"])[
        ["service_rate", "vmt", "total_time", "trip_costs_iter0", "rtv_combos_iter0"]
    ].mean())
