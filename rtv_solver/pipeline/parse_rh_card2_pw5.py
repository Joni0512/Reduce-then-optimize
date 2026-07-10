from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "new_tests" / "card2_pw5"
OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

INSTANCES = [
    "lc108", "lc109", "lc207", "lc208",
    "lr111", "lr112", "lr210", "lr211",
    "lrc107", "lrc108", "lrc207", "lrc208",
]

METRICS = [
    "service_rate",
    "total_time",
    "serviced",
    "total_requests",
]

def get_class(instance):
    if instance.startswith("lrc"):
        return "LRC"
    if instance.startswith("lc"):
        return "LC"
    if instance.startswith("lr"):
        return "LR"
    return "unknown"

def latest_results_json(folder):
    files = list(folder.rglob("results.json"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def read_result(path):
    data = json.load(open(path))
    stats = data["stats"]

    serviced = stats.get("serviced")
    total_requests = stats.get("total_requests")
    service_rate = serviced / total_requests if total_requests else None

    return {
        "service_rate": service_rate,
        "total_time": stats.get("total_time"),
        "serviced": serviced,
        "total_requests": total_requests,
    }

rows = []

for instance in INSTANCES:
    p = latest_results_json(BASE / "rh_baseline" / instance)
    if p:
        row = {
            "pipeline": "rh",
            "class": get_class(instance),
            "instance": instance,
            "threshold": 0.0,
            "use_pruner": False,
            "result_path": str(p),
        }
        row.update(read_result(p))
        rows.append(row)

    for th in [0.1, 0.2, 0.3, 0.4, 0.5]:
        tag = str(th).replace(".", "")
        p = latest_results_json(BASE / f"rh_t{tag}" / instance)
        if p:
            row = {
                "pipeline": "rh",
                "class": get_class(instance),
                "instance": instance,
                "threshold": th,
                "use_pruner": True,
                "result_path": str(p),
            }
            row.update(read_result(p))
            rows.append(row)

df = pd.DataFrame(rows)

base = df[df["threshold"] == 0.0][
    ["instance"] + METRICS
].rename(columns={m: f"{m}_baseline" for m in METRICS})

df = df.merge(base, on="instance", how="left")

for m in METRICS:
    df[f"{m}_delta"] = df[m] - df[f"{m}_baseline"]
    df[f"{m}_pct_change"] = (df[m] / df[f"{m}_baseline"] - 1) * 100

df.to_csv(OUT / "rh_pruning_summary_long.csv", index=False)

wide = df.pivot_table(
    index=["pipeline", "class", "instance"],
    columns="threshold",
    values=METRICS,
    aggfunc="first",
)

wide.columns = [
    f"{metric}_t{str(th).replace('.', '')}"
    for metric, th in wide.columns
]
wide = wide.reset_index()

for m in METRICS:
    base_col = f"{m}_t00"
    for th in [0.1, 0.2, 0.3, 0.4, 0.5]:
        suffix = str(th).replace(".", "")
        th_col = f"{m}_t{suffix}"
        if base_col in wide.columns and th_col in wide.columns:
            wide[f"{m}_delta_t{suffix}"] = wide[th_col] - wide[base_col]
            wide[f"{m}_pct_t{suffix}"] = (wide[th_col] / wide[base_col] - 1) * 100

wide.to_csv(OUT / "rh_pruning_summary_wide.csv", index=False)

overall = (
    df.groupby("threshold")[["service_rate", "total_time", "serviced", "total_requests"]]
    .mean()
    .reset_index()
)
overall.to_csv(OUT / "rh_pruning_overall_runtime_service.csv", index=False)

print("Saved:")
print(OUT / "rh_pruning_summary_long.csv")
print(OUT / "rh_pruning_summary_wide.csv")
print(OUT / "rh_pruning_overall_runtime_service.csv")
print()
print("Rows by threshold:")
print(df.groupby("threshold").size())
print()
print("Mean runtime/service:")
print(df.groupby("threshold")[["service_rate", "total_time"]].mean().round(4))