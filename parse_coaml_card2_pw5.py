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

def cls(x):
    if x.startswith("lrc"): return "LRC"
    if x.startswith("lc"): return "LC"
    if x.startswith("lr"): return "LR"
    return "unknown"

def latest(folder):
    files = list(folder.rglob("results.json"))
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

for inst in INSTANCES:
    p = latest(BASE / "coaml_baseline" / inst)
    if p:
        rows.append({
            "pipeline": "coaml", "class": cls(inst), "instance": inst,
            "threshold": 0.0, "use_pruner": False, "result_path": str(p),
            **read(p)
        })

    for th in [0.1, 0.2, 0.3, 0.4, 0.5]:
        tag = str(th).replace(".", "")
        p = latest(BASE / f"coaml_t{tag}" / inst)
        if p:
            rows.append({
                "pipeline": "coaml", "class": cls(inst), "instance": inst,
                "threshold": th, "use_pruner": True, "result_path": str(p),
                **read(p)
            })

df = pd.DataFrame(rows)

metrics = [
    "service_rate", "total_time", "serviced", "total_requests",
    "vmt", "average_wait_time", "average_detour"
]

base = df[df.threshold == 0.0][["instance"] + metrics].rename(
    columns={m: f"{m}_baseline" for m in metrics}
)
df = df.merge(base, on="instance", how="left")

for m in metrics:
    df[f"{m}_delta"] = df[m] - df[f"{m}_baseline"]
    df[f"{m}_pct_change"] = (df[m] / df[f"{m}_baseline"] - 1) * 100

df.to_csv(OUT / "coaml_pruning_summary_long.csv", index=False)

wide = df.pivot_table(
    index=["pipeline", "class", "instance"],
    columns="threshold",
    values=metrics,
    aggfunc="first",
)
wide.columns = [f"{m}_t{str(t).replace('.', '')}" for m, t in wide.columns]
wide = wide.reset_index()

for m in metrics:
    base_col = f"{m}_t00"
    for th in [0.1, 0.2, 0.3, 0.4, 0.5]:
        suf = str(th).replace(".", "")
        th_col = f"{m}_t{suf}"
        if base_col in wide.columns and th_col in wide.columns:
            wide[f"{m}_delta_t{suf}"] = wide[th_col] - wide[base_col]
            wide[f"{m}_pct_t{suf}"] = (wide[th_col] / wide[base_col] - 1) * 100

wide.to_csv(OUT / "coaml_pruning_summary_wide.csv", index=False)

overall = (
    df.groupby("threshold")[["service_rate", "total_time", "serviced", "vmt"]]
    .mean()
    .reset_index()
)
overall.to_csv(OUT / "coaml_pruning_overall_runtime_service.csv", index=False)

print("Rows by threshold:")
print(df.groupby("threshold").size())
print()
print("Mean runtime/service:")
print(df.groupby("threshold")[["service_rate", "total_time"]].mean().round(4))
print()
print("Saved:")
print(OUT / "coaml_pruning_summary_long.csv")
print(OUT / "coaml_pruning_summary_wide.csv")
print(OUT / "coaml_pruning_overall_runtime_service.csv")
