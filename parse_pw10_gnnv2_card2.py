from pathlib import Path
import json
import pandas as pd

# 2026-07-07: parser for the new v2 GNN pruner (pos_weight=10, threshold=0.5) RH+COAML
# test runs. Structure differs from the old pw5 parsers (parse_rh_card2_pw5.py /
# parse_coaml_card2_pw5.py):
#   RH:    outputs/outputs/new_tests/card2_b400_s100_pw10_gnnv2/rh_pruner_t05/<inst>/run_offline_.../<inst>_<ts>/final/results.json
#   COAML: outputs/outputs/new_tests/card2_b400_s100_pw10_gnnv2/coaml_pruner_t05/<inst>/batch_lilim_coaml_seed42/<run>/final/results.json
# COAML instances each have an extra, older run folder (the one that trained on the
# full default li_lim directory before the --input_dir "" fix) which has no "final/"
# folder at all - rglob("results.json") naturally skips those, so "latest" here is
# safe (only the corrected single-instance runs produce a results.json).
ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "outputs" / "new_tests" / "card2_b400_s100_pw10_gnnv2"
BASELINE_BASE = ROOT / "outputs" / "new_tests" / "card2_b400_s100_pw5"
OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

INSTANCES = [
    "lc108", "lc109", "lc207", "lc208",
    "lr111", "lr112", "lr210", "lr211",
    "lrc107", "lrc108", "lrc207", "lrc208",
]


def cls(x):
    if x.startswith("lrc"):
        return "LRC"
    if x.startswith("lc"):
        return "LC"
    if x.startswith("lr"):
        return "LR"
    return "unknown"


def latest(folder: Path):
    files = list(folder.rglob("results.json"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def read(path: Path):
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
        "average_dropoff_goal_lateness": stats.get("average_dropoff_goal_lateness"),
        "violations": len(data.get("violations", [])),
        "result_path": str(path),
    }


rows = []

for pipeline, pruner_dir, baseline_dir in [
    ("rh", BASE / "rh_pruner_t05", BASELINE_BASE / "rh_baseline"),
    ("coaml", BASE / "coaml_pruner_t05", BASELINE_BASE / "coaml_baseline"),
]:
    for inst in INSTANCES:
        p = latest(pruner_dir / inst)
        if p is not None:
            rows.append({
                "pipeline": pipeline, "class": cls(inst), "instance": inst,
                "threshold": 0.5, "use_pruner": True, "gnn_version": "v2_pw10",
                **read(p),
            })

        bp = latest(baseline_dir / inst)
        if bp is not None:
            rows.append({
                "pipeline": pipeline, "class": cls(inst), "instance": inst,
                "threshold": 0.0, "use_pruner": False, "gnn_version": None,
                **read(bp),
            })

df = pd.DataFrame(rows)

metrics = [
    "service_rate", "total_time", "serviced", "total_requests",
    "vmt", "average_wait_time", "average_detour",
    "average_dropoff_goal_lateness", "violations",
]

# baseline (threshold 0.0) only exists for lc108/lc109/lc207/lc208 so far -
# merge is left-join, lr/lrc rows simply get NaN baseline/delta columns.
base = df[df.threshold == 0.0][["pipeline", "instance"] + metrics].rename(
    columns={m: f"{m}_baseline" for m in metrics}
)
df = df.merge(base, on=["pipeline", "instance"], how="left")

for m in metrics:
    df[f"{m}_delta"] = df[m] - df[f"{m}_baseline"]
    df[f"{m}_pct_change"] = (df[m] / df[f"{m}_baseline"] - 1) * 100

df.to_csv(OUT / "pw10_gnnv2_summary_long.csv", index=False)

pruner_only = df[df.use_pruner].sort_values(["pipeline", "class", "instance"])
pruner_only.to_csv(OUT / "pw10_gnnv2_pruner_only.csv", index=False)

overall = (
    df.groupby(["pipeline", "use_pruner"])[["service_rate", "total_time", "vmt", "violations"]]
    .mean()
    .reset_index()
)
overall.to_csv(OUT / "pw10_gnnv2_overall_by_pipeline.csv", index=False)

print("Rows by pipeline/threshold:")
print(df.groupby(["pipeline", "threshold"]).size())
print()
print("Mean service_rate / total_time / vmt / violations by pipeline (pruner vs baseline where available):")
print(overall.round(4))
print()
print("Saved:")
print(OUT / "pw10_gnnv2_summary_long.csv")
print(OUT / "pw10_gnnv2_pruner_only.csv")
print(OUT / "pw10_gnnv2_overall_by_pipeline.csv")
