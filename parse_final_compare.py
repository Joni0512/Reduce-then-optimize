"""
Parses the RH/COAML baseline vs. v1_final vs. v2_final comparison runs
(outputs/outputs/new_tests/final_compare/bi{X}_ss{Y}/...) into a table with
runtime/service-rate and their % change vs. the no-pruner baseline.

Only pipelines/instances that actually have a final/results.json are included -
missing runs (e.g. an in-progress COAML sweep) are skipped with a warning
instead of crashing, so this can be re-run at any point to see partial results.

Usage:
    python3 parse_final_compare.py --bi 400 --ss 100
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

INSTANCES = [
    "lc108", "lc109", "lc207", "lc208",
    "lr111", "lr112", "lr210", "lr211",
    "lrc107", "lrc108", "lrc207", "lrc208",
]

def runs_for_threshold(t: str, seed_suffix: str = ""):
    """(pipeline, method, folder_name). Baseline has no pruner, so it's
    threshold-independent -- always read from the tXX baseline folders
    rather than requiring a per-threshold rerun. seed_suffix is e.g.
    '_seed1' for a non-default-seed rerun (folders created by
    run_*_only_window.sh with a seed arg != 42), or '' for seed 42."""
    return [
        ("rh", "baseline", f"rh_baseline{seed_suffix}"),
        ("rh", "v1_final", f"rh_v1final_t{t}{seed_suffix}"),
        ("rh", "v2_final", f"rh_v2final_t{t}{seed_suffix}"),
        ("coaml", "baseline", f"coaml_baseline{seed_suffix}"),
        ("coaml", "v1_final", f"coaml_v1final_t{t}{seed_suffix}"),
        ("coaml", "v2_final", f"coaml_v2final_t{t}{seed_suffix}"),
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
    files = list(folder.rglob("final/results.json"))
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
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bi", type=int, default=400, help="batch_interval")
    parser.add_argument("--ss", type=int, default=100, help="step_size")
    parser.add_argument("--threshold", default="05", help="threshold suffix as used in folder names, e.g. '05' or '04'")
    parser.add_argument("--seed", type=int, default=42, help="seed used for the run (42 = default folders, anything else reads _seedN folders)")
    args = parser.parse_args()

    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    base = ROOT / "outputs" / "new_tests" / "final_compare" / f"bi{args.bi}_ss{args.ss}"
    out_dir = base / "analysis" / f"t{args.threshold}{seed_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pipeline, method, folder_name in runs_for_threshold(args.threshold, seed_suffix):
        for inst in INSTANCES:
            p = latest(base / folder_name / inst)
            if p is None:
                print(f"WARNING: missing {pipeline}/{method}/{inst} (not run yet or still running)")
                continue
            rows.append({
                "pipeline": pipeline,
                "class": cls(inst),
                "instance": inst,
                "method": method,
                **read(p),
            })

    if not rows:
        raise SystemExit(f"No results found under {base}")

    df = pd.DataFrame(rows)

    base_df = df[df.method == "baseline"][["pipeline", "instance", "service_rate", "total_time"]].rename(
        columns={"service_rate": "service_rate_baseline", "total_time": "total_time_baseline"}
    )
    df = df.merge(base_df, on=["pipeline", "instance"], how="left")
    df["service_rate_pct_change"] = (df["service_rate"] / df["service_rate_baseline"] - 1) * 100
    df["total_time_pct_change"] = (df["total_time"] / df["total_time_baseline"] - 1) * 100

    long_path = out_dir / "final_compare_long.csv"
    df.to_csv(long_path, index=False)

    overall = df.groupby(["pipeline", "method"])[
        ["service_rate", "total_time", "service_rate_pct_change", "total_time_pct_change"]
    ].mean().reset_index()
    overall_path = out_dir / "final_compare_overall.csv"
    overall.to_csv(overall_path, index=False)

    print(f"\nRows found: {len(df)} (of {len(runs_for_threshold(args.threshold, seed_suffix)) * len(INSTANCES)} possible)")
    print(df.groupby(["pipeline", "method"]).size())
    print()
    print("Mean % change vs. baseline, by pipeline/method:")
    print(overall.to_string(index=False))
    print()
    print("Saved:", long_path)
    print("Saved:", overall_path)


if __name__ == "__main__":
    main()
