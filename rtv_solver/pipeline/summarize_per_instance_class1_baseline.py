"""
2026-09-24: Per-instance breakdown for class1 BASELINE (no pruner) runs - checking
whether the lc-vs-lr/lrc gap already exists without any pruner at class1 too (same
question as summarize_per_instance_mixedclass_baseline.py, now for class1). Depends on
the class1 baseline fill-in runs (run_sil_class1_baseline_only.sh, submitted 2026-09-24)
having progressed far enough to have a val/epoch_N directory.

Run (on the cluster): ./venv/bin/python3 -m rtv_solver.pipeline.summarize_per_instance_class1_baseline
"""
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS = ["bi200_ss100", "bi400_ss80", "bi400_ss200"]
SEEDS = [42, 1, 2, 3, 4]


def find_best_epoch(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    m = re.search(r"Best val service rate = [\d.]+ at epoch (\d+)", text)
    return int(m.group(1)) if m else None


def find_run_dir(cfg: str, seed: int) -> Path | None:
    base = REPO_ROOT / "outputs" / "outputs" / f"sil_training_{cfg}_class1_baseline_seed{seed}"
    candidates = sorted(base.glob(f"batch_lilim_coaml_seed{seed}/mc2_*"))
    return candidates[-1] if candidates else None


def main():
    rows = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            log_path = REPO_ROOT / f"sil_training_{cfg}_class1_baseline_seed{seed}.log"
            epoch = find_best_epoch(log_path)
            if epoch is None:
                print(f"skip {cfg} seed{seed}: no 'Best val service rate' line found (run not finished yet?)")
                continue
            run_dir = find_run_dir(cfg, seed)
            if run_dir is None:
                print(f"skip {cfg} seed{seed}: no run dir found")
                continue
            # 2026-09-24 fix: see summarize_per_instance_mixedclass.py - directory is 0-indexed,
            # reported "Best val service rate ... at epoch N" is 1-indexed.
            val_epoch_dir = run_dir / "val" / f"epoch_{epoch - 1}"
            if not val_epoch_dir.exists():
                print(f"skip {cfg} seed{seed}: {val_epoch_dir} missing")
                continue
            for inst_dir in sorted(val_epoch_dir.iterdir()):
                results_path = inst_dir / "results.json"
                if not results_path.exists():
                    continue
                stats = json.load(open(results_path)).get("stats", {})
                serviced = stats.get("serviced")
                total = stats.get("total_requests")
                if serviced is None or total is None or total == 0:
                    continue
                rows.append({
                    "config": cfg, "seed": seed, "instance": inst_dir.name,
                    "serviced": serviced, "total": total, "rate": serviced / total,
                })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/per_instance_class1_baseline.csv", index=False)
    print(f"\n{len(df)} rows written to outputs/per_instance_class1_baseline.csv")

    if df.empty:
        print("No data yet - baseline runs likely still in progress.")
        return

    print("\n=== Mean rate per instance, per config (across available seeds) - class1 BASELINE (no pruner) ===")
    summary = df.groupby(["config", "instance"])["rate"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.sort_values(["config", "mean"])
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    summary.to_csv("outputs/per_instance_class1_baseline_summary.csv", index=False)


if __name__ == "__main__":
    main()
