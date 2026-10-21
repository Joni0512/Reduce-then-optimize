"""
2026-10-10: Threshold-parametrized copy of summarize_per_instance_mixedclass.py (which hardcodes
t0.5). Per-instance service rate of the c2 mixedclass Request-only (expert-prefix-label) runs at a
given threshold, to check per instance where the pruner hurts / whether the collapse hits a floor
(user question, 2026-10-10: identical pooled rates across configs at t=0.7/0.8 suggest only lc
instances are still served).

Reads the "Best val service rate = ... at epoch N" line of each run's own .log in the repo root and
then outputs/outputs/{run}/batch_lilim_coaml_seed{seed}/mc2_*/val/epoch_{N-1}/{instance}/results.json
(val/epoch_ dirs are 0-indexed, the logged epoch is 1-indexed - same fix as 2026-09-24). Only exists on
the cluster, run it there.

Usage (cluster): ./venv/bin/python3 -m rtv_solver.pipeline.summarize_per_instance_mixedclass_threshold <threshold> [seed ...]
  e.g. ... 0.8 42 1        (default seeds: 42 1)
Writes outputs/per_instance_mixedclass_prefixlabel_t<threshold>.csv (same columns as the t0.5 CSV).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS = ["bi200_ss100", "bi400_ss80", "bi400_ss200"]


def find_best_epoch(log_path: Path):
    if not log_path.exists():
        return None
    m = re.search(r"Best val service rate = [\d.]+ at epoch (\d+)", log_path.read_text(errors="ignore"))
    return int(m.group(1)) if m else None


def find_run_dir(name: str, seed: int):
    base = REPO_ROOT / "outputs" / "outputs" / name
    candidates = sorted(base.glob(f"batch_lilim_coaml_seed{seed}/mc2_*"))
    return candidates[-1] if candidates else None


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    thr = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]] or [42, 1]
    rows = []
    for cfg in CONFIGS:
        for seed in seeds:
            name = f"sil_training_{cfg}_mixedclass_requestpruner_prefixlabel_t{thr}_seed{seed}"
            epoch = find_best_epoch(REPO_ROOT / f"{name}.log")
            if epoch is None:
                print(f"skip {cfg} seed{seed}: no 'Best val service rate' line in {name}.log")
                continue
            run_dir = find_run_dir(name, seed)
            if run_dir is None:
                print(f"skip {cfg} seed{seed}: no run dir")
                continue
            val_epoch_dir = run_dir / "val" / f"epoch_{epoch - 1}"
            if not val_epoch_dir.exists():
                print(f"skip {cfg} seed{seed}: {val_epoch_dir} missing")
                continue
            for inst_dir in sorted(val_epoch_dir.iterdir()):
                results_path = inst_dir / "results.json"
                if not results_path.exists():
                    continue
                stats = json.load(open(results_path)).get("stats", {})
                serviced, total = stats.get("serviced"), stats.get("total_requests")
                if serviced is None or not total:
                    continue
                rows.append({"config": cfg, "seed": seed, "instance": inst_dir.name,
                             "serviced": serviced, "total": total, "rate": serviced / total})
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "outputs" / f"per_instance_mixedclass_prefixlabel_t{thr}.csv"
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows written to {out}")
    summary = df.groupby(["config", "instance"])["rate"].agg(["mean", "std", "count"]).reset_index()
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.float_format", "{:.3f}".format):
        print(summary.sort_values(["config", "mean"]).to_string(index=False))


if __name__ == "__main__":
    main()
