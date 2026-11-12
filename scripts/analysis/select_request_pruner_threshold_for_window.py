"""
2026-11-06: Rule A threshold for ONE window-specific Request Pruner (wrapper around evaluate() of select_request_pruner_by_vehicle_loss.py, which is NOT modified).
Rule A (CLAUDE.md 2026-10-20): among the (checkpoint x threshold) candidates with pruning rate >= --min_prune (default 30 %), the smallest lost_1st;
tie-break smaller lost_anyK, then higher pruning rate. Thresholds = THRESHOLDS of the original script (0.1 ... 0.9), min_retention 0.3, K=2 (RHO cardinality 2).
Data: dataset/request_pruning_bi<BI>_ss<SS> (RHO labels); the 8 pruner-VAL instances come from ITS manifest.json (split == "val"), never the 12 test instances (no leakage).
(The original script reads them from the *_prefix dataset, which does not exist for these windows; the val split is the same.)
Checkpoints: every *_best_val_f3.pt under outputs/request_pruner_mlp_bi<BI>_ss<SS>/.
If no candidate reaches the minimum pruning rate the script exits with code 1 (no guessing).

Run: ./venv/bin/python3 scripts/analysis/select_request_pruner_threshold_for_window.py --bi 100 --ss 25
Output: outputs/request_pruner_threshold_rule_a_bi<BI>_ss<SS>.json (threshold, checkpoint, prune rate, lost_1st, lost_anyK) and the candidate table as .csv
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_request_pruner_by_vehicle_loss as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", type=int, required=True)
    ap.add_argument("--ss", type=int, required=True)
    ap.add_argument("--min_prune", type=float, default=0.30)
    ap.add_argument("--min_retention", type=float, default=0.3)
    a = ap.parse_args()
    lab = f"bi{a.bi}_ss{a.ss}"
    ds = REPO / f"dataset/request_pruning_{lab}"
    info = dict(rho_dataset=ds, prefix_dataset=ds, step=float(a.ss), ckpt_dirs=[REPO / f"outputs/request_pruner_mlp_{lab}"])
    res = S.evaluate(lab, info, a.min_retention)
    res.to_csv(REPO / f"outputs/request_pruner_threshold_rule_a_{lab}.csv", index=False)
    ok = res[res.prune_rate >= 100 * a.min_prune].sort_values(["lost_1st", "lost_anyK", "prune_rate"], ascending=[True, True, False])
    print(f"== {lab}: {len(res)} candidates (checkpoint x threshold), {int(res.vehicles.iloc[0])} vehicles in window 0, {len(ok)} reach {100 * a.min_prune:.0f} % pruning")
    print(ok.head(8).to_string(index=False) if len(ok) else "none reaches the minimum pruning rate")
    if not len(ok):
        sys.exit(1)
    best = ok.iloc[0]
    ck = next((REPO / f"outputs/request_pruner_mlp_{lab}").rglob(f"request_pruner_mlp_{best.checkpoint}_best_val_f3.pt"))
    out = dict(window=lab, threshold=float(best.threshold), checkpoint=str(ck.relative_to(REPO)), prune_rate=float(best.prune_rate),
               lost_1st=float(best.lost_1st), lost_anyK=float(best.lost_anyK), vehicles=int(best.vehicles), min_prune=a.min_prune, min_retention=a.min_retention)
    (REPO / f"outputs/request_pruner_threshold_rule_a_{lab}.json").write_text(json.dumps(out, indent=2))
    print("RULE A:", out)


if __name__ == "__main__":
    main()
