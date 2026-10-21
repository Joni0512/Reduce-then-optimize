"""
2026-09-21: Trains the Request Pruner on EXPERT-PREFIX labels (see
build_request_pruner_prefix_dataset.py) and compares it against the existing RHO-label
pruner with the replay that matters for SIL - offline, no SIL run.

Metrics on the 6 class-1 SIL-val instances (unseen by pruner training), replaying
RequestPruner.filter_requests exactly like analyze_request_pruner_false_prunes.py
(validated against the real solver in window 0):
  - window 0 (state identical to a real SIL run => exact): share of vehicles that LOSE
    their first expert request (=> no positive imitation trip), and share that lose any of
    their first K=2;
  - all windows (approximation, RHO-baseline batches): share of prefix requests pruned;
  - pruning strength: pruned / all requests in the window (same denominator as the
    "kept X/Y" solver log line) = proxy for the runtime saving;
  - false-prune rate vs. the RHO assignment (what the old label measured), for reference.

Training: same MLP/HPs as the existing bi200/ss100 winner (hidden 32, 1 layer, dropout 0.2,
lr 0.01, seed 42), only pos_weight varied (recall matters: a missing prefix request is
expensive) - a full HP sweep is deliberately NOT run before we know whether the labels
matter at all.

Run: python -m rtv_solver.pipeline.train_eval_request_pruner_prefix_labels
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtv_solver.pipeline import analyze_request_pruner_false_prunes as A
from rtv_solver.pipeline.build_request_pruner_prefix_dataset import DST_DIR, K, expert_by_vehicle, prefix_labels
from rtv_solver.pipeline.train_request_pruner import load_all_splits, train_one_config

OUT_DIR = Path("outputs/request_pruner_mlp_bi200_ss100_prefix")
OLD_CKPT = A.CKPT
POS_WEIGHTS = [1.0, 2.0, 3.0]
THRESHOLDS = [0.3, 0.5]
MIN_RETENTIONS = [0.1, 0.3, 0.5]


def evaluate(ckpt, threshold, min_retention, feature_index):
    A.CKPT = Path(ckpt)
    A.THRESHOLD = threshold
    model, ck = A.load_model()
    frames, veh_rows = [], []
    for stem in A.SIL_VAL:
        df = A.replay_instance(stem, model, ck, feature_index)
        z = np.load(A.find_instance(stem)[1])
        df["prefix"] = prefix_labels(z["node_ids"], z["window_index"], stem) > 0.5
        pruned, floor_saved = A.apply_pruner(df, min_retention)
        df["pruned"] = pruned
        frames.append(df)
        # window 0: per-vehicle check with the fixed reference route (exact state)
        w0 = df[df.window == 0]
        pr = dict(zip(w0.request, w0.pruned))
        batch = set(w0.request)
        for seq in expert_by_vehicle(stem).values():
            proj = [r for r in seq if r in batch]
            if proj:
                veh_rows.append((pr[proj[0]], any(pr[r] for r in proj[:K])))
    d = pd.concat(frames, ignore_index=True)
    cand = ~(d["active"] | d["urgent"])
    fa = d["first_assign"] & cand
    v = np.array(veh_rows)
    return {
        "w0 veh. lose 1st expert req %": 100 * v[:, 0].mean(),
        "w0 veh. lose any of first K %": 100 * v[:, 1].mean(),
        "all-win prefix reqs pruned %": 100 * (d["prefix"] & d["pruned"]).sum() / d["prefix"].sum(),
        "pruned / all requests %": 100 * d["pruned"].sum() / len(d),
        "RHO false-prune rate %": 100 * (fa & d["pruned"]).sum() / max(fa.sum(), 1),
    }


def main():
    manifest = json.load(open(A.DATASET_DIR / "manifest.json"))
    feature_index = {n: i for i, n in enumerate(manifest["feature_names"])}

    train, val, test = load_all_splits(DST_DIR)
    ckpts = {"RHO labels (existing)": OLD_CKPT}
    for pw in POS_WEIGHTS:
        res = train_one_config(
            train, val, test, hidden_dim=32, num_hidden_layers=1, dropout=0.2,
            pos_weight=pw, learning_rate=0.01, seed=42, verbose=False,
            save_checkpoint=True, output_dir=OUT_DIR / f"pw{pw}",
        )
        ckpts[f"prefix labels, pos_weight {pw}"] = res["best_model_path"]
        print(f"trained pos_weight={pw}: best_val_f3={res['best_val_f3']:.3f} -> {res['best_model_path']}")

    rows = []
    for name, ck in ckpts.items():
        for t in THRESHOLDS:
            for mr in MIN_RETENTIONS:
                r = evaluate(ck, t, mr, feature_index)
                rows.append({"pruner": name, "threshold": t, "min_retention": mr, **r})
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "prefix_vs_rho_pruner_replay.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.1f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
