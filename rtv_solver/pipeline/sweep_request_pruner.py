"""
sweep_request_pruner.py
========================
2026-07-14/15: Phase 1 (sweep) of the request-pruner design discussion.

Runs the full hyperparameter grid over RequestPrunerMLP:

    hidden_dim         16, 32, 64
    num_hidden_layers  1, 2, 3
    dropout            0.0, 0.1, 0.2
    pos_weight         1.0, 1.5, 2.0, 3.0
    learning_rate      0.001, 0.003, 0.01, 0.03

3 x 3 x 3 x 4 x 4 = 432 configs, each trained over 5 seeds -> 2160 trainings.

--- Version history (why it looks the way it does) ---
v1 ranked configs by val_f3 at the fixed selection_threshold=0.5, single
seed. That ranking turned out to be confounded by pos_weight: pos_weight
shifts every sigmoid output up or down, which moves recall/precision/F3 at
a FIXED threshold without necessarily improving how well the model actually
ranks served vs. unserved requests. Groupby showed val_f3 climbing
monotonically with pos_weight (0.77 at pw=1.0 up to 0.87 at pw=3.0) while
val_roc_auc (threshold-independent) barely moved the same way - i.e. v1's
"winner" was mostly just whichever pos_weight pushed the most predictions
over 0.5, not the best-discriminating model.

v2 fixed that in two ways:
    1. Ranks by val_roc_auc (mean across seeds) instead of val_f3 at a fixed
       threshold - AUC only depends on the ranking of scores, so pos_weight
       can no longer game it by shifting the decision boundary.
    2. Runs every config over SEEDS (5 seeds) instead of one, because v1's
       single-seed run also showed pos_weight=1.0 configs varying wildly in
       val_roc_auc (std=0.136, min as low as 0.495 = random) - training
       instability, not necessarily a bad hyperparameter choice, and a
       single seed can't tell those apart. Ranking now uses the PER-CONFIG
       MEAN across 5 seeds, and reports the std alongside it so an unstable-
       but-lucky config doesn't look better than a config that's reliably
       good.

v3 (2026-07-15) adds two things:
    1. learning_rate as a swept hyperparameter - it was fixed at 0.01 for
       every v1/v2 run and never actually varied, an oversight caught only
       when reviewing what the deployed checkpoint's parameters were.
    2. dataset_dir/output_dir made parameters so this same sweep can be run
       once per window config (BATCH_INTERVAL/STEP_SIZE) to train a
       SEPARATE, window-config-specific model, instead of merging multiple
       window configs' data into one dataset - see the design discussion
       2026-07-15 (build_window_specific_request_pruners.py calls this once
       per config). A model is always deployed against one known window
       config at a time, so there was no need for one generalist model.

Only val metrics are used for ranking/selection - test is evaluated and
reported for every config, but never influences which one is picked, or it
stops being an honest held-out read (same principle as train_request_pruner.py).

Checkpoints are NOT saved for all trainings (save_checkpoint=False, see that
flag's docstring in train_request_pruner.py) - only the single winning
hyperparameter combination gets retrained once at the end with
save_checkpoint=True, using SEEDS[0] (42) so the persisted checkpoint
corresponds to one of the 5 seeds actually measured, not an unseen 6th one.

Output (under output_dir, default outputs/request_pruner_mlp/):
    sweep_detailed_5seed.csv   one row per (config, seed)
    sweep_summary_5seed.csv    one row per config, mean/std over 5 seeds, sorted by val_roc_auc
    <winning_run_name>/        checkpoint + metadata.json for the winning config

Run:
    python3 -m rtv_solver.pipeline.sweep_request_pruner
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import pandas as pd

from rtv_solver.pipeline.train_request_pruner import DATASET_DIR, OUTPUT_DIR, load_all_splits, train_one_config


HIDDEN_DIMS = [16, 32, 64]
NUM_HIDDEN_LAYERS = [1, 2, 3]
DROPOUTS = [0.0, 0.1, 0.2]
POS_WEIGHTS = [1.0, 1.5, 2.0, 3.0]
LEARNING_RATES = [0.001, 0.003, 0.01, 0.03]  # 2026-07-15: new - was fixed at 0.01 in v1/v2
SEEDS = [42, 1, 2, 3, 4]  # 42 first/kept as SEEDS[0] - reused for the final winning-config checkpoint below

SELECTION_THRESHOLD = 0.5  # matches train_one_config's fixed checkpoint-selection threshold, used only for precision/recall/f3 reporting, NOT for ranking (see module docstring)


def run_sweep(dataset_dir: Path = DATASET_DIR, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    train, val, test = load_all_splits(dataset_dir)

    grid = list(itertools.product(HIDDEN_DIMS, NUM_HIDDEN_LAYERS, DROPOUTS, POS_WEIGHTS, LEARNING_RATES))
    total_runs = len(grid) * len(SEEDS)
    print(f"Sweeping {len(grid)} configs x {len(SEEDS)} seeds = {total_runs} trainings...\n")

    rows = []
    t_sweep_start = time.time()
    run_index = 0

    for hidden_dim, num_hidden_layers, dropout, pos_weight, learning_rate in grid:
        for seed in SEEDS:
            run_index += 1
            result = train_one_config(
                train, val, test,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                dropout=dropout,
                pos_weight=pos_weight,
                learning_rate=learning_rate,
                seed=seed,
                verbose=False,
                save_checkpoint=False,  # many runs - see module docstring for why nothing is persisted here
                output_dir=output_dir,
            )

            val_m = result["metrics"]["val"][SELECTION_THRESHOLD]
            test_m = result["metrics"]["test"][SELECTION_THRESHOLD]

            rows.append({
                "hidden_dim": hidden_dim,
                "num_hidden_layers": num_hidden_layers,
                "dropout": dropout,
                "pos_weight": pos_weight,
                "learning_rate": learning_rate,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "train_time_seconds": result["train_time_seconds"],
                "val_roc_auc": val_m["roc_auc"], "val_pr_auc": val_m["pr_auc"],
                "val_recall": val_m["recall"], "val_precision": val_m["precision"], "val_f3": val_m["f3"],
                "test_roc_auc": test_m["roc_auc"], "test_pr_auc": test_m["pr_auc"],
                "test_recall": test_m["recall"], "test_precision": test_m["precision"], "test_f3": test_m["f3"],
            })

        if run_index % (5 * 24) == 0 or run_index == total_runs:  # progress line every 24 configs (120 runs)
            elapsed = time.time() - t_sweep_start
            print(f"  ... {run_index}/{total_runs} trainings done ({elapsed:.1f}s elapsed)")

    total_time = time.time() - t_sweep_start
    print(f"\nSweep finished in {total_time:.1f}s ({total_time / total_runs:.3f}s/training average)")

    detailed = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / "sweep_detailed_5seed.csv"
    detailed.to_csv(detailed_path, index=False)
    print(f"Detailed (per-seed) results written to {detailed_path}")

    # Aggregate across seeds: mean = expected performance, std = how much a
    # config's result depends on lucky/unlucky weight initialization. Ranked
    # by mean val_roc_auc (threshold-independent, not confounded by
    # pos_weight - see module docstring for why this replaced val_f3).
    hyperparam_cols = ["hidden_dim", "num_hidden_layers", "dropout", "pos_weight", "learning_rate"]
    metric_cols = [c for c in detailed.columns if c not in hyperparam_cols + ["seed"]]
    agg = detailed.groupby(hyperparam_cols)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index().sort_values("val_roc_auc_mean", ascending=False).reset_index(drop=True)

    summary_path = output_dir / "sweep_summary_5seed.csv"
    agg.to_csv(summary_path, index=False)
    print(f"Aggregated (per-config, mean+-std over {len(SEEDS)} seeds) summary written to {summary_path}")

    print(f"\n=== Top 10 configs by mean val_roc_auc (over {len(SEEDS)} seeds) ===")
    top10 = agg.head(10)
    for _, r in top10.iterrows():
        print(
            f"  h={int(r['hidden_dim']):2d} l={int(r['num_hidden_layers'])} "
            f"d={r['dropout']:.1f} pw={r['pos_weight']:.1f} lr={r['learning_rate']:.3f}  "
            f"val_roc_auc={r['val_roc_auc_mean']:.3f}+-{r['val_roc_auc_std']:.3f}  "
            f"val_pr_auc={r['val_pr_auc_mean']:.3f}+-{r['val_pr_auc_std']:.3f}  "
            f"test_roc_auc={r['test_roc_auc_mean']:.3f}+-{r['test_roc_auc_std']:.3f}"
        )

    winner = agg.iloc[0]
    print(
        f"\nWinning config: hidden_dim={int(winner['hidden_dim'])}, "
        f"num_hidden_layers={int(winner['num_hidden_layers'])}, dropout={winner['dropout']}, "
        f"pos_weight={winner['pos_weight']}, learning_rate={winner['learning_rate']}"
    )
    print(
        f"  mean val_roc_auc={winner['val_roc_auc_mean']:.3f} (std={winner['val_roc_auc_std']:.3f} "
        f"across {len(SEEDS)} seeds)"
    )
    print(
        f"  mean test_roc_auc={winner['test_roc_auc_mean']:.3f} (std={winner['test_roc_auc_std']:.3f})"
    )

    # Retrain the winner once more, this time persisting the checkpoint -
    # using SEEDS[0] so the saved weights correspond to one of the 5 seeds
    # actually measured above, not an unseen 6th one.
    print(f"\nRetraining winning config with save_checkpoint=True (seed={SEEDS[0]})...")
    final_result = train_one_config(
        train, val, test,
        hidden_dim=int(winner["hidden_dim"]),
        num_hidden_layers=int(winner["num_hidden_layers"]),
        dropout=float(winner["dropout"]),
        pos_weight=float(winner["pos_weight"]),
        learning_rate=float(winner["learning_rate"]),
        seed=SEEDS[0],
        verbose=True,
        save_checkpoint=True,
        output_dir=output_dir,
    )
    print(f"\nFinal checkpoint: {final_result['best_model_path']}")

    return agg


if __name__ == "__main__":
    run_sweep()
