"""
2026-09-23: Offline check of the TRAINED EXPERT-PREFIX-LABEL pruner's own accuracy per
config - candidate explanation for why bi200/ss100 recovers much less SIL damage (61%
class-1, 8% mixed) than bi400/ss80 (132%, beats baseline) and bi400/ss200 (146%, beats
baseline), after the window-count/divergence hypothesis was tested and refuted (see
analyze_prefix_label_batch_divergence.py).

Earlier analyses (analyze_request_pruner_false_prunes.py) measured the OLD RHO-label
pruner's accuracy against the RHO baseline. This script instead measures each FINAL,
already-deployed prefix-label pruner checkpoint against what actually matters for SIL:
in window 0 (state identical to a real SIL run, by construction - no assignment has
happened yet), does it keep each vehicle's TRUE first expert-route request (the one
legacy imitation scoring needs)? If bi200/ss100's own trained pruner drops its true
target more often than bi400/ss80's or bi400/ss200's, that's evidence the calibration
(not just the label formula) differs by config; if the miss-rates are similar across
configs, the cause lies elsewhere (e.g. SIL training dynamics unrelated to the pruner).

Run: python -m rtv_solver.pipeline.analyze_prefix_pruner_own_accuracy
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rtv_solver.pipeline import analyze_request_pruner_false_prunes as A
from rtv_solver.pipeline.build_request_pruner_prefix_dataset import K, expert_by_vehicle, prefix_labels

CONFIGS = {
    "bi200_ss100": {
        "dataset_dir": Path("dataset/request_pruning_bi200_ss100"),
        "ckpt": Path("outputs/request_pruner_mlp_bi200_ss100_prefix/pw1.0/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"),
    },
    "bi400_ss80": {
        "dataset_dir": Path("dataset/request_pruning_bi400_ss80"),
        "ckpt": Path("outputs/request_pruner_mlp_bi400_ss80_prefix_sweep/request_pruner_mlp_h64_l1_d0p0_pw1p5_lr0p03/request_pruner_mlp_h64_l1_d0p0_pw1p5_lr0p03_best_val_f3.pt"),
    },
    "bi400_ss200": {
        "dataset_dir": Path("dataset/request_pruning_bi400_ss200"),
        "ckpt": Path("outputs/request_pruner_mlp_bi400_ss200_prefix/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"),
    },
}
THRESHOLD = 0.5
MIN_RETENTION = 0.3


def main():
    rows = []
    for cfg, info in CONFIGS.items():
        A.DATASET_DIR = info["dataset_dir"]
        A.CKPT = info["ckpt"]
        A.THRESHOLD = THRESHOLD
        manifest = __import__("json").load(open(A.DATASET_DIR / "manifest.json"))
        feature_index = {n: i for i, n in enumerate(manifest["feature_names"])}
        model, ck = A.load_model()

        lost_first = lost_any_k = total_veh = 0
        cand_total = pruned_total = 0
        for stem in A.SIL_VAL:
            df = A.replay_instance(stem, model, ck, feature_index)
            pruned, _ = A.apply_pruner(df, MIN_RETENTION)
            cand = ~(df["active"] | df["urgent"])
            cand_total += int(cand.sum()); pruned_total += int(pruned.sum())

            w0 = df[df.window == 0].reset_index(drop=True)
            pruned_w0, _ = A.apply_pruner(w0, MIN_RETENTION)
            pr = dict(zip(w0.request, pruned_w0))
            batch = set(w0.request)
            for seq in expert_by_vehicle(stem).values():
                proj = [r for r in seq if r in batch]
                if not proj:
                    continue
                total_veh += 1
                if pr[proj[0]]:
                    lost_first += 1
                if any(pr[r] for r in proj[:K]):
                    lost_any_k += 1

        rows.append({
            "config": cfg, "vehicles_w0": total_veh,
            "lost_1st_target_%": 100 * lost_first / total_veh,
            "lost_any_of_K_%": 100 * lost_any_k / total_veh,
            "pruned_share_all_windows_%": 100 * pruned_total / cand_total,
        })

    df = pd.DataFrame(rows)
    print("Own accuracy of each FINAL prefix-label pruner, threshold=0.5, min_retention=0.3, "
          "class-1 val instances, window 0 (exact state):\n")
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(df.to_string(index=False))
    df.to_csv("outputs/prefix_pruner_own_accuracy_by_config.csv", index=False)


if __name__ == "__main__":
    main()
