"""
2026-09-26: c3 counterpart to make_pruner_score_distribution.py - checks WHY the c3
Request Pruner (expert-prefix-label, k=3, threshold=0.5) prunes almost nothing in
practice (0-1% measured live in the SIL logs, see chat 2026-09-26), by checking two
candidate explanations:
  (a) label positivity rate in the c3 prefix-label dataset itself (train+val+test) -
      if k=3 prefixes cover most requests already, most training labels are positive,
      so the MLP learns to mostly predict "keep".
  (b) the deployed checkpoint's raw sigmoid score distribution on val+test - if scores
      are overwhelmingly > 0.5, the model itself (not just force-keep/floor rules) is
      the reason almost nothing gets pruned.
This must run where the c3 datasets/checkpoints actually live (the cluster - not synced
locally as of 2026-09-26).

Run: ./venv/bin/python3 scripts/analysis/check_c3_pruner_near_noop.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from rtv_solver.pipeline.request_pruner_mlp import RequestPrunerMLP

CONFIGS = {
    "bi200/ss100": {
        "dataset_dir": Path("dataset/request_pruning_bi200_ss100_c3_prefix"),
        "ckpt": Path("outputs/request_pruner_mlp_bi200_ss100_c3_prefix/request_pruner_mlp_h16_l2_d0p1_pw2p0_lr0p03/request_pruner_mlp_h16_l2_d0p1_pw2p0_lr0p03_best_val_f3.pt"),
    },
    "bi400/ss200": {
        "dataset_dir": Path("dataset/request_pruning_bi400_ss200_c3_prefix"),
        "ckpt": Path("outputs/request_pruner_mlp_bi400_ss200_c3_prefix/request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03/request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03_best_val_f3.pt"),
    },
    "bi400/ss80": {
        "dataset_dir": Path("dataset/request_pruning_bi400_ss80_c3_prefix"),
        "ckpt": Path("outputs/request_pruner_mlp_bi400_ss80_c3_prefix/request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03/request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03_best_val_f3.pt"),
    },
}
THRESHOLD = 0.5


def label_positivity_rate(dataset_dir: Path):
    n_pos = n_total = 0
    for split in ("train", "val", "test"):
        for p in sorted((dataset_dir / split).glob("*.npz")):
            z = np.load(p)
            labels = z["label"]
            n_pos += int(labels.sum())
            n_total += len(labels)
    return n_pos, n_total


def score_stats(dataset_dir: Path, ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = RequestPrunerMLP(
        feature_dim=len(ck["feature_names"]), hidden_dim=ck["hidden_dim"],
        num_hidden_layers=ck["num_hidden_layers"], dropout=ck["dropout"],
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    manifest = json.load(open(dataset_dir / "manifest.json"))
    feature_index = {n: i for i, n in enumerate(manifest["feature_names"])}
    cols = [feature_index[n] for n in ck["feature_names"]]
    mean = np.array(ck["feature_mean"], dtype=np.float32)
    std = np.array(ck["feature_std"], dtype=np.float32)

    all_scores = []
    for split in ("val", "test"):
        for p in sorted((dataset_dir / split).glob("*.npz")):
            z = np.load(p)
            feats = z["features"][:, cols]
            x = (feats - mean) / std
            with torch.no_grad():
                s = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).numpy()
            all_scores.append(s)
    return np.concatenate(all_scores)


if __name__ == "__main__":
    for cfg, info in CONFIGS.items():
        print(f"=== {cfg} ===")
        n_pos, n_total = label_positivity_rate(info["dataset_dir"])
        print(f"  Label positivity rate (train+val+test): {n_pos}/{n_total} = {100*n_pos/n_total:.2f}%")
        scores = score_stats(info["dataset_dir"], info["ckpt"])
        below_t = np.mean(scores < THRESHOLD) * 100
        print(f"  Score stats (val+test, n={len(scores)}): mean={scores.mean():.3f}, "
              f"median={np.median(scores):.3f}, min={scores.min():.3f}, max={scores.max():.3f}")
        print(f"  Fraction of rows with score < {THRESHOLD}: {below_t:.2f}%")
        print()

        # 2026-09-26: dump raw scores + label positivity so the histogram chart can be
        # built locally (Mac has no c3 dataset/checkpoint copies, only the cluster does).
        out_dir = Path("outputs/c3_pruner_noop_check")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = cfg.replace("/", "_")
        np.savez(out_dir / f"{safe_name}_scores.npz", scores=scores,
                 label_pos=n_pos, label_total=n_total)
        print(f"  saved raw scores -> {out_dir / f'{safe_name}_scores.npz'}")
        print()
