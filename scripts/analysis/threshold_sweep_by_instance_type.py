"""
2026-09-25: Threshold sweep for the Request Pruner MLP, split by instance geometry (lc vs.
lr/lrc), to check whether a different threshold than the current default 0.5 would help -
motivated by the score-distribution finding (make_pruner_score_distribution.py) and the
pruning-sensitivity finding (lc tolerates pruning, lr/lrc is fragile to it, 2026-09-24/25).

For each config and threshold in THRESHOLDS, computes precision/recall/F3 against the true
expert-prefix label, separately for lc-instance rows and lr/lrc-instance rows (val+test
split). Ground truth label and instance name come from the dataset npz files themselves
(same source as make_pruner_score_distribution.py's scores).

Run: python3 -m scripts.analysis.threshold_sweep_by_instance_type
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtv_solver.pipeline.request_pruner_mlp import RequestPrunerMLP

CONFIGS = {
    "bi200/ss100": {
        "dataset_dir": Path("dataset/request_pruning_bi200_ss100_prefix"),
        "ckpt": Path("outputs/request_pruner_mlp_bi200_ss100_prefix/pw1.0/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"),
    },
    "bi400/ss80": {
        "dataset_dir": Path("dataset/request_pruning_bi400_ss80_prefix"),
        "ckpt": Path("outputs/request_pruner_mlp_bi400_ss80_prefix_sweep/request_pruner_mlp_h64_l1_d0p0_pw1p5_lr0p03/request_pruner_mlp_h64_l1_d0p0_pw1p5_lr0p03_best_val_f3.pt"),
    },
    "bi400/ss200": {
        "dataset_dir": Path("dataset/request_pruning_bi400_ss200_prefix"),
        "ckpt": Path("outputs/request_pruner_mlp_bi400_ss200_prefix/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"),
    },
}
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def load_scores_labels_types(dataset_dir: Path, ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = RequestPrunerMLP(
        feature_dim=len(ck["feature_names"]), hidden_dim=ck["hidden_dim"],
        num_hidden_layers=ck["num_hidden_layers"], dropout=ck["dropout"],
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    import json
    manifest = json.load(open(dataset_dir / "manifest.json"))
    feature_index = {n: i for i, n in enumerate(manifest["feature_names"])}
    cols = [feature_index[n] for n in ck["feature_names"]]
    mean = np.array(ck["feature_mean"], dtype=np.float32)
    std = np.array(ck["feature_std"], dtype=np.float32)

    scores, labels, itypes = [], [], []
    for split in ("val", "test"):
        for p in sorted((dataset_dir / split).glob("*.npz")):
            stem = p.stem
            itype = "lc" if stem.startswith("lc") else "lr/lrc"
            z = np.load(p)
            feats = z["features"][:, cols]
            x = (feats - mean) / std
            with torch.no_grad():
                s = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).numpy()
            scores.append(s)
            labels.append(z["label"])
            itypes.extend([itype] * len(s))
    return np.concatenate(scores), np.concatenate(labels), np.array(itypes)


def prf3(scores, labels, threshold):
    pred = (scores >= threshold).astype(int)
    labels = labels.astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta2 = 9.0
    f3 = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (precision + recall) else 0.0
    return precision, recall, f3, tp + fp, len(labels)


if __name__ == "__main__":
    rows = []
    for cfg, info in CONFIGS.items():
        scores, labels, itypes = load_scores_labels_types(info["dataset_dir"], info["ckpt"])
        for group in ("lc", "lr/lrc"):
            mask = itypes == group
            for t in THRESHOLDS:
                p, r, f3, kept, total = prf3(scores[mask], labels[mask], t)
                rows.append({
                    "config": cfg, "group": group, "threshold": t,
                    "precision": p, "recall": r, "f3": f3,
                    "kept": kept, "total": total, "kept_pct": 100 * kept / total,
                })
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_rows", None, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    df.to_csv("outputs/threshold_sweep_by_instance_type.csv", index=False)
    print("\nsaved outputs/threshold_sweep_by_instance_type.csv")
