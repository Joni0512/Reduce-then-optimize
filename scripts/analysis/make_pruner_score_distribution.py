"""
2026-09-25: Plots the Request Pruner MLP's raw sigmoid score distribution (val+test split)
for all 3 window configs, to visually check how well-calibrated threshold=0.5 is - if a
large mass of scores sits close to 0.5, the threshold could plausibly be tuned; if scores
are cleanly bimodal (most near 0 or 1), 0.5 is already a good cut. User question, 2026-09-25.

Uses each config's FINAL deployed expert-prefix-label pruner checkpoint (class1, cardinality
2 - the same ones used in run_sil_prefix_pruner.sh). Same normalization as training
(feature_mean/feature_std stored in the checkpoint).

Run: python3 -m scripts.analysis.make_pruner_score_distribution
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
THRESHOLD = 0.5
COLORS = ["#276575", "#A6493A", "#3E8F6C"]


def load_scores(dataset_dir: Path, ckpt_path: Path) -> np.ndarray:
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, (cfg, info), color in zip(axes, CONFIGS.items(), COLORS):
        scores = load_scores(info["dataset_dir"], info["ckpt"])
        mean_s = float(np.mean(scores))
        median_s = float(np.median(scores))
        ax.hist(scores, bins=40, range=(0, 1), color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"threshold={THRESHOLD}")
        ax.axvline(mean_s, color="#E8A33D", linestyle="-", linewidth=2, label=f"mean={mean_s:.3f}")
        ax.axvline(median_s, color="#5A5A5A", linestyle=":", linewidth=2, label=f"median={median_s:.3f}")
        near_mid = np.mean((scores > 0.3) & (scores < 0.7)) * 100
        ax.set_title(f"{cfg}\n{near_mid:.1f}% of scores in [0.3, 0.7]", fontsize=11.5)
        ax.set_xlabel("Sigmoid score", fontsize=10.5)
        ax.legend(fontsize=8.5, frameon=False, loc="upper center")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Count (val+test rows)", fontsize=10.5)
    fig.suptitle("Request Pruner MLP score distribution (expert-prefix-label, mixed val+test split, cardinality 2)", fontsize=13)
    fig.tight_layout()

    out = "figures_export/Request Pruner SIL/pruner_score_distribution_all_configs"
    fig.savefig(f"{out}.png", dpi=200)
    fig.savefig(f"{out}.pdf")
    print(f"saved {out}.png / .pdf")
