"""
Summarizes and plots the training loss curves for the clean mixed_c2 pos_weight
sweep (pos_weight in {1,2,3,5,8,10}, both GNN architectures, trained once each
via train_models_v2.py --gnn_version {v1,v2}).

Reports, per (architecture, pos_weight):
    - epochs_trained: the exact epoch early stopping fired at (patience=25 epochs
      without validation-F3 improvement), i.e. how many episodes were actually run
    - best_epoch: the epoch whose checkpoint was kept (best_val_f3.pt)
    - train loss trajectory, sampled every 10 epochs (matches how
      train_models_v2.py logs metrics: every 10th epoch, or on improvement)

Usage:
    python3 -m rtv_solver.pipeline.plot_mixed_c2_pw_sweep_loss --run-suffix ""
    python3 -m rtv_solver.pipeline.plot_mixed_c2_pw_sweep_loss --run-suffix "_e200p40"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# exact "Early stopping at epoch X; best epoch was Y" lines from the training run
# console logs - not reconstructable purely from the every-10-epoch CSV log, since
# early stopping can fire between two logged points. Keyed by (gnn_version,
# pos_weight, run_suffix) so both the original patience=25 sweep and the
# patience=40/epochs=200 re-run ("_e200p40") can be plotted with the same script.
STOPPING_INFO = {
    ("v1", 1.0, ""): {"stopped_epoch": 25, "best_epoch": 0},
    ("v1", 2.0, ""): {"stopped_epoch": 52, "best_epoch": 27},
    ("v1", 3.0, ""): {"stopped_epoch": 27, "best_epoch": 2},
    ("v1", 5.0, ""): {"stopped_epoch": 79, "best_epoch": 54},
    ("v1", 8.0, ""): {"stopped_epoch": 78, "best_epoch": 53},
    ("v1", 10.0, ""): {"stopped_epoch": 94, "best_epoch": 69},
    ("v2", 1.0, ""): {"stopped_epoch": 32, "best_epoch": 7},
    ("v2", 2.0, ""): {"stopped_epoch": 25, "best_epoch": 0},
    ("v2", 3.0, ""): {"stopped_epoch": 33, "best_epoch": 8},
    ("v2", 5.0, ""): {"stopped_epoch": 93, "best_epoch": 68},
    ("v2", 8.0, ""): {"stopped_epoch": 69, "best_epoch": 44},
    ("v2", 10.0, ""): {"stopped_epoch": 88, "best_epoch": 63},
    ("v1", 1.0, "_e200p40"): {"stopped_epoch": 40, "best_epoch": 0},
    ("v1", 2.0, "_e200p40"): {"stopped_epoch": 83, "best_epoch": 43},
    ("v1", 3.0, "_e200p40"): {"stopped_epoch": 42, "best_epoch": 2},
    ("v1", 5.0, "_e200p40"): {"stopped_epoch": 101, "best_epoch": 61},
    ("v1", 8.0, "_e200p40"): {"stopped_epoch": 72, "best_epoch": 32},
    ("v1", 10.0, "_e200p40"): {"stopped_epoch": 95, "best_epoch": 55},
    ("v2", 1.0, "_e200p40"): {"stopped_epoch": 47, "best_epoch": 7},
    ("v2", 2.0, "_e200p40"): {"stopped_epoch": 40, "best_epoch": 0},
    ("v2", 3.0, "_e200p40"): {"stopped_epoch": 107, "best_epoch": 67},
    ("v2", 5.0, "_e200p40"): {"stopped_epoch": 144, "best_epoch": 104},
    ("v2", 8.0, "_e200p40"): {"stopped_epoch": 122, "best_epoch": 82},
    ("v2", 10.0, "_e200p40"): {"stopped_epoch": 78, "best_epoch": 38},
}


def load_loss_curve(csv_path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df.run_name.str.contains("mixed_c2") & df.run_name.str.endswith(suffix)]
    df = df[(df.epoch >= 0) & (df.split == "train")]
    # loss is logged once per epoch but repeated across every (instance, threshold)
    # row for that epoch - collapse back to one loss value per (run_name, epoch).
    return df.drop_duplicates(["run_name", "epoch"])[["run_name", "pos_weight", "epoch", "loss"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-suffix", default="", help='"" for the original patience=25 sweep, "_e200p40" for the redone one')
    args = parser.parse_args()
    suffix = args.run_suffix

    tag = suffix.lstrip("_") or "original"
    out_dir = ROOT / "outputs" / "models_v2_gnnv2" / "pw_sweep_analysis" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    v1 = load_loss_curve(ROOT / "outputs/models_v2/training_metrics_all.csv", f"_v1{suffix}")
    v2 = load_loss_curve(ROOT / "outputs/models_v2_gnnv2/training_metrics_all.csv", f"_v2{suffix}")
    v1["gnn_version"] = "v1"
    v2["gnn_version"] = "v2"
    combined = pd.concat([v1, v2], ignore_index=True)

    # --- summary table ---
    rows = []
    for (version, pw, s), info in STOPPING_INFO.items():
        if s != suffix:
            continue
        curve = combined[(combined.gnn_version == version) & (combined.pos_weight == pw)].sort_values("epoch")
        rows.append({
            "gnn_version": version,
            "pos_weight": pw,
            "epochs_trained": info["stopped_epoch"] + 1,  # epoch is 0-indexed
            "best_epoch": info["best_epoch"],
            "initial_train_loss": curve["loss"].iloc[0] if len(curve) else None,
            "final_train_loss": curve["loss"].iloc[-1] if len(curve) else None,
        })
    summary = pd.DataFrame(rows).sort_values(["gnn_version", "pos_weight"])
    summary_path = out_dir / "pw_sweep_epochs_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")

    combined.to_csv(out_dir / "pw_sweep_loss_curves_long.csv", index=False)

    # --- plot: loss vs epoch, one line per pos_weight, one subplot per architecture ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    pos_weights = sorted(combined["pos_weight"].unique())
    colors = plt.cm.viridis([i / (len(pos_weights) - 1) for i in range(len(pos_weights))])

    for ax, version in zip(axes, ["v1", "v2"]):
        sub = combined[combined.gnn_version == version]
        for pw, color in zip(pos_weights, colors):
            curve = sub[sub.pos_weight == pw].sort_values("epoch")
            if curve.empty:
                continue
            ax.plot(curve["epoch"], curve["loss"], marker="o", markersize=3, color=color, label=f"pw={pw:g}")
            best_epoch = STOPPING_INFO[(version, pw, suffix)]["best_epoch"]
            best_row = curve[curve["epoch"] <= best_epoch]
            if not best_row.empty:
                ax.scatter([best_epoch], [best_row["loss"].iloc[-1]], color=color, s=60, zorder=5,
                           edgecolor="black", linewidth=0.5)
        ax.set_title(f"GNN {version} - training loss (mixed_c2, sampled every 10 epochs)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCEWithLogitsLoss (train)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="pos_weight")

    fig.suptitle(f"Mixed_c2 pos_weight sweep ({tag}) - training loss curves\n(dots with black outline = checkpoint kept / best val F3 epoch)")
    fig.tight_layout()
    plot_path = out_dir / "pw_sweep_loss_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
