"""
2026-10-20: selection of a window-specific Request Pruner by VEHICLE-level loss (rule A), instead of F3 / ROC-AUC.
Rule A: among all candidates (checkpoint x threshold) whose pruning rate is >= MIN_PRUNE (default 30 %), take the one with the
smallest "lost 1st target" rate (share of vehicles whose TRUE first reference-route request is pruned in window 0).
Tie-break (changed 2026-10-20 after the first run): smaller "lost any of the first K", then higher pruning rate - with the pruning
rate first, bi400/ss80 picked t=0.5 (lost_1st is flat 1.9 % = one vehicle for t=0.1..0.5), which the SIL runs show as damaging. Each config is evaluated only with its OWN window-specific
pruners and its own dataset (pruners are always trained per batch_interval/step_size).

Data: the 8 pruner-VAL instances of dataset/request_pruning_*_prefix/manifest.json (lc107 lc206 lr109 lr110 lr208 lr209 lrc106 lrc206),
NOT the 12 test instances (= SIL validation) - selecting on the evaluation instances would leak. 53 vehicles in total (42 class 1,
11 class 2), i.e. one vehicle ~1.9 pp; pruning rate pooled over all windows (replay of the RHO-baseline windows, approximate from window 1).
Replay uses the RHO-label dataset (as analyze_prefix_pruner_own_accuracy.py) so that the "active" proxy (label 1 in an earlier window)
means "already assigned"; the prefix labels enter only through the expert route projection (expert_by_vehicle, K = max cardinality).
Local candidates: only the few checkpoints in outputs/request_pruner_mlp_*prefix* (the full 432-config sweep is not local).

Run: ./venv/bin/python3 scripts/analysis/select_request_pruner_by_vehicle_loss.py [--min_prune 0.30] [--min_retention 0.3]
Output: outputs/request_pruner_selection_by_vehicle_loss.csv, figures_export/Request Pruner SIL/Cardinality 2/c2_request_pruner_selection_pareto.png/.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rtv_solver.pipeline import analyze_request_pruner_false_prunes as A
from rtv_solver.pipeline.build_request_pruner_prefix_dataset import expert_by_vehicle

K = 2  # c2: = max cardinality (the c3 datasets are only on the cluster)
CONFIGS = {
    "bi200_ss100": dict(rho_dataset=Path("dataset/request_pruning_bi200_ss100"), prefix_dataset=Path("dataset/request_pruning_bi200_ss100_prefix"),
                        step=100.0, ckpt_dirs=[Path("outputs/request_pruner_mlp_bi200_ss100_prefix")]),
    "bi400_ss200": dict(rho_dataset=Path("dataset/request_pruning_bi400_ss200"), prefix_dataset=Path("dataset/request_pruning_bi400_ss200_prefix"),
                        step=200.0, ckpt_dirs=[Path("outputs/request_pruner_mlp_bi400_ss200_prefix")]),
    "bi400_ss80": dict(rho_dataset=Path("dataset/request_pruning_bi400_ss80"), prefix_dataset=Path("dataset/request_pruning_bi400_ss80_prefix"),
                       step=80.0, ckpt_dirs=[Path("outputs/request_pruner_mlp_bi400_ss80_prefix"), Path("outputs/request_pruner_mlp_bi400_ss80_prefix_sweep")]),
}
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def val_instances(prefix_dataset):
    m = json.load(open(prefix_dataset / "manifest.json"))
    return sorted(k for k, v in m["instances"].items() if v["split"] == "val")


def evaluate(cfg, info, min_retention):
    A.DATASET_DIR, A.URGENT_SLACK = info["rho_dataset"], info["step"]
    manifest = json.load(open(A.DATASET_DIR / "manifest.json"))
    feature_index = {n: i for i, n in enumerate(manifest["feature_names"])}
    stems = val_instances(info["prefix_dataset"])
    rows = []
    for d in info["ckpt_dirs"]:
        for ck_path in sorted(d.rglob("*_best_val_f3.pt")):
            A.CKPT = ck_path
            model, ck = A.load_model()
            dfs = {s: A.replay_instance(s, model, ck, feature_index) for s in stems}  # scores do not depend on the threshold
            for t in THRESHOLDS:
                A.THRESHOLD = t
                cand = pruned = veh = lost1 = lostK = 0
                for s, df in dfs.items():
                    pr, _ = A.apply_pruner(df, min_retention)
                    c = ~(df["active"] | df["urgent"])
                    cand += int(c.sum()); pruned += int(pr.sum())
                    w0 = df[df.window == 0].reset_index(drop=True)
                    pw0, _ = A.apply_pruner(w0, min_retention)
                    flag = dict(zip(w0.request, pw0))
                    batch = set(w0.request)
                    for seq in expert_by_vehicle(s).values():
                        proj = [r for r in seq if r in batch]
                        if proj:
                            veh += 1; lost1 += int(flag[proj[0]]); lostK += int(any(flag[r] for r in proj[:K]))
                rows.append(dict(config=cfg, checkpoint=ck_path.parent.name.replace("request_pruner_mlp_", ""), threshold=t, vehicles=veh,
                                 prune_rate=100 * pruned / max(cand, 1), lost_1st=100 * lost1 / max(veh, 1), lost_anyK=100 * lostK / max(veh, 1)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_prune", type=float, default=0.30)
    ap.add_argument("--min_retention", type=float, default=0.3)
    a = ap.parse_args()
    out = pd.concat([evaluate(c, i, a.min_retention) for c, i in CONFIGS.items()], ignore_index=True)
    Path("outputs").mkdir(exist_ok=True)
    out.to_csv("outputs/request_pruner_selection_by_vehicle_loss.csv", index=False)
    print(f"Rule A: pruning rate >= {100 * a.min_prune:.0f} %, then smallest lost_1st (tie: smaller lost_anyK, then higher pruning rate); min_retention {a.min_retention}\n")
    for cfg in CONFIGS:
        sub = out[out.config == cfg]
        ok = sub[sub.prune_rate >= 100 * a.min_prune].sort_values(["lost_1st", "lost_anyK", "prune_rate"], ascending=[True, True, False])
        print(f"== {cfg}: {len(sub)} candidates x thresholds, {int(sub.vehicles.iloc[0])} vehicles in window 0; {len(ok)} reach the minimum pruning rate")
        with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
            print(ok.head(5).to_string(index=False) if len(ok) else "  none reaches the minimum pruning rate")
        print()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    for ax, cfg in zip(axes, CONFIGS):
        sub = out[out.config == cfg]
        for name, g in sub.groupby("checkpoint"):
            ax.plot(g.prune_rate, g.lost_1st, "-o", markersize=4, label=name, alpha=0.85)
        ax.axvline(100 * a.min_prune, color="#A6493A", linestyle="--", linewidth=1)
        ax.set_title({"bi200_ss100": "batch_interval=200/step_size=100", "bi400_ss200": "batch_interval=400/step_size=200",
                      "bi400_ss80": "batch_interval=400/step_size=80"}[cfg], fontsize=11)
        ax.set_xlabel("Pruning rate (%, all windows)")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Vehicles losing their true 1st target (%), window 0")
    fig.suptitle("Request Pruner (c2, window-specific, 8 pruner-val instances): vehicle loss vs. pruning rate, thresholds 0.1 ... 0.9 per checkpoint", fontsize=11.5)
    fig.text(0.5, 0.005, "53 vehicles in window 0 (one vehicle = 1.9 pp); dashed = minimum pruning rate of rule A; each point = one threshold; "
             f"min_retention {a.min_retention}", ha="center", fontsize=7.6, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out_dir = Path("figures_export/Request Pruner SIL/Cardinality 2")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"c2_request_pruner_selection_pareto.{ext}", dpi=200)


if __name__ == "__main__":
    main()
