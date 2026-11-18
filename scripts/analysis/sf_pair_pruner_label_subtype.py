"""
2026-11-17: SF pair pruner (uniform road-node starts, batch_interval=1800/step_size=900), AUC and recall per label variant,
split by positive subtype (user: "Tabelle und Balkendiagramm mit AUC und Recall pro Label, alle Thresholds").
Read-only evaluation on VAL = instance seed 2 (never the test instance seed 42).
  * models: L1 / pos_weight 5 / lr 0.003 / hidden 64 / 100 epochs, seeds 1-3, trained with --label_mode overlap or vehicle
    (rtv_solver/pipeline/sweep_sf_pair_pruner_window_bi1800_ss900_uniformroad.py)
  * positive subtypes in the window graph: "time overlap" = both on board at the same time; "sequential only" = same
    vehicle, never on board together (vehicle label = both subtypes)
  * AUC of each subtype vs the negatives (pairs never in the same vehicle); recall + kept share at thresholds 0.5-0.9
Output: figures_export/SF/sf_pair_pruner_label_subtype_{auc_recall.png,.pdf,table.csv}
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import rtv_solver.pipeline.sweep_sf_pair_pruner_window_bi1800_ss900_uniformroad as m
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.pipeline.train_nyc_pair_pruner_gnn_rho3060 import HIDDEN_DIM

ROOT = Path(__file__).resolve().parents[2]
CKPT = "outputs/pair_pruner_window_sf_uniformroad_bi1800_ss900/L1_pw5_lr0.003_seed{s}_window_sf_uniformroad_bi1800_ss900_lbl{l}_best_val_auc.pt"
OUT = ROOT / "figures_export" / "SF"
THRS = (0.5, 0.6, 0.7, 0.8, 0.9)
SEEDS = (1, 2, 3)
LABELS = ("overlap", "vehicle")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"overlap": "#2a78d6", "vehicle": "#d97a2b"}
NAME = {"overlap": "trained on overlap label", "vehicle": "trained on vehicle label"}


def auc(pos, neg):
    s = np.r_[pos, neg]
    r = np.argsort(np.argsort(s)) + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    b = m.base
    p = m.solution_path("val")
    b.RequestGraphLabelBuilder = m.RequestGraphLabelBuilder
    o = b.build_window_dataset(p)
    b.RequestGraphLabelBuilder = m._VehicleLabelBuilder
    v = b.build_window_dataset(p)
    lo, lv = o["labels"].numpy().astype(bool), v["labels"].numpy().astype(bool)
    seq, neg = lv & ~lo, ~lv
    print(f"val edges {len(lo)} | time overlap {lo.sum()} | sequential only {seq.sum()} | negatives {neg.sum()}")

    rows = []
    for lbl in LABELS:
        for s in SEEDS:
            mdl = RequestGraphEdgeGNN(node_feature_dim=o["x"].shape[1], edge_feature_dim=o["edge_attr"].shape[1],
                                      hidden_dim=HIDDEN_DIM, message_passing_steps=1)
            mdl.load_state_dict(torch.load(ROOT / CKPT.format(s=s, l=lbl)))
            mdl.eval()
            with torch.no_grad():
                pr = torch.sigmoid(b.forward(mdl, o)).numpy()
            own = lo if lbl == "overlap" else lv
            a = {"auc_own_label": auc(pr[own], pr[~own]), "auc_time_overlap_vs_neg": auc(pr[lo], pr[neg]),
                 "auc_sequential_vs_neg": auc(pr[seq], pr[neg])}
            for t in THRS:
                k = pr >= t
                rows.append({"label_mode": lbl, "seed": s, "threshold": t, **{x: round(y, 4) for x, y in a.items()},
                             "kept_share": round(k.mean(), 4), "recall_own_label": round(k[own].mean(), 4),
                             "recall_time_overlap": round(k[lo].mean(), 4), "recall_sequential": round(k[seq].mean(), 4),
                             "n_time_overlap": int(lo.sum()), "n_sequential": int(seq.sum()), "n_edges": len(lo),
                             "checkpoint": CKPT.format(s=s, l=lbl)})

    with open(OUT / "sf_pair_pruner_label_subtype_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def agg(lbl, key, t=None):
        v_ = np.array([r[key] for r in rows if r["label_mode"] == lbl and (t is None or r["threshold"] == t)
                       and (t is not None or r["threshold"] == THRS[0])])
        return v_

    # summary (mean over seeds 1-3) for the console
    for lbl in LABELS:
        print(f"{lbl}: AUC own {agg(lbl,'auc_own_label').mean():.3f} | time-overlap {agg(lbl,'auc_time_overlap_vs_neg').mean():.3f}"
              f" | sequential {agg(lbl,'auc_sequential_vs_neg').mean():.3f}")
        for t in THRS:
            print(f"   t={t}: kept {agg(lbl,'kept_share',t).mean():.3f} recall own {agg(lbl,'recall_own_label',t).mean():.3f}"
                  f" time-overlap {agg(lbl,'recall_time_overlap',t).mean():.3f} sequential {agg(lbl,'recall_sequential',t).mean():.3f}")

    panels = [("AUC", None), ("Recall: time-overlap positives", "recall_time_overlap"),
              ("Recall: sequential-only positives", "recall_sequential"), ("Kept share of all pairs", "kept_share")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))
    w = 0.38
    for ax, (title, key) in zip(axes, panels):
        if key is None:
            groups = [("own label", "auc_own_label"), ("time overlap\nvs negatives", "auc_time_overlap_vs_neg"),
                      ("sequential only\nvs negatives", "auc_sequential_vs_neg")]
            xl = [g for g, _ in groups]
            vals = {lbl: [agg(lbl, k) for _, k in groups] for lbl in LABELS}
            fmt, lo_y = "{:.3f}", 0.8
        else:
            xl = [f"t={t}" for t in THRS]
            vals = {lbl: [agg(lbl, key, t) for t in THRS] for lbl in LABELS}
            fmt, lo_y = ("{:.1%}" if key == "kept_share" else "{:.2f}"), 0.0
        for li, lbl in enumerate(LABELS):
            for gi, v_ in enumerate(vals[lbl]):
                x = gi + (li - 0.5) * w
                ax.bar(x, v_.mean(), width=w * 0.92, color=COL[lbl], edgecolor="white", linewidth=1.5,
                       label=NAME[lbl] if gi == 0 else None)
                ax.vlines(x, v_.min(), v_.max(), color=INK, linewidth=1)
                ax.scatter(np.full(len(v_), x), v_, s=8, color=INK, zorder=3)
                ax.text(x, v_.max(), fmt.format(v_.mean()), ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(xl)), xl, fontsize=8)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        if key != "kept_share":
            ax.set_ylim(lo_y, 1.04)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.07), ncol=2)
    fig.suptitle("SF pair pruner (batch_interval=1800/step_size=900): AUC and recall per label variant, split by positive subtype",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.text(0.01, 0.01,
             f"Val = SF instance seed 2 window graphs ({len(lo)} pairs: {lo.sum()} time-overlap, {seq.sum()} sequential-only, "
             f"{neg.sum()} negatives), labels from RHO 150 vehicles, uniform road-node starts (start seed 42), max_cardinality=2, "
             "no time-rule filter. Models: RequestGraphEdgeGNN, 1 layer, hidden 64, pos_weight 5, lr 0.003, 100 epochs, balanced, "
             "geographic features.\nBar = mean over training seeds 1-3, whisker = min-max, dots = single seeds. AUC is threshold-free. "
             "Checkpoints: outputs/pair_pruner_window_sf_uniformroad_bi1800_ss900/L1_pw5_lr0.003_seed*_..._lbl{overlap,vehicle}"
             "_best_val_auc.pt. Script: scripts/analysis/sf_pair_pruner_label_subtype.py",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"sf_pair_pruner_label_subtype_auc_recall.{ext}", dpi=200)
    (ROOT / "outputs/pair_pruner_window_sf_uniformroad_bi1800_ss900/subtype_recall_val.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
