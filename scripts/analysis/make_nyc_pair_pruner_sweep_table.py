"""
2026-11-03: LaTeX table of the NYC pair-pruner hyperparameter sweep trained on
RHO batch_interval=3600/step_size=1800 expert labels (27 configs, seed 1).

Source: outputs/pair_pruner_sweep_rho3060/*.json, written by
rtv_solver/pipeline/sweep_nyc_pair_pruner_rho3060.py (SLURM array job, 27 tasks).
Rows sorted by val ROC-AUC (the selection criterion; F3@0.5 is confounded by
pos_weight).
"""
import glob
import json
from pathlib import Path

SRC = "outputs/pair_pruner_sweep_rho3060/*.json"
OUT = Path("figures_export/NYC/nyc_pair_pruner_rho3060_sweep_table.tex")

rows = []
for f in sorted(glob.glob(SRC)):
    d = json.load(open(f))
    h, b, r = d["hyperparameters"], d["best"], d["runtime_s"]
    rows.append({
        "layers": h["message_passing_steps"], "pw": h["pos_weight"], "lr": h["learning_rate"],
        "epoch": b["epoch"], "auc": b["val_auc"], "rec50": b["val_recall_at_50pct_pruned"],
        "f3": b["val_f3_at_0.5"], "kept": b["val_kept_edges_at_0.5"], "train_s": r["training"],
    })
assert len(rows) == 27, len(rows)
rows.sort(key=lambda x: -x["auc"])


def lr_tex(lr):
    return {0.01: r"$10^{-2}$", 0.003: r"$3\cdot10^{-3}$", 0.001: r"$10^{-3}$"}[lr]


lines = [
    r"\begin{table}[htbp]",
    r"    \centering",
    r"    \small",
    r"    \caption{NYC pair-pruner hyperparameter sweep (edge-classifier GNN, RHO labels with "
    r"batch\_interval=3600/step\_size=1800, cardinality 3). Label: two requests are served by the same "
    r"vehicle in the RHO solution. Train day 2016-01-12 (499{,}500 edges, 1.83\% positive), validation day "
    r"2016-01-13 (547{,}581 edges, 1.75\% positive); test day not used. Fixed: hidden\_dim=64, 100 epochs, "
    r"balanced 50/50 edge sampling per epoch, seed 1, evaluation every 5 epochs. Rows sorted by validation "
    r"ROC-AUC; ``Recall@50\%'' is the recall of the top 50\% scored validation edges (50\% pruned); F3, "
    r"kept edges: threshold 0.5. Best epoch = epoch of best validation AUC.}",
    r"    \label{tab:nyc-pair-pruner-rho3060-sweep}",
    r"    \begin{tabular}{ccccccccr}",
    r"        \toprule",
    r"        Layers & pos\_weight & lr & Best epoch & Val AUC & Recall@50\% & F3@0.5 & Kept edges & Train time \\",
    r"        \midrule",
]
for x in rows:
    lines.append(
        f"        {x['layers']} & {x['pw']:g} & {lr_tex(x['lr'])} & {x['epoch']} & {x['auc']:.4f} & "
        f"{x['rec50']:.4f} & {x['f3']:.4f} & {x['kept']:,}".replace(",", "{,}")
        + f" & {x['train_s']:.0f}\\,s \\\\"
    )
lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines))
print("saved:", OUT, "| rows:", len(rows))
