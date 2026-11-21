"""
2026-11-18: effect of the pair-pruner GNN hyperparameters on SF (user: "Diagramm und CSV, um zu sehen, was die
Hyperparameter ausmachen"). Input: the 72-run grids per window config, outputs/pair_pruner_window_sf_uniformroad_bi{BI}_ss{SS}/
L*_lbl*.json (sweep_sf_pair_pruner_window_uniformroad.py; train instance seed 1, val instance seed 2, hidden_dim 64 fixed,
100 epochs, balanced sampling, best epoch by val ROC-AUC). Grid: message-passing layers {1,2,3} x pos_weight {1,5} x
lr {0.003,0.01} x seeds {1,2,3}, labels overlap / vehicle. 3600/1800 added 2026-11-18 after its grid finished.
Per config x label x factor level: marginal mean of val AUC over the other factors and seeds (n=12 for layers, 18 for pw/lr),
shown as DIFFERENCE to the config x label grand mean (bars from 0, no truncated AUC axis); error bar = std over those runs.
Second figure: pos_weight vs kept share / recall at threshold 0.5 (val).
Output: figures_export/SF/sf_pair_pruner_hparam_effects.{png,pdf,csv}, sf_pair_pruner_pw_operating_point.{png,pdf}
"""
import collections, csv, glob, json, statistics as st
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_export" / "SF"
CFGS = [("300", "150"), ("600", "300"), ("900", "450"), ("1800", "900"), ("3600", "1800")]  # 2026-11-18: 3600/1800 grid finished
R = []
for bi, ss in CFGS:
    for f in glob.glob(str(ROOT / f"outputs/pair_pruner_window_sf_uniformroad_bi{bi}_ss{ss}/L*_lbl*.json")):
        d = json.load(open(f)); h, b = d["hyperparameters"], d["best"]
        R.append(dict(cfg=f"{bi}/{ss}", label=h["label_mode"], layers=h["message_passing_steps"], pos_weight=h["pos_weight"],
                      lr=h["learning_rate"], seed=h["seed"], auc=b["val_auc"], kept05=b["val_kept_share_at_0.5"], recall05=b["val_recall_at_0.5"]))
assert len(R) == 360, len(R)
GROUPS = [(f"{bi}/{ss}", l) for bi, ss in CFGS for l in ("overlap", "vehicle")]
FACTORS = [("layers", [1, 2, 3], "message-passing layers"), ("pos_weight", [1.0, 5.0], "pos_weight"), ("lr", [0.003, 0.01], "learning rate")]
rows = []
for c, l in GROUPS:
    sel = [r for r in R if r["cfg"] == c and r["label"] == l]
    gm = st.mean(r["auc"] for r in sel)
    for fac, levels, _ in FACTORS:
        for lv in levels:
            v = [r for r in sel if r[fac] == lv]
            rows.append(dict(config=f"batch_interval={c.split('/')[0]}/step_size={c.split('/')[1]}", label=l, factor=fac, level=lv, n=len(v),
                             auc_mean=st.mean(r["auc"] for r in v), auc_std=st.pstdev([r["auc"] for r in v]),
                             delta_auc_vs_group_mean=st.mean(r["auc"] for r in v) - gm,
                             kept_at_0p5=st.mean(r["kept05"] for r in v), recall_at_0p5=st.mean(r["recall05"] for r in v)))
with open(OUT / "sf_pair_pruner_hparam_effects.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

COLS = ["#2a6fdb", "#e07b39", "#3a9e5f"]
xlab = [f"{c}\n{l}" + ("\n(=overlap)" if c == "300/150" and l == "vehicle" else "") for c, l in GROUPS]
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
x = np.arange(len(GROUPS))
for ax, (fac, levels, name) in zip(axes, FACTORS):
    wb = 0.8 / len(levels)
    for j, lv in enumerate(levels):
        rr = [next(r for r in rows if r["config"].endswith(f"{c.split('/')[0]}/step_size={c.split('/')[1]}") and r["label"] == l and r["factor"] == fac and r["level"] == lv) for c, l in GROUPS]
        ax.bar(x + (j - (len(levels) - 1) / 2) * wb, [r["delta_auc_vs_group_mean"] for r in rr], wb * 0.9,
               yerr=[r["auc_std"] for r in rr], capsize=2, color=COLS[j], label=f"{name} = {lv:g}")
    ax.axhline(0, color="gray", lw=1); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.set_ylabel("Δ val AUC vs.\nconfig×label mean"); ax.set_title(name, fontsize=10)
axes[-1].set_xticks(x); axes[-1].set_xticklabels(xlab, fontsize=8); axes[-1].set_xlabel("window config (batch_interval/step_size) and label")
fig.suptitle("SF pair pruner: effect of GNN hyperparameters on val ROC-AUC (val instance seed 2; grid 3 layers × 2 pos_weight × 2 lr × 3 seeds, hidden_dim 64, 100 epochs)\n"
             "bar = marginal mean over the other factors and seeds minus the config×label mean; error bar = std over those runs (n=12 layers, n=18 pos_weight / lr)", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"): fig.savefig(OUT / f"sf_pair_pruner_hparam_effects.{ext}", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
for j, pw in enumerate([1.0, 5.0]):
    rr = [next(r for r in rows if r["config"].endswith(f"{c.split('/')[0]}/step_size={c.split('/')[1]}") and r["label"] == l and r["factor"] == "pos_weight" and r["level"] == pw) for c, l in GROUPS]
    for a, key, scale in ((ax[0], "kept_at_0p5", 100), (ax[1], "recall_at_0p5", 1)):
        a.bar(x + (j - 0.5) * 0.4, [r[key] * scale for r in rr], 0.36, color=COLS[j], label=f"pos_weight = {pw:g}")
ax[0].set_ylabel("pairs kept at threshold 0.5 [%]"); ax[1].set_ylabel("recall of positive pairs at threshold 0.5"); ax[1].set_ylim(0.8, 1.0)
for a in ax:
    a.set_xticks(x); a.set_xticklabels(xlab, fontsize=7); a.grid(axis="y", alpha=.3); a.legend(fontsize=8)
ax[1].set_title("note: y-axis starts at 0.8", fontsize=8)
fig.suptitle("SF pair pruner: pos_weight shifts the operating point at threshold 0.5 (val instance seed 2; mean over layers, lr and seeds, n=18)", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "pdf"): fig.savefig(OUT / f"sf_pair_pruner_pw_operating_point.{ext}", dpi=150)
print(len(rows), "rows")
