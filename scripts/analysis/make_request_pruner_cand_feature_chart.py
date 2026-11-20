"""
2026-11-19: Request Pruner WITH vs. WITHOUT the four vehicle-related candidate-insertion features (cand_min_veh_to_pickup, cand_min_added_cost, cand_min_cost_ratio, cand_n_feasible_vehicles),
window batch_interval=40/step_size=10, C=2 baseline logs (dataset/request_pruning_bi40_ss10_c2cand, 36 train / 8 val / 12 test instances). Both variants: same rows, same grid (432 configs x 5 seeds),
same selection (mean val ROC-AUC), winner retrained with seed 42 (sweep_request_pruner_cand.py). Evaluation with the two winner checkpoints:
  (a) ROC-AUC on ALL rows, on NON-ACTIVE rows (request not assigned in an earlier window; proxy for 'active' like in analyze_request_pruner_false_prunes.py) and on non-active AND non-urgent rows
      (time_until_latest_pickup > step_size=10) = the rows the pruner really decides on (active/urgent requests are force-kept at inference).
  (b) share of FIRST assignments (positive non-active rows) that survive when the lowest-scored 30/50/70 % of the non-active non-urgent (window, request) rows are pruned. A pruned request is only
      deferred, not rejected (it can be assigned in a later window), so this is a first-order measure, not the RHO effect.
Test is reported only; selection used val. Imputation of missing cand values with TRAIN medians as in the sweep.
Run: ./venv/bin/python3 scripts/analysis/make_request_pruner_cand_feature_chart.py
Output (figures_export/Request Pruner RHO/): request_pruner_cand_feature_bi40_ss10.{png,pdf}, ..._table.csv, ..._table.tex
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rtv_solver.pipeline import train_request_pruner as trp
from rtv_solver.pipeline.request_pruner_mlp import RequestPrunerMLP

OUT = Path("figures_export/Request Pruner RHO")
OUT.mkdir(parents=True, exist_ok=True)
D = Path("dataset/request_pruning_bi40_ss10_c2cand")
N = trp.NUM_DEPLOYABLE_FEATURES
fn = json.load(open(D / "manifest.json"))["feature_names"]
TUL = fn.index("time_until_latest_pickup")
STEP = 10  # urgent = time_until_latest_pickup <= step_size


def inst_class(stem):
    return stem[3] if stem.startswith("lrc") else stem[2]


def load(split):
    out = []
    for p in sorted((D / split).glob("*.npz")):
        z = np.load(p)
        w, ids, lab = z["window_index"], z["node_ids"], z["label"]
        active, seen = np.zeros(len(ids), bool), set()
        for ww in np.unique(w):
            idx = np.where(w == ww)[0]
            new = []
            for i in idx:
                if ids[i] in seen:
                    active[i] = True
                elif lab[i] == 1:
                    new.append(ids[i])
            seen.update(new)
        out.append(dict(inst=p.stem, cls=inst_class(p.stem), feat=z["features"][:, :N], cand=z["cand_features"], lab=lab, active=active, tul=z["features"][:, TUL]))
    return out


sp = {s: load(s) for s in ("train", "val", "test")}
med = np.nanmedian(np.vstack([d["cand"] for d in sp["train"]]), axis=0)
PATH = {v: glob.glob(f"outputs/request_pruner_mlp_bi40_ss10_c2cand_{v}/*/*.pt")[0] for v in ("base", "plus")}


def model_for(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = RequestPrunerMLP(feature_dim=len(ck["feature_names"]), hidden_dim=ck["hidden_dim"], num_hidden_layers=ck["num_hidden_layers"], dropout=ck["dropout"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, np.array(ck["feature_mean"], np.float32), np.array(ck["feature_std"], np.float32)


M = {v: model_for(p) for v, p in PATH.items()}


def scores(v, d):
    m, mu, sd = M[v]
    x = d["feat"] if v == "base" else np.hstack([d["feat"], np.where(np.isnan(d["cand"]), med[None, :], d["cand"])]).astype(np.float32)
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor((x - mu) / sd, dtype=torch.float32))).numpy()


def kept(S, L, rate):
    k = int(round(rate * len(S)))
    return 100 * (1 - L[np.argsort(S)[:k]].sum() / L.sum())


SEL = {"all rows": lambda d: np.ones(len(d["lab"]), bool), "non-active": lambda d: ~d["active"], "non-active, non-urgent": lambda d: (~d["active"]) & (d["tul"] > STEP)}
rows = []
for split in ("val", "test"):
    for name, sel in SEL.items():
        r = dict(split=split, subset=name)
        for v in ("base", "plus"):
            L = np.concatenate([d["lab"][sel(d)] for d in sp[split]])
            S = np.concatenate([scores(v, d)[sel(d)] for d in sp[split]])
            r[f"auc_{v}"] = trp.roc_auc(L, S)
            r["rows"], r["positives"] = len(L), int(L.sum())
        rows.append(r)
A = pd.DataFrame(rows)
rows2 = []
for split in ("val", "test"):
    for cls in ("all", "1", "2"):
        r = dict(split=split, cls=cls)
        for v in ("base", "plus"):
            Ls = [d["lab"][SEL["non-active, non-urgent"](d)] for d in sp[split] if cls == "all" or d["cls"] == cls]
            Ss = [scores(v, d)[SEL["non-active, non-urgent"](d)] for d in sp[split] if cls == "all" or d["cls"] == cls]
            L, S = np.concatenate(Ls), np.concatenate(Ss)
            for rate in (0.3, 0.5, 0.7):
                r[f"kept_{v}_{int(rate * 100)}"] = kept(S, L, rate)
            r["rows"], r["positives"] = len(L), int(L.sum())
        rows2.append(r)
K = pd.DataFrame(rows2)
A.round(3).to_csv(OUT / "request_pruner_cand_feature_bi40_ss10_table_auc.csv", index=False)
K.round(1).to_csv(OUT / "request_pruner_cand_feature_bi40_ss10_table.csv", index=False)

CB, CP = "#8C8C8C", "#E8A33D"
fig, ax = plt.subplots(1, 2, figsize=(15, 5.9))
a = ax[0]
labels = [f"{r.split}\n{r.subset}" for r in A.itertuples()]
x = np.arange(len(A))
a.bar(x - 0.19, A.auc_base, 0.38, color=CB, label="13 features (base)")
a.bar(x + 0.19, A.auc_plus, 0.38, color=CP, label="+ 4 vehicle features (plus)")
for i, r in enumerate(A.itertuples()):
    a.text(i - 0.19, r.auc_base + 0.004, f"{r.auc_base:.3f}", ha="center", fontsize=7.5)
    a.text(i + 0.19, r.auc_plus + 0.004, f"{r.auc_plus:.3f}", ha="center", fontsize=7.5)
a.set_xticks(x)
a.set_xticklabels(labels, fontsize=7.5)
a.set_ylim(0.75, 1.0)
a.set_ylabel("ROC-AUC (label: request assigned in this window)", fontsize=10)
a.set_title("(a) ROC-AUC of the two winner checkpoints (y-axis starts at 0.75)", fontsize=10)
a.legend(frameon=False, fontsize=8.5, loc="upper left")
b = ax[1]
T = K[K.split == "test"].reset_index(drop=True)
x = np.arange(len(T) * 3)
w = 0.38
for i, r in enumerate(T.itertuples()):
    for j, rate in enumerate((30, 50, 70)):
        xx = i * 3.6 + j
        vb, vp = getattr(r, f"kept_base_{rate}"), getattr(r, f"kept_plus_{rate}")
        b.bar(xx - 0.19, vb, 0.38, color=CB)
        b.bar(xx + 0.19, vp, 0.38, color=CP)
        b.text(xx - 0.19, vb + 0.8, f"{vb:.0f}", ha="center", fontsize=7)
        b.text(xx + 0.19, vp + 0.8, f"{vp:.0f}", ha="center", fontsize=7)
    b.text(i * 3.6 + 1, 36, {"all": "all classes", "1": "class 1", "2": "class 2"}[r.cls] + f"\n({r.rows} rows, {r.positives} first assignments)", ha="center", fontsize=7.8)
b.set_xticks([i * 3.6 + j for i in range(len(T)) for j in range(3)])
b.set_xticklabels([f"{r} %\npruned" for _ in range(len(T)) for r in (30, 50, 70)], fontsize=7.5)
b.set_ylim(30, 104)
b.set_ylabel("Share of first assignments that survive (%)", fontsize=10)
b.set_title("(b) Test: lowest-scored share of non-active, non-urgent (window, request) rows pruned; y-axis starts at 30 %", fontsize=9.5)
for z in ax:
    z.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp_ in ("top", "right"):
        z.spines[sp_].set_visible(False)
fig.suptitle("Request Pruner with vs. without vehicle-related features (extra trip time / pickup time), batch_interval=40/step_size=10, C=2 baseline logs", fontsize=11.5)
fig.text(0.5, 0.012, "Data: dataset/request_pruning_bi40_ss10_c2cand (RHO offline, batch_interval=40, step_size=10, C=2, max_cardinality 2, no pruner, seed 42; 36 train / 8 val / 12 test Li&Lim instances). Both variants: 432 configs x 5 seeds, selection = mean val ROC-AUC,\n"
         "winners retrained with seed 42: base h64/l3/d0.2/pw1.0/lr0.003, plus h32/l3/d0.2/pw1.0/lr0.03 (sweep_request_pruner_cand.py). New features = min over vehicles of pickup time, added_cost, added_cost/pickup time, number of feasible vehicles.\n"
         "'Non-active' = not assigned in an earlier window (proxy); a pruned request is only deferred, not rejected -> first-order measure, not the RHO effect. Test reported only (selection on val). Script: scripts/analysis/make_request_pruner_cand_feature_chart.py",
         ha="center", fontsize=6.7, color="#666666")
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"request_pruner_cand_feature_bi40_ss10.{ext}", dpi=200)
plt.close(fig)

lines = []
for r in K.itertuples():
    lines.append(f"{r.split} & {r.cls} & {r.rows} & {r.positives} & " + " & ".join(f"{getattr(r, f'kept_base_{k}'):.0f} / {getattr(r, f'kept_plus_{k}'):.0f}" for k in (30, 50, 70)) + " \\\\")
tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
       "\\caption{Request Pruner without / with the four vehicle-related features (\\texttt{batch\\_interval}=40, \\texttt{step\\_size}=10, $C{=}2$ baseline logs): share (\\%) of first assignments that survive when the "
       "lowest-scored 30/50/70\\,\\% of the non-active, non-urgent (window, request) rows are pruned; each cell is base / plus. Pruned requests are deferred, not rejected.}\n"
       "\\begin{tabular}{llrrccc}\n\\toprule\nSplit & Class & Rows & First assign. & 30\\,\\% & 50\\,\\% & 70\\,\\% \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(OUT / "request_pruner_cand_feature_bi40_ss10_table.tex").write_text(tex)
print(A.round(3).to_string(index=False))
print(K.round(1).to_string(index=False))
