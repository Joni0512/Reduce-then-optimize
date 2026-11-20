"""
2026-11-19: threshold selection for the Request Pruners trained on the C=2 data (variants base = 13 features, plus = 13 + 4 vehicle-related candidate features), in the spirit of Rule A
(CLAUDE.md 2026-10-20) but with the RHO-label analogue of its loss measure, because the original Rule A evaluator (select_request_pruner_by_vehicle_loss.py) needs the reference-solution
prefix dataset and the base feature array only (no cand_ columns):
  - candidates: thresholds 0.1 ... 0.9 (step 0.1) with pruning rate >= 30 % ("pruning rate" = pruned / candidates, candidates = non-active, non-urgent (window, request) rows,
    exactly the rows the runtime pruner scores; active proxy = assigned in an earlier window, urgent = time_until_latest_pickup <= step_size);
  - among them the smallest LOSS = share of first assignments (positive candidate rows) that are pruned; tie-break: higher pruning rate.
  - retention floor like the runtime pruner: per (instance, window) at least ceil(min_retention_fraction=0.3 x n_candidates) highest-scoring candidates are kept.
  - evaluated ONLY on the 8 pruner-val instances (the "val" split of the dataset), never on the 12 test instances (no leakage).
Deviation from the written Rule A: loss measure = first assignments of the RHO baseline (label) instead of the true first targets of the reference solution (lost_1st); the rest is identical.
Run: ./venv/bin/python3 scripts/analysis/select_request_pruner_threshold_cand.py --bi 40 --ss 10
Output: outputs/request_pruner_threshold_cand_bi<BI>_ss<SS>.{csv,json} (grid per variant + selected threshold)
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtv_solver.pipeline import train_request_pruner as trp
from rtv_solver.pipeline.request_pruner_mlp import RequestPrunerMLP

ap = argparse.ArgumentParser()
ap.add_argument("--bi", default="40")
ap.add_argument("--ss", default="10")
ap.add_argument("--min_prune", type=float, default=0.30)
ap.add_argument("--min_retention", type=float, default=0.3)
# 2026-11-19: optional finer grid (e.g. --ths 0.01 0.02 0.05 0.1 0.2) to find thresholds with a MATCHED pruning rate for two pruners; the default 0.1..0.9 grid keeps writing the selection json
ap.add_argument("--ths", type=float, nargs="+", default=None)
A = ap.parse_args()
D = Path(f"dataset/request_pruning_bi{A.bi}_ss{A.ss}_c2cand")
STEP = float(A.ss)
N = trp.NUM_DEPLOYABLE_FEATURES
fn = json.load(open(D / "manifest.json"))["feature_names"]
TUL = fn.index("time_until_latest_pickup")
THS = A.ths if A.ths else [round(0.1 * k, 1) for k in range(1, 10)]
SUFFIX = "_grid" if A.ths else ""  # custom grid never overwrites the selection json


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
        out.append(dict(inst=p.stem, win=w, feat=z["features"][:, :N], cand=z["cand_features"], lab=lab, active=active, tul=z["features"][:, TUL]))
    return out


train, val = load("train"), load("val")
med = np.nanmedian(np.vstack([d["cand"] for d in train]), axis=0)


def model_for(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = RequestPrunerMLP(feature_dim=len(ck["feature_names"]), hidden_dim=ck["hidden_dim"], num_hidden_layers=ck["num_hidden_layers"], dropout=ck["dropout"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, np.array(ck["feature_mean"], np.float32), np.array(ck["feature_std"], np.float32)


rows, selected = [], {}
for v in ("base", "plus"):
    path = glob.glob(f"outputs/request_pruner_mlp_bi{A.bi}_ss{A.ss}_c2cand_{v}/*/*_best_val_f3.pt")[0]
    m, mu, sd = model_for(path)
    parts = []
    for d in val:
        x = d["feat"] if v == "base" else np.hstack([d["feat"], np.where(np.isnan(d["cand"]), med[None, :], d["cand"])]).astype(np.float32)
        with torch.no_grad():
            s = torch.sigmoid(m(torch.tensor((x - mu) / sd, dtype=torch.float32))).numpy()
        cand_mask = (~d["active"]) & (d["tul"] > STEP)
        parts.append(pd.DataFrame(dict(inst=d["inst"], win=d["win"], score=s, lab=d["lab"])[:0] if False else dict(inst=d["inst"], win=d["win"], score=s, lab=d["lab"]))[cand_mask])
    df = pd.concat(parts, ignore_index=True)
    n_pos = int(df.lab.sum())
    for t in THS:
        pruned_pos = pruned_n = 0
        for _, g in df.groupby(["inst", "win"]):
            sc = g.score.to_numpy()
            keep = sc >= t
            floor_n = math.ceil(A.min_retention * len(g))
            if int(keep.sum()) < floor_n:
                keep = keep.copy()
                keep[np.argsort(-sc)[:floor_n]] = True
            pruned_n += int((~keep).sum())
            pruned_pos += int(g.lab.to_numpy()[~keep].sum())
        rows.append(dict(variant=v, threshold=t, candidates=len(df), first_assignments=n_pos, prune_rate=100 * pruned_n / len(df), lost_first_assign=100 * pruned_pos / n_pos))
    g = pd.DataFrame([r for r in rows if r["variant"] == v])
    ok = g[g.prune_rate >= 100 * A.min_prune]
    if len(ok) == 0:
        selected[v] = None
        continue
    best = ok.sort_values(["lost_first_assign", "prune_rate"], ascending=[True, False]).iloc[0]
    selected[v] = dict(threshold=float(best.threshold), prune_rate=float(best.prune_rate), lost_first_assign=float(best.lost_first_assign), checkpoint=path)
R = pd.DataFrame(rows)
R.round(2).to_csv(f"outputs/request_pruner_threshold_cand_bi{A.bi}_ss{A.ss}{SUFFIX}.csv", index=False)
if not A.ths:
    json.dump(selected, open(f"outputs/request_pruner_threshold_cand_bi{A.bi}_ss{A.ss}.json", "w"), indent=2)
print(R.round(1).to_string(index=False))
print(json.dumps(selected, indent=2))
