"""
2026-11-19: SF request pruner (batch_interval=900/step_size=450, label pickup_before_next_resolve) vs. the time rule,
offline on the label dataset (no RHO run). Checkpoint = winner of sweep_sf_request_pruner_bi900_ss450.py
(h64/l1/d0.2/pw1.0/lr0.03, seed 42). Data = dataset/request_pruning_sf_pickup_bi900_ss450 (train seed 1, val seed 2).
Time rule (approximation, same as sf_request_pruner_label_check.py): keep iff the request was assigned in an earlier
window of the same run or earliest_pickup_time < t + step_size; in the dataset: time_since_earliest_pickup > -step_size
or seen assigned before (reconstructed per instance from window order + label-independent "assigned" info is NOT in the
npz, so the "assigned earlier" part is taken from the RHO log via node_ids/window_index).
Per variant: share of rows kept, recall of positive rows (= label-1 rows kept), positives lost.
Variants: time rule alone; pruner alone at thresholds 0.1-0.9; pruner applied ON TOP of the time rule.
Output: figures_export/SF/sf_request_pruner_vs_time_rule.csv
"""
import csv, glob, json
from pathlib import Path

import numpy as np
import torch

from rtv_solver.pipeline.request_pruner_mlp import RequestPrunerMLP

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset/request_pruning_sf_pickup_bi900_ss450"
CKPT = ROOT / "outputs/request_pruner_mlp_sf_pickup_bi900_ss450/request_pruner_mlp_h64_l1_d0p2_pw1p0_lr0p03/request_pruner_mlp_h64_l1_d0p2_pw1p0_lr0p03_best_val_f3.pt"
STEP = 450
OUT = ROOT / "figures_export/SF/sf_request_pruner_vs_time_rule.csv"

c = torch.load(CKPT, weights_only=False)
model = RequestPrunerMLP(feature_dim=len(c["feature_names"]), hidden_dim=c["hidden_dim"], num_hidden_layers=c["num_hidden_layers"], dropout=c["dropout"])
model.load_state_dict(c["model_state_dict"]); model.eval()
mean, std = np.array(c["feature_mean"]), np.array(c["feature_std"])
manifest = json.load(open(DS / "manifest.json"))
names = manifest["feature_names"]
rows = []
for split in ("train", "val"):
    f = glob.glob(str(DS / split / "*.npz"))[0]
    d = np.load(f); X = d["features"]; y = d["label"].astype(bool); nid = d["node_ids"]; w = d["window_index"]
    # "assigned in an earlier window" from the RHO log that produced this npz
    log = Path(manifest["instances"][Path(f).stem]["source_run"]) / "assignment_data.jsonl"
    st = [json.loads(l) for l in open(ROOT / log) if json.loads(l).get("message") == "Status"]
    st = [e for e in st if e["extra"]["status"]["assigned_requests"] or e["extra"]["status"]["unserved_requests"]]
    seen, earlier = set(), []
    for e in st:
        earlier.append(set(seen)); seen |= {int(r) for r in e["extra"]["status"]["assigned_requests"]}
    was_assigned = np.array([n in earlier[wi] for n, wi in zip(nid, w)])
    rule_keep = was_assigned | (X[:, names.index("time_since_earliest_pickup")] > -STEP)
    with torch.no_grad():
        Xn = torch.tensor((X[:, :len(c["feature_names"])] - mean) / std, dtype=torch.float32)
        score = torch.sigmoid(model(Xn).squeeze(-1)).numpy()

    def add(variant, thr, keep):
        rows.append({"split": split, "variant": variant, "threshold": thr, "rows": len(y), "kept_share": round(keep.mean(), 4),
                     "pruned_share": round(1 - keep.mean(), 4), "recall_pos": round((keep & y).sum() / y.sum(), 4),
                     "positives_lost": int((~keep & y).sum()), "positives": int(y.sum())})
    add("time rule", "-", rule_keep)
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        add("pruner alone", t, score >= t)
        add("time rule + pruner", t, rule_keep & (score >= t))
with open(OUT, "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
for r in rows:
    if r["split"] == "val": print(r)
