"""
2026-11-20: QUICK CHECK (no solver code, no new Pair Pruner): would vehicle-related PAIR features help to predict which request pairs RHO puts into one chosen trip?
Data: the C=2 baseline logs with "CandidateInsertions" (run_baseline_data_c2_candidates.py, outputs/eval_rh_no_learning_bi<BI>_ss<SS>_c2cand/<inst>/rh_no_learning/assignment_data.jsonl),
split train/val/test = request_pruner_dataset_builder.REQUEST_PRUNER_SPLIT ("mixed_all", same 36/8/12 instances as the Request Pruner).

Pair unit = (instance, window, request i < j) with BOTH requests in the window's CandidateInsertions record (>= 1 feasible single-request insertion) and NEITHER assigned before this window
(not in assigned_requests of the previous Status record), i.e. a decision about a NEW pair. LABEL = 1 if both requests are in the SAME chosen trip of this window (TripMetrics.booking_ids of the
window; TripMetrics records precede that window's CandidateInsertions/Status record). After a pair is chosen together both are assigned, so it never appears again (no commitment bias).
Limit: 1 RHO run per instance; label = RHO's own choice in the ILP, NOT the label of the existing Pair Pruner (v1_final: not verified, trained on whole-day class-2 solutions); the geometric
features below are my re-computation of the main Pair-Pruner edge features (Euclidean = Li&Lim travel time), not the checkpoint's inputs.

Features
  geo (9): pickup_distance, dropoff_distance, cross p_i->d_j, cross p_j->d_i, trip_distance_difference, pickup_time_difference (earliest pickup), pickup_window_overlap_seconds,
           latest_pickup_time_difference, direction_similarity (cosine of the pickup->dropoff vectors)
  veh (5): sum_best = cmin_i + cmin_j; max_best = max(cmin_i, cmin_j) (cmin = min over vehicles of added_cost of the single-request insertion); n_common = number of vehicles feasible for BOTH;
           same_best = 1 if the sets of cheapest vehicles intersect; best_common_sum = min over common vehicles v of cost_i(v) + cost_j(v) (NaN -> train median when n_common = 0).
Evaluation: (1) single-feature AUC (direction-free, max(auc, 1 - auc)) on all pairs of the split; (2) logistic regression and small MLP (hidden 32, 2 layers, 5 seeds) trained on train, AUC and
share of positive pairs KEPT when the worst 50 % / 70 % of the pairs of each window are removed (rank-based per (instance, window)) on val and test, for geo vs geo + veh.
Run: nice -n 19 ./venv/bin/python3 scripts/analysis/pair_vehicle_feature_check.py --bi 40 --ss 10
Output: outputs/pair_vehicle_feature_check_bi<BI>_ss<SS>_{single,models}.csv
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtv_solver.pipeline.request_pruner_dataset_builder import REQUEST_PRUNER_SPLIT

torch.set_num_threads(1)
REPO = Path(__file__).resolve().parents[2]
GEO = ["pickup_distance", "dropoff_distance", "cross_p1_d2", "cross_p2_d1", "trip_distance_difference", "pickup_time_difference", "pickup_window_overlap_seconds",
       "latest_pickup_time_difference", "direction_similarity"]
VEH = ["sum_best", "max_best", "n_common", "same_best", "best_common_sum"]


def pairs_for_instance(stem, bi, ss):
    log = REPO / f"outputs/eval_rh_no_learning_bi{bi}_ss{ss}_c2cand/{stem}/rh_no_learning/assignment_data.jsonl"
    if not log.exists():
        return None
    man = json.load(open(REPO / "solutions/li_lim/manifests" / f"{stem}.json"))
    R = {int(r["booking_id"]): r for r in man["requests"]}
    xy = lambda r, k: (r[k]["lon"], r[k]["lat"])
    dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    rows, trips, cand, assigned_prev, w = [], [], None, set(), 0
    for line in open(log):
        rec = json.loads(line)
        m = rec.get("message")
        if m == "TripMetrics":
            ids_ = rec["extra"]["booking_ids"]
            trips.append([int(b) for b in (json.loads(ids_) if isinstance(ids_, str) else ids_)])
        elif m == "CandidateInsertions":
            cand = rec["extra"]
        elif m == "Status":
            if cand is not None:
                ix = {c: i for i, c in enumerate(cand["columns"])}
                cost = {}
                for row in cand["candidates"]:
                    cost.setdefault(int(row[ix["request_id"]]), {})[int(row[ix["vehicle_id"]])] = float(row[ix["added_cost"]])
                same_trip = set()
                for t in trips:
                    for a in range(len(t)):
                        for b in range(a + 1, len(t)):
                            same_trip.add((min(t[a], t[b]), max(t[a], t[b])))
                ids = sorted(i for i in cost if i not in assigned_prev and i in R)
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        i, j = ids[a], ids[b]
                        ri, rj = R[i], R[j]
                        pi, di, pj, dj = xy(ri, "pickup_pt"), xy(ri, "dropoff_pt"), xy(rj, "pickup_pt"), xy(rj, "dropoff_pt")
                        vi, vj = (di[0] - pi[0], di[1] - pi[1]), (dj[0] - pj[0], dj[1] - pj[1])
                        ni, nj = math.hypot(*vi), math.hypot(*vj)
                        ci, cj = cost[i], cost[j]
                        mi, mj = min(ci.values()), min(cj.values())
                        bi = {v for v, c in ci.items() if c <= mi + 1e-9}
                        bj = {v for v, c in cj.items() if c <= mj + 1e-9}
                        common = set(ci) & set(cj)
                        ov = min(ri["pickup_time_window_end"], rj["pickup_time_window_end"]) - max(ri["pickup_time_window_start"], rj["pickup_time_window_start"])
                        rows.append(dict(instance=stem, window=w, i=i, j=j, label=int((i, j) in same_trip),
                                         pickup_distance=dist(pi, pj), dropoff_distance=dist(di, dj), cross_p1_d2=dist(pi, dj), cross_p2_d1=dist(pj, di),
                                         trip_distance_difference=abs(ni - nj), pickup_time_difference=abs(ri["pickup_time_window_start"] - rj["pickup_time_window_start"]),
                                         pickup_window_overlap_seconds=max(ov, 0), latest_pickup_time_difference=abs(ri["pickup_time_window_end"] - rj["pickup_time_window_end"]),
                                         direction_similarity=(vi[0] * vj[0] + vi[1] * vj[1]) / max(ni * nj, 1e-9),
                                         sum_best=mi + mj, max_best=max(mi, mj), n_common=len(common), same_best=int(bool(bi & bj)),
                                         best_common_sum=min(ci[v] + cj[v] for v in common) if common else np.nan))
            trips, cand = [], None
            # TripMetrics of this window were collected before the C/S records; the assignment after this window is what counts as "committed" for the next one
            st = rec["extra"]["status"]
            st = json.loads(st.replace("'", '"')) if isinstance(st, str) else st
            assigned_prev = {int(k) for k in st["assigned_requests"]}
            w += 1
    return pd.DataFrame(rows)


def auc(score, y):
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def kept_at(df, score, frac_removed):
    """share of positive pairs kept when per (instance, window) the worst `frac_removed` of the pairs (by score, higher = better) are removed"""
    d = df.assign(s=score)
    pos = int(d.label.sum())
    kept = 0
    for _, g in d.groupby(["instance", "window"]):
        n_keep = max(1, math.ceil((1 - frac_removed) * len(g)))
        kept += int(g.sort_values("s", ascending=False).head(n_keep).label.sum())
    return kept / pos if pos else float("nan")


def fit_predict(Xtr, ytr, Xev_list, kind, seed):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    f = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)
    net = torch.nn.Linear(Xtr.shape[1], 1) if kind == "logreg" else torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    X, y = f(Xtr), torch.tensor(ytr, dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - ytr.mean()) / max(ytr.mean(), 1e-6)))
    for _ in range(200):
        opt.zero_grad()
        loss = lossf(net(X).squeeze(1), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return [net(f(Xe)).squeeze(1).numpy() for Xe in Xev_list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", default="40")
    ap.add_argument("--ss", default="10")
    a = ap.parse_args()
    split_of = {n: s for s, names in REQUEST_PRUNER_SPLIT.items() for n in names}
    parts = []
    for stem, sp in split_of.items():
        d = pairs_for_instance(stem, a.bi, a.ss)
        if d is not None and len(d):
            d["split"] = sp
            parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    print(f"window {a.bi}/{a.ss}: pairs {len(D)}, positives {int(D.label.sum())} ({100 * D.label.mean():.2f} %), instances {D.instance.nunique()}",
          {s: (len(g), int(g.label.sum())) for s, g in D.groupby('split')})
    med = D[D.split == "train"].best_common_sum.median()
    D["best_common_sum"] = D.best_common_sum.fillna(med)
    single = []
    for f in GEO + VEH:
        for sp in ("train", "val", "test"):
            g = D[D.split == sp]
            s = g[f].to_numpy(float)
            au = auc(s, g.label.to_numpy())
            single.append(dict(window=f"{a.bi}/{a.ss}", feature=f, group="geo" if f in GEO else "veh", split=sp, auc=max(au, 1 - au), sign="high=positive" if au >= 0.5 else "low=positive"))
    S = pd.DataFrame(single)
    S.round(4).to_csv(REPO / f"outputs/pair_vehicle_feature_check_bi{a.bi}_ss{a.ss}_single.csv", index=False)
    print(S[S.split == "test"].round(3).to_string(index=False))
    tr = D[D.split == "train"]
    res = []
    for kind in ("logreg", "mlp"):
        for name, cols in (("geo", GEO), ("geo+veh", GEO + VEH)):
            for seed in range(5 if kind == "mlp" else 1):
                evs = [D[D.split == sp] for sp in ("val", "test")]
                preds = fit_predict(tr[cols].to_numpy(float), tr.label.to_numpy(), [e[cols].to_numpy(float) for e in evs], kind, seed)
                for sp, e, p in zip(("val", "test"), evs, preds):
                    res.append(dict(window=f"{a.bi}/{a.ss}", model=kind, features=name, seed=seed, split=sp, auc=auc(p, e.label.to_numpy()),
                                    kept_prune50=kept_at(e, p, 0.5), kept_prune70=kept_at(e, p, 0.7)))
    M = pd.DataFrame(res)
    M.round(4).to_csv(REPO / f"outputs/pair_vehicle_feature_check_bi{a.bi}_ss{a.ss}_models.csv", index=False)
    print(M.groupby(["model", "features", "split"])[["auc", "kept_prune50", "kept_prune70"]].mean().round(3))


if __name__ == "__main__":
    main()
