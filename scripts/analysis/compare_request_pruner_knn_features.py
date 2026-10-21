"""
2026-10-20: Offline test - do top-k nearest-neighbour pickup distances (knn1, knn2, knn3, absolute) help the
Request Pruner (c2, batch_interval=200/step_size=100, prefix labels)?

Motivation: single-feature AUC check (chat 2026-10-20) showed knn2/knn3 separate the prefix label (lc 0.67-0.80,
lr/lrc 0.55-0.65, isolated request = more often a first target), median-normalised versions did not. Single-feature
correlation does not show added value over the existing 13 features -> this script trains with and without.

Design (nothing in the solver / existing pruner code is changed):
  - Same trainer as the existing pruners (train_request_pruner.train_one_config) and the HPs of the existing
    bi200/ss100 winner (hidden 32, 1 layer, dropout 0.2, pos_weight 1.0, lr 0.01), 5 seeds; ONLY the feature set differs:
        base: the 13 deployable features    |    knn: 13 + knn1, knn2, knn3
    knn_k = distance (pickup coordinates, Euclidean, as local_request_count_10min) from a request to its k-th nearest other
    request in the SAME window (all active requests of the window, force-kept ones included - same population as at runtime).
    Windows with fewer than k+1 requests use their farthest available neighbour; single-request windows use the largest
    knn value seen in train (placeholder, decision still open).
  - Evaluation = the replay of select_request_pruner_by_vehicle_loss.py (rule A) on the 8 pruner-VAL instances (never the
    12 test = SIL-val instances): min_retention 0.3, pruning rate = pruned/candidates over all windows, lost_1st/lost_anyK =
    vehicles losing their true first / any of first K=2 reference targets in window 0 (53 vehicles, one = 1.9 pp),
    prefix_pruned = share of prefix-label requests pruned over all windows (smoother, approximate after window 0).
  - Compared AT EQUAL pruning rate (30/40/50 %) by linear interpolation over the threshold grid 0.05..0.95.
  - Sanity: the existing checkpoint (h32_l1_d0p2_pw1p0_lr0p01) is replayed with the same code; at t=0.2 it must reproduce
    prune 33.58 % / lost_1st 5.66 % from outputs/request_pruner_selection_by_vehicle_loss.csv.
  - The val split also selects the best epoch inside train_one_config (val F3) - identical for both variants.

2026-10-20 (extension): third variant "drop" = 13 + time_until_latest_arrival (dropoff_time_window_end - window timestamp, the
drop-off analogue of the existing time_until_latest_pickup) + dropoff_window_length (end - start). Reason: user asked for
drop-off urgency; the dropoff window is not in the npz, so it is read from solutions/li_lim/manifests/<stem>.json
(latest_arrival_time = dropoff_time_window_end, verified in request_handler.py).

2026-10-20 (extension 2): fourth variant "share" = 13 + share_deg + share_frac ("request shareability"): number / share of the other
requests in the window with which r forms a FEASIBLE 2-request trip, using the same check as the solver's cardinality-2 step
(VehicleHandler.get_exact_stop_sequence via can_serve_trips: start at one of the two pickups, try every pickup/dropoff order,
arrival = previous time + travel time, wait until the window start, feasible iff every arrival <= latest, dwell added after).
Assumptions (not verified against the solver in every detail): start time = max(window timestamp, min(earliest pickups));
vehicle capacity and SHAREABLE_COST_FACTOR are ignored; travel times = instance travel_time_matrix. No vehicle state is used,
as in the solver's pair check (only _check_any_vehicles_available adds vehicles later). Reason: user asked for a "request
shareability" feature; computing it exactly would be the cardinality-2 trip generation itself, this is the cheap, vehicle-free version.

Run: ./venv/bin/python3 scripts/analysis/compare_request_pruner_knn_features.py [--seeds 42 1 2 3 4]
Output: outputs/request_pruner_feature_test.csv (all points), summary printed.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtv_solver.pipeline import analyze_request_pruner_false_prunes as A
from rtv_solver.pipeline.build_request_pruner_prefix_dataset import K, expert_by_vehicle
from rtv_solver.pipeline.request_pruner_mlp import RequestPrunerMLP
from rtv_solver.pipeline.train_request_pruner import NUM_DEPLOYABLE_FEATURES, train_one_config

PREFIX_DIR = Path("dataset/request_pruning_bi200_ss100_prefix")
RHO_DIR = Path("dataset/request_pruning_bi200_ss100")
OLD_CKPT = Path("outputs/request_pruner_mlp_bi200_ss100_prefix/pw1.0/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/"
                "request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt")
KS = (1, 2, 3)
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
TARGET_RATES = (30, 40, 50)
MIN_RETENTION = 0.3

manifest = json.load(open(PREFIX_DIR / "manifest.json"))
FI = {n: i for i, n in enumerate(manifest["feature_names"])}


def knn_cols(features, window_index):
    """(rows, len(KS)) knn distances per window; NaN for single-request windows (filled later)."""
    out = np.full((len(features), len(KS)), np.nan, dtype=np.float32)
    for w in np.unique(window_index):
        idx = np.where(window_index == w)[0]
        n = len(idx)
        if n < 2:
            continue
        xy = features[idx][:, [FI["pickup_lat"], FI["pickup_lon"]]]
        d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        ds = np.sort(d, axis=1)[:, : n - 1]
        for j, k in enumerate(KS):
            out[idx, j] = ds[:, min(k, n - 1) - 1]
    return out


MANIFEST_DIR = Path("solutions/li_lim/manifests")
_DROP_CACHE = {}


def drop_cols(stem, node_ids, timestamp):
    """(rows, 2): time_until_latest_arrival, dropoff_window_length (2026-10-20, drop-off urgency test)."""
    if stem not in _DROP_CACHE:
        reqs = json.load(open(MANIFEST_DIR / f"{stem}.json"))["requests"]
        _DROP_CACHE[stem] = {r["booking_id"]: (r["dropoff_time_window_start"], r["dropoff_time_window_end"]) for r in reqs}
    tw = _DROP_CACHE[stem]
    start = np.array([tw[int(i)][0] for i in node_ids], dtype=np.float32)
    end = np.array([tw[int(i)][1] for i in node_ids], dtype=np.float32)
    return np.stack([end - timestamp, end - start], axis=1).astype(np.float32)


_INST_CACHE = {}


def _instance(stem):
    if stem not in _INST_CACHE:
        d = json.load(open(MANIFEST_DIR / f"{stem}.json"))
        _INST_CACHE[stem] = (np.array(d["travel_time_matrix"], dtype=np.float64), {r["booking_id"]: r for r in d["requests"]})
    return _INST_CACHE[stem]


def _pair_feasible(tt, a, b, t0):
    """Exact search over all stop orders of two requests (solver-style), start at either pickup at time t0."""
    stops = {}
    for k, r in (("a", a), ("b", b)):
        stops[k] = dict(
            p=r["pickup_pt"]["node_id"], d=r["dropoff_pt"]["node_id"],
            pe=r["pickup_time_window_start"], pl=r["pickup_time_window_end"],
            de=r["dropoff_time_window_start"], dl=r["dropoff_time_window_end"],
            ps=r["pickup_service_time"], ds=r["dropoff_service_time"])

    def rec(node, t, to_pick, to_drop):
        if not to_pick and not to_drop:
            return True
        for k in list(to_pick):
            x = stops[k]
            arr = max(t + tt[node][x["p"]], x["pe"])
            if arr <= x["pl"] and rec(x["p"], arr + x["ps"], to_pick - {k}, to_drop):
                return True
        for k in list(to_drop):
            if k in to_pick:
                continue
            x = stops[k]
            arr = t + tt[node][x["d"]]
            if arr <= x["dl"]:
                arr = max(arr, x["de"])
                if rec(x["d"], arr + x["ds"], to_pick, to_drop - {k}):
                    return True
        return False

    for start in (stops["a"]["p"], stops["b"]["p"]):
        if rec(start, t0, {"a", "b"}, {"a", "b"}):
            return True
    return False


def share_cols(stem, node_ids, window_index, timestamp):
    """(rows, 2): share_deg, share_frac per request within its window (2026-10-20, shareability test)."""
    tt, reqs = _instance(stem)
    out = np.zeros((len(node_ids), 2), dtype=np.float32)
    for w in np.unique(window_index):
        idx = np.where(window_index == w)[0]
        n, ts = len(idx), float(timestamp[idx[0]])
        deg = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = reqs[int(node_ids[idx[i]])], reqs[int(node_ids[idx[j]])]
                t0 = max(ts, min(a["pickup_time_window_start"], b["pickup_time_window_start"]))
                if _pair_feasible(tt, a, b, t0):
                    deg[i] += 1; deg[j] += 1
        out[idx, 0] = deg
        out[idx, 1] = deg / (n - 1) if n > 1 else 0.0
    return out


def load_split(split, fill=None):
    feats, extra, drop, share, labels, inst = [], [], [], [], [], []
    for p in sorted((PREFIX_DIR / split).glob("*.npz")):
        z = np.load(p)
        feats.append(z["features"][:, :NUM_DEPLOYABLE_FEATURES])
        extra.append(knn_cols(z["features"], z["window_index"]))
        drop.append(drop_cols(p.stem, z["node_ids"], z["timestamp"]))
        share.append(share_cols(p.stem, z["node_ids"], z["window_index"], z["timestamp"]))
        labels.append(z["label"])
        inst.append(np.full(len(z["label"]), p.stem))
    extra = np.concatenate(extra)
    if fill is None:
        fill = float(np.nanmax(extra))
    extra = np.where(np.isnan(extra), fill, extra)
    base = np.concatenate(feats).astype(np.float32)
    lab, ins = np.concatenate(labels).astype(np.float32), np.concatenate(inst)
    drop, share = np.concatenate(drop), np.concatenate(share)
    mk = lambda cols: {"features": np.concatenate([base] + cols, axis=1).astype(np.float32), "labels": lab, "instances": ins}
    return {"base": mk([]), "knn": mk([extra]), "drop": mk([drop]), "share": mk([share])}, fill


def load_model(ckpt_path, feature_dim=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = RequestPrunerMLP(feature_dim=feature_dim or len(ck["feature_names"]), hidden_dim=ck["hidden_dim"],
                             num_hidden_layers=ck["num_hidden_layers"], dropout=ck["dropout"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def val_stems():
    return sorted(k for k, v in manifest["instances"].items() if v["split"] == "val")


def prepare_replay(base_model, base_ck, fill):
    """Per val instance: replay frame (active/urgent proxy exactly as rule A) + prefix label + feature matrices."""
    A.DATASET_DIR, A.URGENT_SLACK = RHO_DIR, 100.0
    out = {}
    for s in val_stems():
        df = A.replay_instance(s, base_model, base_ck, FI)
        z = np.load(PREFIX_DIR / "val" / f"{s}.npz")
        df["prefix"] = z["label"] > 0.5
        extra = knn_cols(z["features"], z["window_index"])
        extra = np.where(np.isnan(extra), fill, extra)
        base = z["features"][:, :NUM_DEPLOYABLE_FEATURES].astype(np.float32)
        drop = drop_cols(s, z["node_ids"], z["timestamp"])
        share = share_cols(s, z["node_ids"], z["window_index"], z["timestamp"])
        out[s] = (df, {"share": np.concatenate([base, share], axis=1).astype(np.float32),"base": base, "knn": np.concatenate([base, extra], axis=1).astype(np.float32),
                       "drop": np.concatenate([base, drop], axis=1).astype(np.float32)})
    return out


def sweep_points(replay, model, mean, std, variant):
    scored = {}
    for s, (df, mats) in replay.items():
        x = (mats[variant] - mean) / std
        with torch.no_grad():
            sc = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).numpy()
        d = df.copy()
        d["score"] = sc
        scored[s] = d
    rows = []
    for t in THRESHOLDS:
        A.THRESHOLD = t
        cand = pruned = veh = lost1 = lostK = pre = pre_pruned = 0
        for s, d in scored.items():
            pr, _ = A.apply_pruner(d, MIN_RETENTION)
            c = ~(d["active"] | d["urgent"])
            cand += int(c.sum()); pruned += int(pr.sum())
            pre += int(d["prefix"].sum()); pre_pruned += int((d["prefix"].to_numpy() & pr).sum())
            w0 = d[d.window == 0].reset_index(drop=True)
            pw0, _ = A.apply_pruner(w0, MIN_RETENTION)
            flag = dict(zip(w0.request, pw0))
            batch = set(w0.request)
            for seq in expert_by_vehicle(s).values():
                proj = [r for r in seq if r in batch]
                if proj:
                    veh += 1; lost1 += int(flag[proj[0]]); lostK += int(any(flag[r] for r in proj[:K]))
        rows.append(dict(threshold=t, vehicles=veh, prune_rate=100 * pruned / max(cand, 1), lost_1st=100 * lost1 / max(veh, 1),
                         lost_anyK=100 * lostK / max(veh, 1), prefix_pruned=100 * pre_pruned / max(pre, 1)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=["base", "knn", "drop", "share"])
    a = ap.parse_args()

    tr, fill = load_split("train")
    va, _ = load_split("val", fill)
    te, _ = load_split("test", fill)
    print(f"rows train/val/test = {len(tr['base']['labels'])}/{len(va['base']['labels'])}/{len(te['base']['labels'])}; fill for 1-request windows = {fill:.2f}")

    old_model, old_ck = load_model(OLD_CKPT)
    A.CKPT = OLD_CKPT
    replay = prepare_replay(old_model, old_ck, fill)
    print(f"replay on {len(replay)} pruner-val instances: {', '.join(replay)}")

    all_pts = []
    pts = sweep_points(replay, old_model, np.array(old_ck["feature_mean"], dtype=np.float32),
                       np.array(old_ck["feature_std"], dtype=np.float32), variant="base")
    pts.insert(0, "variant", "old_checkpoint"); pts.insert(1, "seed", -1); all_pts.append(pts)
    chk = pts[np.isclose(pts.threshold, 0.2)].iloc[0]
    print(f"sanity old checkpoint t=0.2: prune {chk.prune_rate:.2f} % / lost_1st {chk.lost_1st:.2f} % (rule-A csv: 33.58 / 5.66)")

    tmp = Path(tempfile.mkdtemp(prefix="knn_pruner_"))
    for variant in a.variants:
        for seed in a.seeds:
            res = train_one_config(tr[variant], va[variant], te[variant], hidden_dim=32, num_hidden_layers=1, dropout=0.2, pos_weight=1.0,
                                   learning_rate=0.01, seed=seed, verbose=False, save_checkpoint=True,
                                   output_dir=tmp / variant / f"seed{seed}")
            model, ck = load_model(res["best_model_path"], feature_dim=tr[variant]["features"].shape[1])
            pts = sweep_points(replay, model, np.array(ck["feature_mean"], dtype=np.float32),
                               np.array(ck["feature_std"], dtype=np.float32), variant=variant)
            pts.insert(0, "variant", variant); pts.insert(1, "seed", seed)
            pts["val_f3"] = res["best_val_f3"]; pts["val_roc_auc"] = res["metrics"]["val"][0.5]["roc_auc"]
            all_pts.append(pts)
            print(f"  {variant:4s} seed {seed}: best_val_f3={res['best_val_f3']:.3f} val_roc_auc={res['metrics']['val'][0.5]['roc_auc']:.3f}")

    df = pd.concat(all_pts, ignore_index=True)
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/request_pruner_feature_test.csv", index=False)

    summ = []
    for (variant, seed), g in df[df.variant != "old_checkpoint"].groupby(["variant", "seed"]):
        g = g.sort_values("prune_rate")
        for r in TARGET_RATES:
            summ.append(dict(variant=variant, seed=seed, target_prune=r, **{m: np.interp(r, g.prune_rate, g[m])
                             for m in ("lost_1st", "lost_anyK", "prefix_pruned")}))
    s = pd.DataFrame(summ)
    agg = s.groupby(["target_prune", "variant"])[["lost_1st", "lost_anyK", "prefix_pruned"]].agg(["mean", "std"])
    with pd.option_context("display.width", 220, "display.float_format", "{:.2f}".format):
        print("\nAt equal pruning rate (mean/std over seeds; lower = better)")
        print(agg.to_string())
        piv = s.pivot_table(index=["target_prune", "seed"], columns="variant", values="prefix_pruned")
        for v in [v for v in a.variants if v != "base"]:
            piv[f"{v} - base"] = piv[v] - piv["base"]
        print("\nPer-seed prefix_pruned (pp), variant minus base, negative = variant better")
        print(piv.to_string())
        f = df[df.variant != "old_checkpoint"].drop_duplicates(["variant", "seed"]).groupby("variant")[["val_f3", "val_roc_auc"]].agg(["mean", "std"])
        print("\nval F3 / ROC-AUC at threshold 0.5"); print(f.to_string())


if __name__ == "__main__":
    main()
