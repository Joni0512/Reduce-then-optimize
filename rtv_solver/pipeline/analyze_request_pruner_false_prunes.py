"""
2026-09-21: Offline false-prune analysis of the Request Pruner (Li&Lim, bi200/ss100).

Question: which requests does the pruner drop that the RHO baseline would
actually have assigned in that window? (Source of the label: RHO-baseline
assignment_data.jsonl, see request_pruner_dataset_builder.py.)

Method: replays RequestPruner.filter_requests (request_pruner.py) on the
stored per-window feature rows in dataset/request_pruning_bi200_ss100/ -
same order of steps as the solver: (1) force-keep urgent
(time_until_latest_pickup <= STEP_SIZE=100) and active requests, (2) score the
remaining candidates with the trained MLP, keep score >= threshold, (3) retention
floor: top up with the highest-scoring pruned candidates until
ceil(min_retention_fraction * n_candidates) are kept.

Limitations (deliberately stated, not hidden):
  - Replay runs along the RHO-baseline trajectory (states never diverge). In a
    real SIL run the trajectory changes after the first pruning, so this
    measures the pruner's first-order error, not the full SIL effect.
  - The dataset has no "active" flag. Proxy: a request is treated as active in
    window w if it already had label 1 in an earlier window (assigned but not yet
    boarded, i.e. it is still in the payload). Real solver uses active_requests.
  - Error is counted at the FIRST window a request gets label 1 ("first
    assignment"). A request pruned there is not necessarily lost - it may be
    reconsidered in a later window; this script does not model that.
  - ScoringMLP / SIL training is not involved at all.

Usage: python -m rtv_solver.pipeline.analyze_request_pruner_false_prunes
Output: printed tables + CSVs in outputs/request_pruner_false_prune_analysis/
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtv_solver.pipeline.request_pruner_mlp import RequestPrunerMLP

DATASET_DIR = Path("dataset/request_pruning_bi200_ss100")
CKPT = Path(
    "outputs/request_pruner_mlp_bi200_ss100/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/"
    "request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"
)
OUT_DIR = Path("outputs/request_pruner_false_prune_analysis")
URGENT_SLACK = 100.0  # = STEP_SIZE of bi200/ss100 (trip_handler passes STEP_SIZE)
THRESHOLD = 0.5
MIN_RETENTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Class-1 SIL validation instances (training_loop.VALIDATION_FILES). In the pruner dataset
# these live in the "test" split, i.e. unseen by pruner training.
SIL_VAL = ["lc108", "lc109", "lr111", "lr112", "lrc107", "lrc108"]


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = RequestPrunerMLP(
        feature_dim=len(ck["feature_names"]),
        hidden_dim=ck["hidden_dim"],
        num_hidden_layers=ck["num_hidden_layers"],
        dropout=ck["dropout"],
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def find_instance(stem):
    for split in ("train", "val", "test"):
        p = DATASET_DIR / split / f"{stem}.npz"
        if p.exists():
            return split, p
    raise FileNotFoundError(stem)


def all_stems():
    return sorted(p.stem for s in ("train", "val", "test") for p in (DATASET_DIR / s).glob("*.npz"))


def replay_instance(stem, model, ck, feature_index):
    """Returns one DataFrame row per (window, request) with scores and force-keep flags."""
    split, path = find_instance(stem)
    z = np.load(path)
    ids, win, ts, label, feats = z["node_ids"], z["window_index"], z["timestamp"], z["label"], z["features"]

    # Proxy for "active": already label 1 in an earlier window.
    seen_assigned = set()
    active = np.zeros(len(ids), dtype=bool)
    first_assign = np.zeros(len(ids), dtype=bool)
    for w in np.unique(win):
        idx = np.where(win == w)[0]
        newly = []
        for i in idx:
            if ids[i] in seen_assigned:
                active[i] = True
            elif label[i] == 1.0:
                first_assign[i] = True
                newly.append(ids[i])
        seen_assigned.update(newly)

    tul = feats[:, feature_index["time_until_latest_pickup"]]
    urgent = (~active) & (tul <= URGENT_SLACK)

    cols = [feature_index[n] for n in ck["feature_names"]]
    x = (feats[:, cols] - np.array(ck["feature_mean"], dtype=np.float32)) / np.array(ck["feature_std"], dtype=np.float32)
    with torch.no_grad():
        score = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).numpy()

    return pd.DataFrame({
        "instance": stem, "split": split, "window": win, "timestamp": ts, "request": ids,
        "label": label, "score": score, "active": active, "urgent": urgent,
        "first_assign": first_assign, "time_until_latest": tul,
    })


def apply_pruner(df, min_retention):
    """Same steps as RequestPruner.filter_requests, per (instance, window)."""
    pruned = np.zeros(len(df), dtype=bool)
    floor_saved = np.zeros(len(df), dtype=bool)
    for _, g in df.groupby(["instance", "window"]):
        cand = g[~(g["active"] | g["urgent"])]
        if len(cand) == 0:
            continue
        sc = cand["score"].to_numpy()
        keep = sc >= THRESHOLD
        min_keep = math.ceil(min_retention * len(cand))
        floor = np.zeros(len(cand), dtype=bool)
        if int(keep.sum()) < min_keep:
            order = np.argsort(-sc)
            floor[order[:min_keep]] = True
        pos = df.index.get_indexer(cand.index)
        pruned[pos] = ~(keep | floor)
        floor_saved[pos] = floor & ~keep
    return pruned, floor_saved


def summarize(df, pruned, floor_saved):
    cand = ~(df["active"] | df["urgent"])
    fa = df["first_assign"] & cand           # first assignment that pruner could actually drop
    fp = fa & pruned                          # false prunes
    return {
        "windows": df.groupby(["instance", "window"]).ngroups,
        "rows_total": len(df),
        "candidates": int(cand.sum()),
        "pruned": int(pruned.sum()),
        "pruned_share_%": 100 * pruned.sum() / max(cand.sum(), 1),
        "first_assign_candidates": int(fa.sum()),
        "false_prunes": int(fp.sum()),
        "false_prune_rate_%": 100 * fp.sum() / max(fa.sum(), 1),
        "pruned_but_label0": int((pruned & (df["label"] == 0)).sum()),
        "floor_saved_first_assign": int((floor_saved & fa).sum()),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, ck = load_model()
    full_names = list(np.load(next((DATASET_DIR / "test").glob("*.npz")), allow_pickle=True).files)  # sanity only
    # Column order of `features` in the npz = manifest feature_names (15 cols).
    import json
    manifest = json.load(open(DATASET_DIR / "manifest.json"))
    feature_index = {n: i for i, n in enumerate(manifest["feature_names"])}

    frames = [replay_instance(s, model, ck, feature_index) for s in all_stems()]
    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv(OUT_DIR / "replay_rows_scores.csv", index=False)

    groups = {
        "SIL-val (6 class-1 val instances, unseen by pruner)": df_all[df_all["instance"].isin(SIL_VAL)].reset_index(drop=True),
        "all dataset instances": df_all.reset_index(drop=True),
    }
    rows = []
    for gname, g in groups.items():
        for mr in MIN_RETENTIONS:
            pruned, floor_saved = apply_pruner(g, mr)
            s = summarize(g, pruned, floor_saved)
            s.update({"group": gname, "min_retention": mr})
            rows.append(s)
    res = pd.DataFrame(rows)
    order = ["group", "min_retention", "windows", "candidates", "pruned", "pruned_share_%",
             "first_assign_candidates", "false_prunes", "false_prune_rate_%",
             "pruned_but_label0", "floor_saved_first_assign"]
    res = res[order]
    res.to_csv(OUT_DIR / "false_prune_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.1f}".format):
        print(res.to_string(index=False))

    # 2026-09-21: where do the false prunes come from? Breakdown by instance, window
    # position, slack to latest pickup and score distance to the threshold. Done
    # at min_retention 0.3 (old hardcoded default) and 0.5.
    g = groups["SIL-val (6 class-1 val instances, unseen by pruner)"]
    for mr in (0.3, 0.5):
        pruned, floor_saved = apply_pruner(g, mr)
        d = g.copy()
        d["pruned"] = pruned
        cand = ~(d["active"] | d["urgent"])
        d["fa_cand"] = d["first_assign"] & cand
        d["false_prune"] = d["fa_cand"] & d["pruned"]
        fa = d[d["fa_cand"]].copy()
        print(f"\n=== SIL-val, min_retention={mr}, threshold={THRESHOLD}: breakdown of first-assignment candidates ===")

        def tab(by, name):
            t = fa.groupby(by, observed=True).agg(
                first_assign=("request", "size"), false_prunes=("false_prune", "sum"))
            t["rate_%"] = 100 * t["false_prunes"] / t["first_assign"]
            print(f"\n-- by {name}")
            print(t.to_string(float_format=lambda v: f"{v:.1f}"))
            return t

        tab("instance", "instance")
        fa["window_pos"] = pd.cut(fa["window"], [-1, 0, 1, 3, 100], labels=["w0", "w1", "w2-3", "w4+"])
        tab("window_pos", "window index")
        fa["slack_bin"] = pd.cut(fa["time_until_latest"], [-1e9, 100, 200, 400, 800, 1e9],
                                  labels=["<=100 (urgent, never pruned)", "100-200", "200-400", "400-800", ">800"])
        tab("slack_bin", "time_until_latest_pickup [s]")
        fp = fa[fa["false_prune"]]
        if len(fp):
            bins = pd.cut(fp["score"], [0, 0.1, 0.25, 0.4, THRESHOLD], right=False)
            print("\n-- score of the false prunes (threshold 0.5): closer to 0.5 = closer call")
            print(bins.value_counts().sort_index().to_string())
        # Which false prunes were pruned only because of low score in a window where the floor did NOT rescue them?
        print(f"\nfirst assignments rescued by the floor (would have been pruned): "
              f"{int((floor_saved & d['fa_cand'].to_numpy()).sum())}")


if __name__ == "__main__":
    main()
