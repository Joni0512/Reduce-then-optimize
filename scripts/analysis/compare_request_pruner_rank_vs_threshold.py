"""
2026-10-21: Offline test - rank-based Request Pruning vs. threshold pruning (c2, batch_interval=200/step_size=100, prefix labels).

Question: at the SAME overall pruning rate, does pruning the lowest q % of the candidate scores PER WINDOW (rank) lose fewer
reference-route targets than pruning by a fixed score threshold (what RequestPruner does today)?
Reason: idea 2 in docs/Pruning_Improvement_Ideas.md; the pruning rate of a fixed threshold varies a lot between checkpoints / seeds
(Val-ROC-AUC 0.63-0.84 for the same HPs), a rank rule fixes the rate and only depends on the ranking quality.

Design (no solver / pruner code is changed):
  - Models: the 13-feature base MLP of compare_request_pruner_knn_features.py (same trainer and HPs as the existing bi200/ss100
    winner: hidden 32, 1 layer, dropout 0.2, pos_weight 1.0, lr 0.01), 20 seeds (0..19).
  - Replay on the 8 pruner-VAL instances (never the 12 test = SIL-val instances), same active/urgent proxy and the same metrics
    as select_request_pruner_by_vehicle_loss.py: pruning rate = pruned / candidates (candidates = not active, not urgent),
    lost_1st / lost_anyK = vehicles losing their true first / any of first K=2 targets in window 0 (53 vehicles, one = 1.9 pp),
    prefix_pruned = share of prefix-label requests pruned over all windows.
  - Rank rule: per (instance, window), candidates sorted by score ascending, the lowest round(q * n_candidates) are pruned.
    The retention floor (0.3) cannot bind for q <= 0.5. Realised global rate differs slightly from q (rounding in small windows).
  - Threshold rule: thresholds 0.05..0.95, min_retention 0.3 (as in the existing replay); metrics linearly interpolated at the
    pruning rate the rank rule actually realised for that seed -> equal-rate comparison, per seed, then paired over seeds.

Run: ./venv/bin/python3 scripts/analysis/compare_request_pruner_rank_vs_threshold.py [--seeds 0 1 ... 19]
Output: outputs/request_pruner_rank_vs_threshold.csv (per seed and q), summary printed.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_request_pruner_knn_features as C  # noqa: E402  (training/replay helpers, same HPs and instances)
from rtv_solver.pipeline import analyze_request_pruner_false_prunes as A  # noqa: E402
from rtv_solver.pipeline.build_request_pruner_prefix_dataset import K, expert_by_vehicle  # noqa: E402
from rtv_solver.pipeline.train_request_pruner import train_one_config  # noqa: E402

QS = (0.30, 0.40, 0.50)


def apply_rank(df, q):
    """Per (instance, window): prune the lowest round(q * n_candidates) candidate scores. Same shape of result as A.apply_pruner."""
    pruned = np.zeros(len(df), dtype=bool)
    for _, g in df.groupby(["instance", "window"]):
        cand = g[~(g["active"] | g["urgent"])]
        n_prune = int(round(q * len(cand)))
        if n_prune == 0:
            continue
        order = np.argsort(cand["score"].to_numpy(), kind="stable")
        pos = df.index.get_indexer(cand.index)
        pruned[pos[order[:n_prune]]] = True
    return pruned


def metrics(scored, prune_fn):
    cand = pruned = veh = lost1 = lostK = pre = pre_pruned = 0
    for s, d in scored.items():
        pr = prune_fn(d)
        c = ~(d["active"] | d["urgent"])
        cand += int(c.sum()); pruned += int(pr.sum())
        pre += int(d["prefix"].sum()); pre_pruned += int((d["prefix"].to_numpy() & pr).sum())
        w0 = d[d.window == 0].reset_index(drop=True)
        pw0 = prune_fn(w0)
        flag = dict(zip(w0.request, pw0))
        batch = set(w0.request)
        for seq in expert_by_vehicle(s).values():
            proj = [r for r in seq if r in batch]
            if proj:
                veh += 1; lost1 += int(flag[proj[0]]); lostK += int(any(flag[r] for r in proj[:K]))
    return dict(prune_rate=100 * pruned / max(cand, 1), lost_1st=100 * lost1 / max(veh, 1),
                lost_anyK=100 * lostK / max(veh, 1), prefix_pruned=100 * pre_pruned / max(pre, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    a = ap.parse_args()

    tr, fill = C.load_split("train")
    va, _ = C.load_split("val", fill)
    te, _ = C.load_split("test", fill)
    old_model, old_ck = C.load_model(C.OLD_CKPT)
    A.CKPT = C.OLD_CKPT
    replay = C.prepare_replay(old_model, old_ck, fill)

    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="rank_pruner_"))
    for seed in a.seeds:
        res = train_one_config(tr["base"], va["base"], te["base"], hidden_dim=32, num_hidden_layers=1, dropout=0.2,
                               pos_weight=1.0, learning_rate=0.01, seed=seed, verbose=False, save_checkpoint=True,
                               output_dir=tmp / f"seed{seed}")
        model, ck = C.load_model(res["best_model_path"], feature_dim=tr["base"]["features"].shape[1])
        mean, std = np.array(ck["feature_mean"], dtype=np.float32), np.array(ck["feature_std"], dtype=np.float32)
        scored = {}
        for s, (df, mats) in replay.items():
            with torch.no_grad():
                sc = torch.sigmoid(model(torch.tensor((mats["base"] - mean) / std, dtype=torch.float32))).numpy()
            d = df.copy(); d["score"] = sc; scored[s] = d
        # threshold curve (existing rule, min_retention 0.3)
        curve = []
        for t in C.THRESHOLDS:
            A.THRESHOLD = t
            m = metrics(scored, lambda d: A.apply_pruner(d, C.MIN_RETENTION)[0])
            curve.append(dict(threshold=t, **m))
        curve = pd.DataFrame(curve).sort_values("prune_rate")
        for q in QS:
            r = metrics(scored, lambda d: apply_rank(d, q))
            row = dict(seed=seed, q=q, rank_prune_rate=r["prune_rate"])
            for m in ("lost_1st", "lost_anyK", "prefix_pruned"):
                row[f"rank_{m}"] = r[m]
                row[f"thr_{m}"] = float(np.interp(r["prune_rate"], curve.prune_rate, curve[m]))
            rows.append(row)
        print(f"  seed {seed}: val_roc_auc={res['metrics']['val'][0.5]['roc_auc']:.3f}")

    df = pd.DataFrame(rows)
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/request_pruner_rank_vs_threshold.csv", index=False)

    print(f"\nEqual-rate comparison over {df.seed.nunique()} seeds (rank rule vs threshold interpolated at the realised rate); lower = better")
    for m in ("prefix_pruned", "lost_1st", "lost_anyK"):
        print(f"\n== {m}")
        for q in QS:
            g = df[df.q == q]
            d = g[f"rank_{m}"] - g[f"thr_{m}"]
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            print(f"q={int(100 * q)}% (realised rate {g.rank_prune_rate.mean():.1f}%): threshold {g[f'thr_{m}'].mean():.2f}±{g[f'thr_{m}'].std():.2f} | "
                  f"rank {g[f'rank_{m}'].mean():.2f}±{g[f'rank_{m}'].std():.2f} | diff rank-thr {d.mean():+.2f} (t={t:+.2f}, rank better in {(d < 0).sum()}/{len(d)}, median {d.median():+.2f})")


if __name__ == "__main__":
    main()
