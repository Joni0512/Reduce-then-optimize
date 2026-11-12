"""
2026-11-06: score distribution + threshold sweep of the window-specific Request Pruner for batch_interval=40/step_size=10 (MLP request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03,
min_retention_fraction 0.3, K=2, URGENT_SLACK=10 = step_size). Reason: the RHO runs showed 0 % requests pruned at t=0.2/0.3/0.4 and 16 % at t=0.5 -> where do the scores sit?
Method = the Rule A evaluation of select_request_pruner_by_vehicle_loss.py (same replay via analyze_request_pruner_false_prunes: along the RHO-baseline trajectory, proxy for "active"
= already label 1 in an earlier window), on the 8 pruner-val instances only (never the 12 test/SIL-val instances -> no leakage):
lc107 lc206 lr109 lr110 lr208 lr209 lrc106 lrc206. lost_1st / lost_anyK: share of the vehicles (window 0, 51 vehicles) whose first / one of the first K=2 expert requests would be pruned.
false-prune rate: share of first-assignment requests (RHO label) that the pruner drops (summarize() of the analysis module).
Run: ./venv/bin/python3 scripts/analysis/make_request_pruner_bi40ss10_score_threshold_chart.py
Output: figures_export/Request Pruner RHO/request_pruner_bi40ss10_score_threshold.{png,pdf}, outputs/request_pruner_bi40ss10_score_threshold.csv
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from rtv_solver.pipeline import analyze_request_pruner_false_prunes as A  # noqa: E402
from rtv_solver.pipeline.build_request_pruner_prefix_dataset import expert_by_vehicle  # noqa: E402

K, MIN_RET = 2, 0.3
THRESHOLDS = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
VAL = ["lc107", "lc206", "lr109", "lr110", "lr208", "lr209", "lrc106", "lrc206"]
CK = REPO / "outputs/request_pruner_mlp_bi40_ss10/request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03/request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03_best_val_f3.pt"
OUT = REPO / "figures_export/Request Pruner RHO"
OUT.mkdir(parents=True, exist_ok=True)
CB, CP, CR, CC, CG = "#1E2761", "#E8A33D", "#276575", "#6C4A9E", "#888888"

A.DATASET_DIR, A.URGENT_SLACK, A.CKPT = REPO / "dataset/request_pruning_bi40_ss10", 10.0, CK
manifest = json.load(open(A.DATASET_DIR / "manifest.json"))
fidx = {n: i for i, n in enumerate(manifest["feature_names"])}
model, ck = A.load_model()
dfs = {s: A.replay_instance(s, model, ck, fidx) for s in VAL}
allc = pd.concat(dfs.values(), ignore_index=True)
cand_mask = ~(allc["active"] | allc["urgent"])
sc = allc.loc[cand_mask, "score"].to_numpy()
print("candidates", len(sc), "of rows", len(allc), "quantiles", {q: round(float(np.quantile(sc, q)), 3) for q in (0, .05, .25, .5, .75, .9)})

rows = []
for t in THRESHOLDS:
    A.THRESHOLD = t
    cand = pruned = veh = lost1 = lostK = fa = fp = 0
    for s, df in dfs.items():
        pr, fl = A.apply_pruner(df, MIN_RET)
        sm = A.summarize(df, pr, fl)
        fa += sm["first_assign_candidates"]; fp += sm["false_prunes"]
        c = ~(df["active"] | df["urgent"])
        cand += int(c.sum()); pruned += int(pr.sum())
        w0 = df[df.window == 0].reset_index(drop=True)
        pw0, _ = A.apply_pruner(w0, MIN_RET)
        flag = dict(zip(w0.request, pw0)); batch = set(w0.request)
        for seq in expert_by_vehicle(s).values():
            proj = [r for r in seq if r in batch]
            if proj:
                veh += 1; lost1 += int(flag[proj[0]]); lostK += int(any(flag[r] for r in proj[:K]))
    rows.append(dict(threshold=t, candidates=cand, vehicles=veh, prune_rate=100 * pruned / cand, false_prune_rate=100 * fp / max(fa, 1),
                     lost_1st=100 * lost1 / max(veh, 1), lost_anyK=100 * lostK / max(veh, 1)))
res = pd.DataFrame(rows)
res.to_csv(REPO / "outputs/request_pruner_bi40ss10_score_threshold.csv", index=False)
print(res.round(1).to_string())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.6))
a1.hist(sc, bins=np.linspace(0.45, 1.0, 56), color=CR, zorder=3)
for t, c in [(0.3, CG), (0.4, CG), (0.5, CP), (0.55, CC), (0.6, CC)]:
    a1.axvline(t, color=c, ls="--", lw=1.2, zorder=4)
    a1.text(t, a1.get_ylim()[1] * 0.97, f"t={t}", rotation=90, va="top", ha="right", fontsize=8.5, color=c)
a1.set_xlabel("Request Pruner score (candidates only: not active, not urgent)")
a1.set_ylabel("Number of candidate rows (8 val instances, all windows)")
a1.set_title(f"Score distribution ({len(sc)} candidates; median {np.median(sc):.3f}, min {sc.min():.3f})", fontsize=10.5)

a2.plot(res.threshold, res.prune_rate, "-o", color=CB, label="Pruning rate (% of candidates)", zorder=3)
a2.plot(res.threshold, res.false_prune_rate, "--s", color=CG, label="False-prune rate (% of first-assignment requests)", zorder=3)
a2.plot(res.threshold, res.lost_1st, "-^", color=CP, label="lost_1st (% of vehicles, window 0)", zorder=3)
a2.plot(res.threshold, res.lost_anyK, "-v", color=CC, label=f"lost_anyK, K={K} (% of vehicles, window 0)", zorder=3)
a2.axhline(30, color="#bbbbbb", ls=":", lw=1.2)
a2.text(0.3, 31, "Rule A minimum pruning rate (30 %)", fontsize=8.5, color="#777777")
a2.set_xlabel("Threshold t")
a2.set_ylabel("%")
a2.set_title("Threshold sweep (min_retention_fraction 0.3)", fontsize=10.5)
a2.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
for a in (a1, a2):
    a.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("Request Pruner, batch_interval=40/step_size=10: score distribution and threshold sweep (replay along the RHO baseline)", fontsize=12)
fig.text(0.5, 0.012, "MLP request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03 (best_val_f3), pruner-val instances lc107 lc206 lr109 lr110 lr208 lr209 lrc106 lrc206; URGENT_SLACK=10; 51 vehicles in window 0 (1 vehicle = 2 pp).\n"
         "Replay keeps the RHO trajectory (no feedback of pruning on later windows); 'active' is a proxy (label 1 in an earlier window). Source: scripts/analysis/make_request_pruner_bi40ss10_score_threshold_chart.py -> outputs/request_pruner_bi40ss10_score_threshold.csv.",
         ha="center", fontsize=7.6, color="#666666")
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"request_pruner_bi40ss10_score_threshold.{ext}", dpi=170)
