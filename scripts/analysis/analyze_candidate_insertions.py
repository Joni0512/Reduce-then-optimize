"""
2026-11-18: analysis of the logged candidate insertions (rho_candidate_insertions_run.py): which simple per-candidate score separates the (request, vehicle) pairs that the RHO ILP actually
chooses from the ones it does not choose? Scores (lower = better):
  A  veh_to_pickup      travel time vehicle (current/next node) -> pickup of the request                  ("distance request - vehicle")
  B  excess             added_cost - direct_time = extra time beyond serving the request itself          ("how much longer does the trip get" beyond the minimum)
  C  added_cost / best  added_cost relative to the cheapest vehicle for the SAME request in the SAME window (>= 1)
(added_cost = new optimal route cost - cost of the vehicle's current route = TripCost.cost, i.e. what the ILP objective uses. For an idle vehicle added_cost = direct_time + veh_to_pickup.)

Decision candidates (to avoid the bias of committed requests, which the ILP must keep at their vehicle every window and which are logged as 'chosen' in every later window):
only windows up to and INCLUDING the first window in which the request is chosen at all ("first assignment"); never-assigned requests keep all their windows (all not chosen).
Per class (1 / 2) and heuristic: remove candidates with a score above a threshold -> x = share of candidates removed, y = share of FIRST-ASSIGNMENT choices that survive (recall).
Limits: 1 run per instance (RHO noise), only feasible single-request insertions, requests that are chosen later through a shared trip count as chosen for that vehicle, a request first
assigned to vehicle X and re-assigned later to Y is only judged at its first assignment.

Run: ./venv/bin/python3 scripts/analysis/analyze_candidate_insertions.py
Source: outputs/candidate_insertions_C2_c2_bi40_ss10.csv (+ _runs.csv)
Output (figures_export/Request Pruner RHO/): candidate_insertions_bi40_ss10.{png,pdf}, ..._table.csv, ..._table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse

import numpy as np
import pandas as pd

OUT = Path("figures_export/Request Pruner RHO")
OUT.mkdir(parents=True, exist_ok=True)
# 2026-11-19: window as arguments (was hard-wired to batch_interval=40/step_size=10) so the 100/25 run can be analysed the same way; default 40/10 gives the old output unchanged
ap = argparse.ArgumentParser()
ap.add_argument("--bi", default="40")
ap.add_argument("--ss", default="10")
A = ap.parse_args()
BI, SS = A.bi, A.ss
SRC = f"outputs/candidate_insertions_C2_c2_bi{BI}_ss{SS}.csv"
d = pd.read_csv(SRC)
d["excess"] = d.added_cost - d.direct_time
d["cls"] = d.instance.str.extract(r"^[a-z]+(\d)")[0]
key = ["instance", "request_id"]
first = d[d.chosen == 1].groupby(key).window.min().rename("first_w")
d = d.join(first, on=key)
dc = d[d.first_w.isna() | (d.window <= d.first_w)].copy()  # decision candidates
dc["best_added"] = dc.groupby(key + ["window"]).added_cost.transform("min")
dc["rel_best"] = dc.added_cost / dc.best_added.clip(lower=1e-9)
dc["nveh"] = dc.groupby(key + ["window"]).vehicle_id.transform("count")
fa = dc[(dc.chosen == 1) & (dc.window == dc.first_w)]  # first-assignment choices
print("all candidates", len(d), "| decision candidates", len(dc), "| first-assignment choices", len(fa), "| requests never assigned", dc[dc.first_w.isna()].request_id.nunique() if False else int(dc.groupby(key).first_w.first().isna().sum()))

HEUR = {"veh_to_pickup": "A: travel time vehicle -> pickup", "excess": "B: extra time = added_cost - direct_time", "rel_best": "C: added_cost / best vehicle's added_cost"}
COL = {"veh_to_pickup": "#E8A33D", "excess": "#1E2761", "rel_best": "#2A9D8F"}


def curve(x, score):
    """for thresholds at quantiles of the score over all decision candidates: (share removed, share of first-assignment choices kept)"""
    s = x[score].to_numpy()
    ch = x[(x.chosen == 1) & (x.window == x.first_w)][score].to_numpy()
    ths = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
    removed = np.array([(s > t).mean() for t in ths]) * 100
    kept = np.array([(ch <= t).mean() for t in ths]) * 100
    return removed, kept


def recall_at_removed(x, score, removed_target):
    r, k = curve(x, score)
    ok = r >= removed_target
    return k[ok].max() if ok.any() else float("nan")  # best recall that removes at least `removed_target` %


rows = []
for c in ("1", "2"):
    x = dc[dc.cls == c]
    n_fa = int(((x.chosen == 1) & (x.window == x.first_w)).sum())
    f = fa[fa.cls == c]
    cheapest = 100 * (f.rel_best <= 1 + 1e-9)[f.nveh > 1].mean()
    near = f.groupby(key + ["window"]).veh_to_pickup.first()  # placeholder to keep groupby alignment simple
    rank_pick = x.groupby(key + ["window"]).veh_to_pickup.rank(method="min")
    nearest = 100 * (rank_pick[(x.chosen == 1) & (x.window == x.first_w) & (x.nveh > 1)] == 1).mean()
    for h in HEUR:
        rows.append(dict(cls=f"Class {c}", heuristic=HEUR[h], n_candidates=len(x), n_first_assign=n_fa, cheapest_vehicle_chosen_pct=cheapest, nearest_vehicle_chosen_pct=nearest,
                         recall_at_50_removed=recall_at_removed(x, h, 50), recall_at_70_removed=recall_at_removed(x, h, 70), recall_at_30_removed=recall_at_removed(x, h, 30)))
T = pd.DataFrame(rows)
T.round(1).to_csv(OUT / f"candidate_insertions_bi{BI}_ss{SS}_table.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(18, 5.8))
for k, (c, ls) in enumerate((("1", "-"), ("2", "--"))):
    x = dc[dc.cls == c]
    for h in HEUR:
        r, kp = curve(x, h)
        ax[1 + k].plot(r, kp, color=COL[h], lw=2.2, label=HEUR[h])
    ax[1 + k].plot([0, 100], [100, 0], color="#999999", lw=1, ls=":", label="random removal")
    ax[1 + k].set_title(f"({'bc'[k]}) Class {c}: first-assignment choices kept vs. candidates removed", fontsize=9.5)
    ax[1 + k].set_xlabel("Share of candidate (request, vehicle) pairs removed (%)", fontsize=9.5)
    ax[1 + k].set_ylabel("Share of first-assignment choices kept (%)", fontsize=9.5)
    ax[1 + k].set_xlim(0, 100)
    ax[1 + k].set_ylim(0, 102)
for c, col_, ls in (("1", "#1E2761", "-"), ("2", "#E8A33D", "-")):
    x = dc[dc.cls == c]
    ch = x[(x.chosen == 1) & (x.window == x.first_w)].excess.clip(-20, 140)
    nc = x[x.chosen == 0].excess.clip(-20, 140)
    for v, ls2, lab in ((ch, "-", "chosen (first assignment)"), (nc, "--", "not chosen")):
        s = np.sort(v.to_numpy())
        ax[0].plot(s, np.arange(1, len(s) + 1) / len(s) * 100, color=col_, ls=ls2, lw=2, label=f"Class {c}: {lab}")
ax[0].set_title("(a) Extra time = added_cost - direct_time (ECDF, clipped to [-20, 140])", fontsize=9.5)
ax[0].set_xlabel("Extra time beyond the request's own direct time (time units)", fontsize=9.5)
ax[0].set_ylabel("Share of candidates (%)", fontsize=9.5)
ax[0].legend(fontsize=7.5, frameon=False, loc="lower right")
for a in ax:
    a.grid(True, linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
ax[1].legend(fontsize=7.5, frameon=False, loc="lower left")
fig.suptitle(f"RHO candidate insertions: do 'distance to vehicle' and 'extra trip time' separate the chosen (request, vehicle) pairs? batch_interval={BI}/step_size={SS}, C=2, max_cardinality=2", fontsize=11)
fig.text(0.5, 0.01, f"Config: RHO --mode offline, batch_interval={BI}, step_size={SS}, max_cardinality 2, C=2 on board, no total limit, no pruner, 14 Li&Lim test instances, 1 run per instance, --log_candidate_insertions True.\n"
         "Only feasible single-request insertions; decision candidates = windows up to the request's first assignment (committed requests excluded); score thresholds on quantiles of all decision candidates of the class.\n"
         f"Source: {SRC}; analysis: scripts/analysis/analyze_candidate_insertions.py",
         ha="center", fontsize=7, color="#666666")
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"candidate_insertions_bi{BI}_ss{SS}.{ext}", dpi=200)
plt.close(fig)

lines = []
for _, r in T.iterrows():
    lines.append(f"{r.cls} & {r.heuristic.replace('_', chr(92) + '_')} & {r.recall_at_30_removed:.0f} & {r.recall_at_50_removed:.0f} & {r.recall_at_70_removed:.0f} \\\\")
tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
       f"\\caption{{RHO candidate insertions (\\texttt{{batch\\_interval}}={BI}, \\texttt{{step\\_size}}={SS}, $C{{=}}2$, cardinality 2, 14 instances): share of first-assignment choices (\\%) that survive when the "
       "highest-scoring 30/50/70\\,\\% of the (request, vehicle) candidates are removed by a simple score. Higher is better; random removal would give 70/50/30.}\n"
       "\\begin{tabular}{llccc}\n\\toprule\nClass & Score & 30\\,\\% removed & 50\\,\\% removed & 70\\,\\% removed \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(OUT / f"candidate_insertions_bi{BI}_ss{SS}_table.tex").write_text(tex)
with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 50):
    print(T.round(1).to_string(index=False))
