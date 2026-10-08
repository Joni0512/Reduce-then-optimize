"""2026-10-07: why batch_interval=40 beats 10 on dense short-horizon instances (lr101, lr102, lrc101).
RHO offline, step_size=10, card 2, KEEP_ACTIVE=True, seed 42, 1 run per config.
Sources: outputs/lookahead_dense/steps_bi{10,40}.json (outputs/lookahead_dense/record_ilp_steps.py, monkeypatch only),
solutions/li_lim/manifests/<inst>.json (travel_time_matrix, depot node)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
INSTS, BIS = ["lr101", "lr102", "lrc101"], [10, 40]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"served": "#1baf7a", "never": "#eb6834", "rej": "#eda100", "slack10": "#eb6834", "slack40": "#2a78d6", "tt": "#8a8984"}
lost, slack, travel = {}, {}, {}
for bi in BIS:
    S = json.load(open(root / f"outputs/lookahead_dense/steps_bi{bi}.json"))
    for inst in INSTS:
        d = S[inst]; served = set(d["served"]); seen = {}
        m = json.load(open(root / f"solutions/li_lim/manifests/{inst}.json"))
        M = np.array(m["travel_time_matrix"]); dep = m["depot"]["node_id"]
        R = {int(q["booking_id"]): q for q in m["requests"]}
        for st in d["steps"]:
            feas = {int(k) for k in st["feasible"]}
            for r, (e, l, a) in st["requests"].items():
                s = seen.setdefault(int(r), dict(first=st["t"], latest=l, feas=0)); s["feas"] += int(r) in feas
        lr = [r for r in d["all_requests"] if r not in served]
        never = sum(1 for r in lr if r not in seen or seen[r]["feas"] == 0)
        lost[(inst, bi)] = (len(served), never, len(lr) - never)
        slack[(inst, bi)] = np.median([s["latest"] - s["first"] for s in seen.values()])
        travel[inst] = np.median([M[dep][R[r]["pickup_pt"]["node_id"]] for r in seen])
        print(inst, bi, lost[(inst, bi)], slack[(inst, bi)], travel[inst])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(INSTS)); w = 0.36
for k, bi in enumerate(BIS):
    xs = x + (k - 0.5) * w; sv = np.array([lost[(i, bi)][0] for i in INSTS]); nv = np.array([lost[(i, bi)][1] for i in INSTS]); rj = np.array([lost[(i, bi)][2] for i in INSTS])
    a1.bar(xs, sv, w, color=COL["served"], label="served" if k == 0 else None, zorder=2)
    a1.bar(xs, nv, w, bottom=sv, color=COL["never"], label="lost: never feasible" if k == 0 else None, zorder=2)
    a1.bar(xs, rj, w, bottom=sv + nv, color=COL["rej"], label="lost: feasible, rejected" if k == 0 else None, zorder=2)
    for xi in xs: a1.text(xi, -3.5, f"{bi}", ha="center", fontsize=7, color=INK2)
a1.set_xticks(x, INSTS, fontsize=8); a1.tick_params(axis="x", pad=14); a1.set_ylabel("requests", fontsize=8); a1.set_xlabel("batch_interval 10 | 40 per instance (step_size=10)", fontsize=7.5)  # 2026-10-07: spell out batch_interval
a1.set_title("Fate of every request", loc="left", fontsize=9); a1.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=1)
a1.set_ylim(0, 62)
w2 = 0.26
a2.bar(x - w2, [slack[(i, 10)] for i in INSTS], w2, color=COL["slack10"], label="time left when first seen, batch_interval=10", zorder=2)
a2.bar(x, [slack[(i, 40)] for i in INSTS], w2, color=COL["slack40"], label="time left when first seen, batch_interval=40", zorder=2)
a2.bar(x + w2, [travel[i] for i in INSTS], w2, color=COL["tt"], label="depot -> pickup travel", zorder=2)
a2.set_xticks(x, INSTS, fontsize=8); a2.set_ylabel("time units (median)", fontsize=8)
a2.set_title("Time left when first seen vs travel", loc="left", fontsize=9); a2.legend(fontsize=7, loc="upper left")
a2.set_ylim(0, 85)
for a in (a1, a2):
    a.yaxis.grid(True, color=GRID, zorder=0); a.set_axisbelow(True); a.tick_params(labelsize=8)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(here / f"rho_lookahead_dense_lost_causes.{ext}", dpi=220)
