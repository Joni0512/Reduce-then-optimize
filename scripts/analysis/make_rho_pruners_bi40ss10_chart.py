"""
2026-11-04/05: RHO only (no learning) - Baseline vs. Pair Pruner vs. Request Pruner vs. Combined, one common threshold T for both pruners (0.5 = local run, 0.2 = cluster run).
Generalised 2026-11-05: --t 0.5 --env local (default) or --t 0.2 --env cluster.
Setup (scripts/analysis/rho_pair_pruner_runtime.py, --variants base pair0.5 req0.5 comb0.5_0.5): RHO offline, C=2 on board (unit-demand manifest copy), no total limit,
max_cardinality=2, batch_interval=40/step_size=10, 14 Li&Lim instances (7 class 1, 7 class 2), 2 repetitions, variants run interleaved and one at a time (no parallel jobs).
Pair Pruner = GNNv1 rgnn_mixed_c2_pw2_v1_final (not window-specific); Request Pruner = MLP request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03 trained for 40/10 on RHO labels
(min_retention_fraction 0.3); Combined = both with the same threshold T.
Service rate = served requests / len(requests) of solutions/li_lim/manifests/<instance>.json, pooled over the 14 instances (sum served / sum requests), per repetition.
Runtime = wall-clock of the solver subprocess (time.monotonic); trip generation = sum over windows of the solver's own "feasible trip costs generated" time.
Sources: outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10_t0.5_local.csv (local, written 2026-11-02 00:18), ..._cluster_t0.2.csv (LRZ cluster job 5707352, own baseline in the same job;
runtimes are only comparable within one file).

Run: ./venv/bin/python3 scripts/analysis/make_rho_pruners_bi40ss10_chart.py [--t 0.2 --env cluster]
Output: figures_export/Request Pruner RHO/rho_pruners_bi40ss10_t<T>_<env>_{summary,per_instance}.{png,pdf}
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--t", default="0.5")
ap.add_argument("--env", default="local", choices=["local", "cluster"])
# 2026-11-06: --ts 0.3 0.4 = one CSV with several thresholds (outputs/..._t0.3_0.4_local.csv, own baseline in the same run); each threshold t uses pair<t>, req<t>, comb<t>_<t>.
ap.add_argument("--ts", nargs="+", default=None)
A = ap.parse_args()
T, ENV = A.t, A.env
TS = A.ts if A.ts else [T]
if A.ts:
    T = "_".join(TS)
TAG = f"t{T}_{ENV}"
CSV_NAME = f"rho_pair_pruner_runtime_C2_c2_bi40_ss10_{TAG}.csv" if ENV == "local" else f"rho_pair_pruner_runtime_C2_c2_bi40_ss10_cluster_t{T}.csv"
ENV_TXT = "Local run, one solver process at a time" if ENV == "local" else "LRZ cluster job 5707352 (own baseline in the same job), one solver process at a time"

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures_export/Request Pruner RHO"
OUT.mkdir(parents=True, exist_ok=True)
CSV = REPO / "outputs" / CSV_NAME
CB, CP, CR, CC = "#1E2761", "#E8A33D", "#276575", "#6C4A9E"
ORDER, LABEL, COL = ["base"], {"base": "No pruner"}, {"base": CB}
for t in TS:  # 2026-11-06: loop over thresholds so the same code serves single- and multi-threshold CSVs
    for v, lab, c in [(f"pair{t}", f"Pair Pruner\nt={t}", CP), (f"req{t}", f"Request Pruner\nt={t}", CR), (f"comb{t}_{t}", f"Combined\nreq {t} + pair {t}", CC)]:
        ORDER.append(v); LABEL[v] = lab; COL[v] = c
THR_TXT = " / ".join(TS)

d = pd.read_csv(CSV)
d = d[d.status == "ok"]
n_req = {s: len(json.load(open(REPO / "solutions/li_lim/manifests" / f"{s}.json"))["requests"]) for s in d.instance.unique()}
d["n_req"] = d.instance.map(n_req)

# per variant and repetition: pooled sums over the 14 instances
g = d.groupby(["variant", "rep"]).agg(wall=("wall_s", "sum"), rtv=("rtv_s", "sum"), served=("served", "sum"), n_req=("n_req", "sum"),
                                      pk=("pairs_kept", "sum"), pt=("pairs_total", "sum"), rk=("req_kept", "sum"), rt=("req_total", "sum")).reset_index()
g["sr"] = 100 * g.served / g.n_req
m = g.groupby("variant").agg(wall=("wall", "mean"), rtv=("rtv", "mean"), sr=("sr", "mean"), served=("served", "mean"), n_req=("n_req", "first"),
                             pk=("pk", "sum"), pt=("pt", "sum"), rk=("rk", "sum"), rt=("rt", "sum")).loc[ORDER]
base_wall, base_rtv, base_sr = m.loc["base", "wall"], m.loc["base", "rtv"], m.loc["base", "sr"]


def kept_line(v):
    r = m.loc[v]
    parts = []
    if r.rt > 0:
        parts.append(f"{100 * r.rk / r.rt:.0f} % req. kept")  # 2026-11-05: shortened, the long label overlapped the neighbouring tick label
    if r.pt > 0:
        parts.append(f"{100 * r.pk / r.pt:.0f} % pairs kept")
    return "\n".join(parts) if parts else "nothing pruned"


# ---------------------------------------------------------------- figure 1: summary (3 panels)
fig, axs = plt.subplots(1, 3, figsize=(16 if len(ORDER) <= 4 else 21, 5.8))
x = np.arange(len(ORDER))
xl = [f"{LABEL[v]}\n{kept_line(v)}" for v in ORDER]
FS = 8 if len(ORDER) <= 4 else 6.8  # 2026-11-06: smaller tick/value labels for 7 bars
for ax, key, ylab, ttl in [(axs[0], "sr", "Service rate (%, pooled over 14 instances)", "Service rate (dots = the 2 repetitions)"),
                           (axs[1], "wall", "Wall-clock (s, sum over 14 instances)", "Total wall-clock (number = speed-up vs. no pruner)"),
                           (axs[2], "rtv", "Trip generation (s, sum over 14 instances)", "Trip generation (number = speed-up vs. no pruner)")]:
    vals = [m.loc[v, key] for v in ORDER]
    ax.bar(x, vals, color=[COL[v] for v in ORDER], width=0.62, zorder=3)
    for i, v in enumerate(ORDER):  # the two repetitions as dots
        reps = g[g.variant == v][key if key != "sr" else "sr"].values
        ax.scatter([i] * len(reps), reps, color="white", edgecolor="black", s=22, zorder=4, linewidths=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xl, fontsize=FS)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(ttl, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
sr_lo = np.floor(min(m.sr.min(), g.sr.min()) - 1)
axs[0].set_ylim(sr_lo, min(100, np.ceil(g.sr.max() + 1.5)))
axs[0].axhline(base_sr, color="#888888", ls="--", lw=1, zorder=2)
axs[0].set_title(f"Service rate (dashed = no pruner; y-axis starts at {sr_lo:.0f} %; dots = 2 repetitions)", fontsize=9.5)
pad0 = (axs[0].get_ylim()[1] - axs[0].get_ylim()[0]) * 0.015
for i, v in enumerate(ORDER):
    axs[0].text(i, g[g.variant == v].sr.max() + pad0 + 0.1, f"{m.loc[v, 'sr']:.1f} %\n({m.loc[v, 'served']:.1f}/{int(m.loc[v, 'n_req'])})", ha="center", fontsize=8.5, fontweight="bold")
for ax, key, base in [(axs[1], "wall", base_wall), (axs[2], "rtv", base_rtv)]:
    ax.set_ylim(0, m[key].max() * 1.22)
    for i, v in enumerate(ORDER):
        ax.text(i, g[g.variant == v][key].max() + m[key].max() * 0.015, f"{m.loc[v, key]:.0f} s\n{base / m.loc[v, key]:.2f}x", ha="center", fontsize=8.5, fontweight="bold")
fig.suptitle(f"RHO (no learning): Pair Pruner vs. Request Pruner vs. Combined, thresholds {THR_TXT}\n"
             "batch_interval=40/step_size=10, C=2 on board, max_cardinality 2, 14 Li&Lim instances x 2 repetitions", fontsize=11.5)
fig.text(0.5, 0.012, f"{ENV_TXT}, variants interleaved per instance. Pair Pruner = GNNv1 rgnn_mixed_c2_pw2_v1_final; Request Pruner = MLP request_pruner_mlp_h64_l3_d0p0_pw3p0_lr0p03 "
         "(trained for 40/10 on RHO labels, min_retention_fraction 0.3).\nService rate = served / all requests of the 14 instances. Run-to-run differences of a few requests are noise (RHO is not deterministic: served count differs between the 2 repetitions in "
         f"{int((d.pivot_table(index=['instance', 'variant'], columns='rep', values='served').pipe(lambda p: p[0] != p[1])).sum())} of {14 * len(ORDER)} instance/variant pairs).\n"  # 2026-11-06: was hard-coded 56 (= 14 x 4 variants), wrong for 7 variants
         f"Source: outputs/{CSV_NAME} via scripts/analysis/rho_pair_pruner_runtime.py.", ha="center", fontsize=7.4, color="#666666")
fig.tight_layout(rect=(0, 0.09, 1, 0.9))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_pruners_bi40ss10_{TAG}_summary.{ext}", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 2: wall-clock per instance
p = d.pivot_table(index="instance", columns="variant", values="wall_s", aggfunc="mean")[ORDER]
cls = {s: s[len(s.rstrip("0123456789")):][0] for s in p.index}
p["cls"] = p.index.map(cls)
p = p.sort_values(["cls", "base"])
fig, ax = plt.subplots(figsize=(15.5 if len(ORDER) <= 4 else 19, 5.4))
xx = np.arange(len(p))
w = 0.8 / len(ORDER)  # 2026-11-06: was fixed 0.2 for 4 bars
for k, v in enumerate(ORDER):
    ax.bar(xx + (k - (len(ORDER) - 1) / 2) * w, p[v], width=w, color=COL[v], zorder=3, label=LABEL[v].replace("\n", " "))
ax.set_xticks(xx)
ax.set_xticklabels(p.index, fontsize=9)
n1 = int((p.cls == "1").sum())
ax.axvline(n1 - 0.5, color="#999999", ls=":", lw=1)
top = p[ORDER].max().max() * 1.15
ax.text(n1 / 2 - 0.5, top * 0.96, "Class 1", ha="center", fontsize=10, color="#555555")
ax.text(n1 + (len(p) - n1) / 2 - 0.5, top * 0.96, "Class 2", ha="center", fontsize=10, color="#555555")
ax.set_ylim(0, top)
ax.set_ylabel("Wall-clock per run (s, mean of 2 repetitions)")
ax.set_title(f"RHO, batch_interval=40/step_size=10, C=2, cardinality 2, thresholds {THR_TXT}: wall-clock per instance", fontsize=11)
ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), ncol=min(len(ORDER), 4) if len(ORDER) <= 4 else 7, fontsize=9 if len(ORDER) <= 4 else 7.5, frameon=False)
fig.text(0.5, 0.012, f"{ENV_TXT}, 14 Li&Lim instances x 2 repetitions; source: outputs/{CSV_NAME}.", ha="center", fontsize=7.6, color="#666666")
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_pruners_bi40ss10_{TAG}_per_instance.{ext}", dpi=170)
plt.close(fig)

print(m.round(2).to_string())
print(g.round(2).to_string())
