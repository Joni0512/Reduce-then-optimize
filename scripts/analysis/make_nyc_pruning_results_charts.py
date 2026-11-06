"""
2026-11-06: charts of all NYC pruning results (Val day 2016-01-13, Train day 2016-01-12; Test day 2016-01-19 untouched).
Each figure carries its exact setup and source in the footer. Output: figures_export/NYC/nyc_pruning_*.{png,pdf}

Hard-coded numbers (cannot be re-read from this machine; the runs were on the LRZ cluster) and their sources:
  - Val RHO table: pasted cluster evaluation of 2026-11-06 (jobs 5708612 / 5710998 / 5707367 / timerule jobs), columns served, VMT (m), runtime (s).
  - Train RHO table: outputs of scripts/analysis/eval_nyc_rho3060_trainday.py (jobs 5712813_0 / 5712813_1), final/results.json (total_time, vmt, serviced).
  - Request-pruner label rates: bi1800_ss900 / bi900_ss450 from the cluster count of 2026-11-05 (assigned / (assigned + unserved) per window);
    bi3600_ss1800 old label computed locally from outputs/new_tests/nyc_30min_60min_baseline/{train,val}_seed1/**/assignment_data.jsonl (same formula),
    new label from dataset/request_pruning_nyc_pickup_bi3600_ss1800/manifest.json.
Read from files: outputs/pair_pruner_window_rho3060/*.json (36 runs, 12 configs x 3 seeds).

Run: ./venv/bin/python3 scripts/analysis/make_nyc_pruning_results_charts.py
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures_export/NYC")
OUT.mkdir(parents=True, exist_ok=True)
NAVY, AMBER, TEAL, GREY = "#1E2761", "#E8A33D", "#2A9D8F", "#7A7F8C"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

SETUP = "RHO offline, batch_interval=3600/step_size=1800, max_cardinality=3, NYC 50 vehicles, ~8 h window, LRZ cluster (16 CPUs, 20G)"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


def footer(fig, text, y=-0.02):
    fig.text(0.01, y, text, fontsize=7.5, color="#444", ha="left", va="top", wrap=True)


# ---------------------------------------------------------------- 1) Val day: service rate + runtime per variant
val = [  # label, served, VMT_m, runtime_s, colour
    ("No pruner\nseed 1", 1014, 766336, 10094, NAVY),
    ("No pruner\nseed 2", 1012, 765702, 10023, NAVY),
    ("Pair pruner\nseed 1", 983, 808257, 1908, AMBER),
    ("Pair pruner\nseed 2", 1006, 830626, 1562, AMBER),
    ("Time rule\nseed 1", 1035, 847079, 429, TEAL),
    ("Time rule\nseed 2", 1035, 847079, 472, TEAL),
    ("Time rule +\npair pruner\nseed 1", 1036, 845137, 488, GREY),
]
N_VAL = 1047
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))
x = np.arange(len(val))
rates = [100 * v[1] / N_VAL for v in val]
a1.bar(x, rates, color=[v[4] for v in val], width=0.65, zorder=3)
for i, r in enumerate(rates):
    a1.text(i, r + 0.15, f"{r:.2f} %\n{val[i][1]}/{N_VAL}", ha="center", fontsize=7.5)  # 2026-11-06: smaller font so neighbouring labels do not overlap
a1.set_ylim(90, 100.6)
a1.set_xticks(x)
a1.set_xticklabels([v[0] for v in val], fontsize=8.5)
a1.set_ylabel("Service rate [%]   (axis starts at 90 %)")
a1.set_title("Service rate (Val day 2016-01-13, 1047 requests)")
a1.grid(axis="y", alpha=0.25, zorder=0)
a2.bar(x, [v[3] for v in val], color=[v[4] for v in val], width=0.65, zorder=3)
for i, v in enumerate(val):
    a2.text(i, v[3] + 150, f"{v[3]:,} s", ha="center", fontsize=8.5)
a2.set_xticks(x)
a2.set_xticklabels([v[0] for v in val], fontsize=8.5)
a2.set_ylabel("Runtime [s]")
a2.set_title("Runtime")
a2.grid(axis="y", alpha=0.25, zorder=0)
fig.suptitle("NYC Val day: RHO without pruner vs. pair pruner vs. time-rule request filter", fontsize=12, y=1.02)
footer(fig, SETUP + ".\nPair pruner = RequestGraphEdgeGNN trained on window graphs (1 layer, pos_weight=5, lr=0.003, hidden_dim=64, seed 1 checkpoint), threshold 0.5.\n"
       "Time rule = --use_request_time_rule_filter True (keep requests that are active or have earliest_pickup_time < t + step_size, defer the rest).\n"
       "Source: cluster evaluation table of 2026-11-06 (no pruner / pair pruner / time-rule jobs on Val).")
save(fig, "nyc_pruning_val_service_rate_runtime")

# ---------------------------------------------------------------- 2) Train vs Val: baseline vs time rule
days = ["Val 2016-01-13\n(1047 requests)", "Train 2016-01-12\n(1000 requests)"]
base = {"rate": [100 * 1014 / 1047, 100 * 953 / 1000], "vmt": [766.336, 761.9], "rt": [10094, 7042.1]}
rule = {"rate": [100 * 1035 / 1047, 100 * 980 / 1000], "vmt": [847.079, 842.3], "rt": [429, 283.5]}
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
titles = [("rate", "Service rate [%]   (axis starts at 90 %)", "{:.2f} %"), ("vmt", "VMT [km]", "{:.0f} km"), ("rt", "Runtime [s]", "{:,.0f} s")]
w = 0.36
for ax, (k, yl, fmt) in zip(axs, titles):
    xx = np.arange(2)
    b1 = ax.bar(xx - w / 2, base[k], w, color=NAVY, label="No pruner (seed 1)", zorder=3)
    b2 = ax.bar(xx + w / 2, rule[k], w, color=TEAL, label="Time rule (seed 1)", zorder=3)
    for bars, vals in ((b1, base[k]), (b2, rule[k])):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.1 if k == "rate" else 0.012 * max(base[k])), fmt.format(v), ha="center", fontsize=8.5)  # 2026-11-06: offset per panel scale
    ax.set_xticks(xx)
    ax.set_xticklabels(days)
    ax.set_ylabel(yl)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    if k == "rate":
        ax.set_ylim(90, 100.2)
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.99))  # 2026-11-06: legend outside the bars
fig.suptitle("Time-rule filter replicates on a second day (Train) - no pair pruner here, its checkpoint was trained on the Train day", fontsize=11.5, y=1.08)
footer(fig, SETUP + ", seed 1.\nSources: Val = cluster evaluation table of 2026-11-06 (seed 1 rows); Train = eval_nyc_rho3060_trainday.py on final/results.json "
       "(jobs 5712813_0 / 5712813_1; runtime = results.json total_time).\nTest day 2016-01-19 not used. One seed per day for the time rule; the no-pruner baseline has two seeds only on Val.")
save(fig, "nyc_pruning_time_rule_val_vs_train")

# ---------------------------------------------------------------- 3) Pair pruner window-graph grid (read from files)
runs = defaultdict(list)
for f in sorted(glob.glob("outputs/pair_pruner_window_rho3060/*.json")):
    r = json.load(open(f))
    h, b = r["hyperparameters"], r["best"]
    runs[(h["message_passing_steps"], h["pos_weight"], h["learning_rate"])].append(b)
cfgs = sorted(runs, key=lambda k: -np.mean([b["val_auc"] for b in runs[k]]))
lab = [f"L={k[0]}, pw={k[1]:g}, lr={k[2]:g}" for k in cfgs]  # 2026-11-06: one line, rotated (labels overlapped)
auc_m = [np.mean([b["val_auc"] for b in runs[k]]) for k in cfgs]
auc_s = [np.std([b["val_auc"] for b in runs[k]], ddof=1) for k in cfgs]
rec_m = [np.mean([b["val_recall_at_0.5"] for b in runs[k]]) for k in cfgs]
rec_s = [np.std([b["val_recall_at_0.5"] for b in runs[k]], ddof=1) for k in cfgs]
kept_m = [np.mean([b["val_kept_share_at_0.5"] for b in runs[k]]) for k in cfgs]
kept_s = [np.std([b["val_kept_share_at_0.5"] for b in runs[k]], ddof=1) for k in cfgs]
nseed = {len(v) for v in runs.values()}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
x = np.arange(len(cfgs))
a1.bar(x, auc_m, yerr=auc_s, color=NAVY, width=0.65, capsize=3, zorder=3)
for i, (m, s) in enumerate(zip(auc_m, auc_s)):
    a1.text(i, m + s + 0.0008, f"{m:.3f}", ha="center", fontsize=8)
a1.set_ylim(0.95, 0.975)
a1.set_xticks(x)
a1.set_xticklabels(lab, fontsize=8, rotation=45, ha="right")
a1.set_ylabel("Val AUC, window graphs (axis starts at 0.95)")
a1.set_title("Val AUC (mean +- SD over seeds)")
a1.grid(axis="y", alpha=0.25, zorder=0)
w = 0.38
a2.bar(x - w / 2, rec_m, w, yerr=rec_s, color=AMBER, capsize=2, label="Recall of overlap pairs at logit >= 0 (threshold 0.5)", zorder=3)
a2.bar(x + w / 2, kept_m, w, yerr=kept_s, color=TEAL, capsize=2, label="Share of pairs kept at threshold 0.5", zorder=3)
for i in range(len(cfgs)):
    a2.text(i - w / 2, rec_m[i] + rec_s[i] + 0.01, f"{rec_m[i]:.2f}", ha="center", fontsize=7)
    a2.text(i + w / 2, kept_m[i] + kept_s[i] + 0.01, f"{kept_m[i]:.2f}", ha="center", fontsize=7)
a2.set_ylim(0, 1.3)
a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a2.set_xticks(x)
a2.set_xticklabels(lab, fontsize=8, rotation=45, ha="right")
a2.set_title("Recall and kept share at threshold 0.5")
a2.legend(fontsize=8, frameon=False, loc="upper left", ncol=1)
a2.grid(axis="y", alpha=0.25, zorder=0)
fig.suptitle("NYC pair pruner trained on window graphs: 12 configurations x %s seeds" % "/".join(str(n) for n in sorted(nseed)), fontsize=12, y=1.02)
footer(fig, "RequestGraphEdgeGNN, label_mode=overlap, graph=window (per-window z-score), hidden_dim=64, epochs=100, balanced=True, batch_interval=3600/step_size=1800, max_cardinality=3.\n"
       "L = message passing layers, pw = pos_weight, lr = learning rate. Train = NYC 2016-01-12 (15 windows), Val = 2016-01-13 (15 windows); best epoch by Val AUC per run.\n"
       "Source: outputs/pair_pruner_window_rho3060/*.json (36 runs), aggregate_pair_pruner_window_grid.py; trained locally.", y=-0.2)  # 2026-11-06: below the rotated tick labels
save(fig, "nyc_pruning_pair_pruner_window_grid")

# ---------------------------------------------------------------- 4) Request pruner label positive rates
cfg_labels = ["batch_interval=3600\nstep_size=1800", "batch_interval=1800\nstep_size=900", "batch_interval=900\nstep_size=450"]
old_tr = [96.9, 99.4, 95.6]
old_va = [94.0, 99.4, 96.5]
new_tr = [48.3, None, None]
new_va = [47.0, None, None]
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(3)
w = 0.2
series = [("Old label 'assigned in window' - Train", old_tr, NAVY), ("Old label - Val", old_va, "#5A6AA8"),
          ("New label 'pickup before t+step_size' - Train", new_tr, TEAL), ("New label - Val", new_va, "#7BC8BD")]
for j, (nm, vals, c) in enumerate(series):
    for i, v in enumerate(vals):
        if v is None:
            continue
        ax.bar(i + (j - 1.5) * w, v, w, color=c, label=nm if i == 0 or (nm.startswith("New") and i == 0) else None, zorder=3)
        ax.text(i + (j - 1.5) * w, v + 1.2, f"{v:.1f} %", ha="center", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(cfg_labels)
ax.set_ylim(0, 112)
ax.set_ylabel("Share of positive (window, request) rows [%]")
ax.set_title("Request-pruner labels: share of positive rows (old label: almost no negatives)")
ax.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))  # 2026-11-06: legend outside, it covered bars
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.text(1, 20, "new label not built\nfor these configs", ha="center", fontsize=8.5, color="#555")
ax.text(2, 20, "new label not built\nfor these configs", ha="center", fontsize=8.5, color="#555")
footer(fig, "RHO baseline without pruner, max_cardinality=3, seed 1, NYC Train 2016-01-12 / Val 2016-01-13. Rows = requests active in a window (assigned + unserved, windows with 0 requests dropped).\n"
       "Old label = assigned in this window. New label = assigned AND pickup in the final plan < window time + step_size.\n"
       "Sources: 1800/900 and 900/450 = cluster count of 2026-11-05; 3600/1800 old = local count on outputs/new_tests/nyc_30min_60min_baseline (Train 1944/2006, Val 1984/2111);\n"
       "3600/1800 new = dataset/request_pruning_nyc_pickup_bi3600_ss1800/manifest.json (positive_rate 0.4826 / 0.4699).")
save(fig, "nyc_pruning_request_pruner_label_rates")
