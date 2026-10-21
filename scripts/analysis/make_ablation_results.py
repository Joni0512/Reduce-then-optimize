"""
2026-10-14: results of the small ablation sweep of the fixed SRL hyperparameters (base = config D: actor_lr 0.0031923396291543868, critic_lr 0.002, buffered actor update, 20 epochs,
seeds 301 and 302): sigma 0.1 / 0.3, samples 10 / 40, Polyak rate 0.001 / 0.01, gamma 0.97 / 0.98. Baseline = the D runs with the same seeds (bestD_s301/302: sigma 0.2, 20 samples,
Polyak 0.005, gamma 0.99). Run data: outputs/srl_training_sweep/abl_<name>_s<seed>/result.json (finished) or partial_val_curve.json (cluster runs that hit the time limit or are
still running; copied folders), plus outputs/ablation_cluster_manual.json (values pasted from the cluster) for runs whose folders are not local. Missing runs are shown as "pending".
Writes figures_export/SRL/ablation_results.{png,pdf} and ablation_results_table.{png,pdf,tex}. Rerun after new results arrive.
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures_export" / "SRL"
SW = REPO / "outputs" / "srl_training_sweep"
MANUAL = REPO / "outputs" / "ablation_cluster_manual.json"
manual = {k: v for k, v in json.load(open(MANUAL)).items() if not k.startswith('_')} if MANUAL.exists() else {}
SEEDS = (301, 302)
PARAMS = [  # key, label, run-name prefix, baseline, values (as shown), name fragments
    ("sigma", "σ (perturbation noise)", "sigma", 0.2, [0.1, 0.3], ["0p1", "0p3"]),
    ("samples", "samples (candidates)", "samples", 20, [10, 40], ["10", "40"]),
    ("polyak", "Polyak rate", "polyak", 0.005, [0.001, 0.01], ["0p001", "0p01"]),
    ("gamma", "discount γ", "gamma", 0.99, [0.97, 0.98, 0.995], ["0p97", "0p98", "0p995"]),  # 0.995 added 2026-10-14 (local runs)
]


def load_run(dirname):
    # 2026-10-14: a rerun on serial_long (run id suffix _r2, for runs that hit the 24 h limit on serial_std) takes precedence when it has a result
    if dirname.startswith('abl_') and (SW / (dirname + '_r2') / 'result.json').exists():
        dirname = dirname + '_r2'
    d = SW / dirname
    for fn in ("result.json", "partial_val_curve.json"):
        if (d / fn).exists():
            j = json.load(open(d / fn))
            curve = {pt["epoch"]: pt["service_rate"] for pt in j["val_curve"]}
            return dict(curve=curve, best=j["best_val_service_rate"], test=j.get("test_service_rate"), last=max(curve) if curve else 0, finished=(fn == "result.json"))
    m = manual.get(dirname)
    if m:
        curve = {int(k): v for k, v in m["val"].items()}
        return dict(curve=curve, best=m.get("best", max(curve.values())), test=m.get("test"), last=max(curve), finished=m.get("finished", False))
    return None


base = [load_run(f"bestD_s{s}") for s in SEEDS]


def agg(runs, key):
    vals = [key(r) for r in runs if r is not None and key(r) is not None]
    return (st.mean(vals), vals) if vals else (None, [])


METRICS = [("Val at epoch 10", lambda r: r["curve"].get(10)), ("Val at epoch 15", lambda r: r["curve"].get(15)), ("Val at epoch 20", lambda r: r["curve"].get(20)),
           ("Best val (epochs seen)", lambda r: r["best"]), ("Test", lambda r: r["test"])]
rows = []  # (param label, value text, runs)
for key, label, prefix, bval, vals, frags in PARAMS:
    for v, fr in zip(vals, frags):
        rows.append((key, label, v, [load_run(f"abl_{prefix}{fr}_s{s}") for s in SEEDS]))

# ---------------- table
tbl = [["baseline (σ 0.2, 20 samples, Polyak 0.005, γ 0.99)", "-", f"{sum(r is not None for r in base)}/2"] + [("n/a" if agg(base, k)[0] is None else f"{agg(base, k)[0]:.3f}") for _, k in METRICS] + ["finished"]]
for key, label, v, runs in rows:
    done = [r for r in runs if r is not None]
    status = "pending" if not done else ("finished" if all(r["finished"] for r in done) and len(done) == 2 else "partial: through epoch " + "/".join(str(r["last"]) for r in done) + (f" ({len(done)}/2 seeds)" if len(done) < 2 else ""))
    tbl.append([label.split(" (")[0], f"{v:g}", f"{len(done)}/2"] + [("n/a" if agg(runs, k)[0] is None else f"{agg(runs, k)[0]:.3f}") for _, k in METRICS] + [status])
cols = ["Parameter", "Value", "Seeds", "Val@10", "Val@15", "Val@20", "Best val*", "Test", "Status"]
fig, ax = plt.subplots(figsize=(14, 0.5 * len(tbl) + 1.2)); ax.axis("off")
t = ax.table(cellText=tbl, colLabels=cols, loc="center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(9); t.auto_set_column_width(col=list(range(len(cols)))); t.scale(1, 1.5)
for (i, j), c in t.get_celld().items():
    c.set_edgecolor("#bbbbbb")
    if i == 0: c.set_facecolor("#e8e8e8"); c.set_text_props(weight="bold")
    elif i == 1: c.set_facecolor("#eaf3fb")
ax.set_title("Ablation sweep (config D, seeds 301 and 302, mean over the available seeds); *best val over the epochs seen (partial runs: only up to the last validation)", fontsize=10)
fig.savefig(OUT / "ablation_results_table.png", dpi=150, bbox_inches="tight"); fig.savefig(OUT / "ablation_results_table.pdf", bbox_inches="tight"); plt.close(fig)
tex = ["\\begin{tabular}{llrrrrrrl}", "\\hline", " & ".join(cols) + " \\\\", "\\hline"] + [" & ".join(c.replace("σ", "$\\sigma$").replace("γ", "$\\gamma$").replace("_", "\\_") for c in r) + " \\\\" for r in tbl] + ["\\hline", "\\end{tabular}"]
(OUT / "ablation_results_table.tex").write_text("\n".join(tex) + "\n")

# ---------------- figure: 4 parameter groups x (low, baseline, high), 3 metrics as panels
panels = METRICS[:3]  # best val is left to the table: for partial runs it only covers the epochs seen
fig, axes = plt.subplots(1, 3, figsize=(17, 6.0), sharey=True)
for ax, (mlabel, mkey) in zip(axes, panels):
    bm, bvals = agg(base, mkey)
    ticks, tlabels = [], []
    xcur = 0.0
    for gi, (key, label, prefix, bval, vals, frags) in enumerate(PARAMS):
        gruns = [r for r in rows if r[0] == key]
        entries = sorted([(r[2], f"{r[2]:g}", r[3], "#E69F00") for r in gruns] + [(bval, f"{bval:g} (D)", base, "#0072B2")], key=lambda e: e[0])
        gx0 = xcur
        for k, (_, lab, runs, col) in enumerate(entries):
            m, vs = agg(runs, mkey)
            x = xcur
            ticks.append(x); tlabels.append(lab)
            if m is None:
                ax.text(x, 0.65, "pending", rotation=90, ha="center", va="center", fontsize=8.5, color="#888888")
            else:
                ax.bar(x, m, 0.8, color=col, alpha=0.75)
                ax.scatter([x + (i - (len(vs) - 1) / 2) * 0.18 for i in range(len(vs))], vs, s=22, color="black", zorder=3)
            xcur += 1
        ax.text((gx0 + xcur - 1) / 2, 0.797, label.split(" (")[0], ha="center", va="top", fontsize=9.5, weight="bold")
        xcur += 1.5
    if bm is not None:
        ax.axhline(bm, color="#0072B2", ls=":", lw=1)
    ax.axhline(0.7043, color="#444444", ls="--", lw=1.3)
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=8, rotation=45, ha="right")
    ax.set_ylim(0.50, 0.80); ax.set_title(mlabel, fontsize=10.5); ax.grid(True, axis="y", alpha=0.3)
axes[0].set_ylabel("Validation service rate")
axes[0].text(0, 0.7065, "RHO 0.7043", ha="left", fontsize=7.5, color="#444444")
fig.suptitle("Ablation of fixed hyperparameters (config D, 2 seeds each): orange = new value, blue = baseline D (same seeds), dots = single seeds", fontsize=11.5)
fig.text(0.01, 0.005, "Base: actor_lr 0.0031923396291543868, critic_lr 0.002, buffered actor update (batch 16), twin critic, no warmstart, pickup_slack, 20 epochs, bi200/ss100. Runs that did not finish show the epochs they reached "
         "('pending' = no result yet). Sources: outputs/srl_training_sweep/abl_*/ (result.json or partial_val_curve.json), bestD_s301-302; RHO: outputs/rho_val_runs.", fontsize=6.4, color="#444444")
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
fig.savefig(OUT / "ablation_results.png", dpi=150, bbox_inches="tight"); fig.savefig(OUT / "ablation_results.pdf", bbox_inches="tight")
for r in tbl: print(r)
