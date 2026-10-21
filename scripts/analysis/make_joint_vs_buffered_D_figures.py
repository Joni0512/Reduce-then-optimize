"""
2026-10-13: comparison of config D with the 1-epoch actor buffer (bestD_s301-305, "buffered") against config D with the joint 10-epoch buffer
(jointbuf_D_s<seed>, cluster runs; actor_lr 0.0031923396291543868, critic_lr 0.002, 20 epochs, 10-epoch buffer, 7 critic steps per actor step), vs. RHO.
Writes figures_export/SRL/joint_vs_buffered_D_overview.{png,pdf} (val curves + val@20 / best val / test bars) and
figures_export/SRL/joint_vs_buffered_D_per_instance.{png,pdf} (per instance: RHO, buffered, joint, plus a "mean" group).
Joint results are read from outputs/srl_training_sweep/jointbuf_D_s*/result.json (falls back to partial_val_curve.json without test value), so copy the cluster
folders (or at least result.json) there first. Usage: ./venv/bin/python3 scripts/analysis/make_joint_vs_buffered_D_figures.py [joint_prefix, default jointbuf_D_s]
"""
import glob
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures_export" / "SRL"
SW = REPO / "outputs" / "srl_training_sweep"
prefix = sys.argv[1] if len(sys.argv) > 1 else "jointbuf_D_s"
VAL = ["lc103", "lc202", "lr103", "lr107", "lr202", "lr208", "lrc102", "lrc202", "lrc205"]
C_BUF, C_JOINT, C_RHO = "#0072B2", "#D55E00", "#444444"


def load(dirs):
    runs = {}
    for d in dirs:
        r = d / "result.json"
        if r.exists():
            j = json.load(open(r)); test = j.get("test_service_rate")
        elif (d / "partial_val_curve.json").exists():
            j = json.load(open(d / "partial_val_curve.json")); test = None
        else:
            continue
        curve = {pt["epoch"]: pt for pt in j["val_curve"]}
        best_ep = j["best_epoch"]
        runs[d.name] = dict(curve={e: p["service_rate"] for e, p in curve.items()}, per={e: p["per_instance"] for e, p in curve.items()},
                            best=j["best_val_service_rate"], best_ep=best_ep, test=test, last=max(curve))
    return runs


buf = load([SW / f"bestD_s{s}" for s in (301, 302, 303, 304, 305)])
joint = load(sorted(Path(p) for p in glob.glob(str(SW / f"{prefix}*"))))
MANUAL_JOINT = REPO / "outputs" / "joint_D_manual.json"  # 2026-10-15: cluster values pasted into the chat (no per-instance values)
if MANUAL_JOINT.exists():
    for name, m in json.load(open(MANUAL_JOINT)).items():
        if name not in joint and not name.startswith("_"):
            curve = {int(k): v for k, v in m["val"].items()}
            joint[name] = dict(curve=curve, per={}, best=m["best"], best_ep=m.get("best_epoch"), test=m.get("test"), last=max(curve))
joint = {k: v for k, v in joint.items() if v["last"] >= 20}  # only finished 20-epoch runs
if not joint:
    sys.exit(f"no finished joint runs found under {SW}/{prefix}* - copy the cluster output folders there first")
rho_runs = [json.load(open(p)) for p in sorted(glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json")))]
rho = {i: st.mean(r["rates"][i] for r in rho_runs) for i in VAL}; rho_mean = st.mean(rho.values())
rho_test = sum(json.load(open(p))["mean"] for p in glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42_test/run*/summary.json"))) / 5
nb, nj = len(buf), len(joint)
print(f"buffered runs: {nb}, joint runs: {nj} ({sorted(joint)})")

# ---- figure 1: curves + bars
epochs = [5, 10, 15, 20]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.1, 1]})
for runs, col, lab in ((buf, C_BUF, f"1-epoch buffer (n={nb})"), (joint, C_JOINT, f"joint 10-epoch buffer (n={nj})")):
    for r in runs.values():
        ax1.plot(epochs, [r["curve"][e] for e in epochs], color=col, alpha=0.25, lw=1.2)
    m = [st.mean(r["curve"][e] for r in runs.values()) for e in epochs]
    sd = [st.pstdev([r["curve"][e] for r in runs.values()]) for e in epochs]
    ax1.plot(epochs, m, color=col, lw=3, marker="o", label=lab)
    ax1.fill_between(epochs, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)], color=col, alpha=0.10)
ax1.axhline(rho_mean, color=C_RHO, ls="--", lw=1.5, label=f"RHO validation: {rho_mean:.3f}")
ax1.set_xticks(epochs); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Validation service rate"); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8.5, loc="lower right")
ax1.set_title("Validation over the epochs (thin lines = single seeds)", fontsize=10.5)

metrics = [("Val at epoch 20", lambda r: r["curve"][20], rho_mean), ("Best val", lambda r: r["best"], rho_mean), ("Test", lambda r: r["test"], rho_test)]
for k, (lab, f, ref) in enumerate(metrics):
    for off, runs, col in ((-0.2, buf, C_BUF), (0.2, joint, C_JOINT)):
        vals = [f(r) for r in runs.values() if f(r) is not None]
        if not vals:
            continue
        m = st.mean(vals); sd = st.pstdev(vals) if len(vals) > 1 else 0
        ax2.bar(k + off, m, 0.36, yerr=sd, capsize=4, color=col, alpha=0.6)
        ax2.scatter([k + off + (i - (len(vals) - 1) / 2) * 0.05 for i in range(len(vals))], vals, s=22, color=col, edgecolor="white", zorder=3)
        ax2.text(k + off, 0.628, f"{m:.3f}\n±{sd:.3f}", ha="center", fontsize=8.5, weight="bold")
    ax2.plot([k - 0.4, k + 0.4], [ref, ref], color=C_RHO, ls="--", lw=1.8)
    ax2.text(k, ref + 0.003, f"RHO {ref:.4f}", ha="center", fontsize=8, color=C_RHO)
ax2.set_xticks(range(3)); ax2.set_xticklabels([m[0] for m in metrics]); ax2.set_ylim(0.62, 0.80); ax2.set_ylabel("Service rate"); ax2.grid(True, axis="y", alpha=0.3)
ax2.set_title("Mean ± std over seeds (blue: 1-epoch buffer, orange: joint 10-epoch buffer)", fontsize=10.5)
fig.suptitle("Config D (actor_lr=0.003192, critic_lr=0.002), 20 epochs: 1-epoch buffer vs. joint 10-epoch buffer", fontsize=12.5)
fig.text(0.01, 0.005, "Joint: shared buffer of the last 10 epochs, mini-batches of 16 drawn with replacement, 7 critic updates per actor update, Polyak 0.0083 per step; 1-epoch buffer: collect, then mini-batches of 16 without replacement. "
         "Sources: outputs/srl_training_sweep/bestD_s301-305 and jointbuf_D_s*/result.json; RHO: outputs/rho_val_runs (keep_active=False).", fontsize=6.3, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
fig.savefig(OUT / "joint_vs_buffered_D_overview.png", dpi=150, bbox_inches="tight"); fig.savefig(OUT / "joint_vs_buffered_D_overview.pdf", bbox_inches="tight")

# ---- figure 2: per instance (val at epoch 20), RHO / 1-epoch buffer / joint + mean group
if not all(r["per"] for r in joint.values()):
    print("per-instance values missing for the joint runs - overview figure only")
    for lab, f, _ in metrics:
        for name, runs in (("buffered", buf), ("joint", joint)):
            v = [f(r) for r in runs.values() if f(r) is not None]
            if v: print(f"{lab:16s} {name:9s} n={len(v)} mean {st.mean(v):.4f} std {st.pstdev(v):.4f}")
    sys.exit(0)
labels = VAL + ["mean"]
x = np.arange(len(labels)); w = 0.26
fig, ax = plt.subplots(figsize=(14, 6))
def per_inst(runs):
    m = {i: st.mean(r["per"][20][i] for r in runs.values()) for i in VAL}; sd = {i: st.pstdev([r["per"][20][i] for r in runs.values()]) for i in VAL}
    run_means = [st.mean(r["per"][20].values()) for r in runs.values()]
    return [m[i] for i in VAL] + [st.mean(run_means)], [sd[i] for i in VAL] + [st.pstdev(run_means)]
rho_sd = [st.pstdev([r["rates"][i] for r in rho_runs]) for i in VAL] + [st.pstdev([st.mean(r["rates"].values()) for r in rho_runs])]
by, be = per_inst(buf); jy, je = per_inst(joint)
ry = [rho[i] for i in VAL] + [rho_mean]
ax.bar(x - w, ry, w, yerr=rho_sd, capsize=3, color=C_RHO, label=f"RHO, mean of {len(rho_runs)} runs: {rho_mean:.3f}")
ax.bar(x, by, w, yerr=be, capsize=3, color=C_BUF, label=f"1-epoch buffer, epoch 20, mean of {nb} seeds: {by[-1]:.3f}")
ax.bar(x + w, jy, w, yerr=je, capsize=3, color=C_JOINT, label=f"joint 10-epoch buffer, epoch 20, mean of {nj} seeds: {jy[-1]:.3f}")
ax.axvline(len(VAL) - 0.5, color="#999999", ls=":", lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.2, 1.0); ax.set_xlim(-0.6, len(labels) - 0.4); ax.set_ylabel("Validation service rate (epoch 20)"); ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
ax.set_title("Per validation instance: RHO vs. 1-epoch buffer vs. joint 10-epoch buffer (config D)", fontsize=11.5)
fig.text(0.01, 0.005, "Sources: val_curve.per_instance of bestD_s301-305 and jointbuf_D_s*; RHO: outputs/rho_val_runs (keep_active=False, 7 runs). Error bars: std over seeds / runs; mean group: spread of the per-run means.", fontsize=6.5, color="#444444")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "joint_vs_buffered_D_per_instance.png", dpi=150, bbox_inches="tight"); fig.savefig(OUT / "joint_vs_buffered_D_per_instance.pdf", bbox_inches="tight")
for lab, f, _ in metrics:
    for name, runs in (("buffered", buf), ("joint", joint)):
        v = [f(r) for r in runs.values() if f(r) is not None]
        if v: print(f"{lab:16s} {name:9s} n={len(v)} mean {st.mean(v):.4f} std {st.pstdev(v):.4f}")
