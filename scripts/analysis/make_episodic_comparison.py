"""
2026-10-19: bar chart + table: old buffered baseline (config D, 5 seeds) vs new episodic scheme (1-step TD, 5 seeds) vs new episodic scheme with two-step TD (3 seeds so far).
Metrics: best val, val at epoch 20, test (checkpoint of the best-val epoch); bars = mean over seeds, whiskers = population std, dots = single seeds, seeds written above each bar.
Data: ROWS of scripts/analysis/make_all_finished_runs_table.py (result.json of the local runs, values pasted from the cluster result.json files for the cluster seeds).
RHO references (keep_active=False, bi200/ss100): validation 0.7043 (mean of 7 runs), test 0.7379 (mean of 5 runs), from outputs/rho_val_runs. PNG + PDF in figures_export/SRL/.
"""
import runpy
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
ROWS = runpy.run_path(str(REPO / "scripts" / "analysis" / "make_all_finished_runs_table.py"))["ROWS"]
CLR = 0.002
groups = [
    ("old: buffered", lambda r: r["method"] == "new (buffered)" and r.get("note") == "config D" and abs(r["critic_lr"] - CLR) < 1e-9, "#0072B2"),
    ("new: episodic", lambda r: r["method"] == "new (episodic)" and abs(r["critic_lr"] - CLR) < 1e-9, "#D55E00"),
    ("new: episodic\n+ 2-step TD", lambda r: r["method"] == "new (episodic, 2-step TD)" and abs(r["critic_lr"] - CLR) < 1e-9, "#009E73"),
    ("new: episodic\n+ 2-step TD\ncritic_lr 0.004", lambda r: r["method"] == "new (episodic, 2-step TD)" and abs(r["critic_lr"] - 0.004) < 1e-9, "#CC79A7"),  # 2026-10-19: one run (seed 301), critic_lr doubled
]
metrics = [("best val", lambda r: r["best_val"], 0.7043), ("val at epoch 20", lambda r: r["val"].get(20), 0.7043), ("test", lambda r: r["test"], 0.7379)]
data = []
for name, sel, col in groups:
    rs = sorted([r for r in ROWS if sel(r)], key=lambda r: r["seed"])
    data.append((name, rs, col))

fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)
for ax, (mname, fn, rho) in zip(axes, metrics):
    for i, (name, rs, col) in enumerate(data):
        vals = [fn(r) for r in rs if fn(r) is not None]
        m, s = st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)
        ax.bar(i, m, color=col, alpha=0.75, width=0.62, yerr=s, capsize=5)
        ax.scatter([i + 0.0 + 0.16 * ((k % 2) - 0.5) for k in range(len(vals))], vals, color="black", s=14, zorder=3)
        sd = [r["seed"] for r in rs if fn(r) is not None]
        seeds_txt = (str(sd[0]) if len(sd) == 1 else f"{sd[0]}-{sd[-1]}") if sd == list(range(sd[0], sd[-1] + 1)) else ",".join(map(str, sd))
        ax.text(i, max(vals) + 0.012, f"seed{'s' if len(sd) > 1 else ''} {seeds_txt}\nn={len(vals)}", ha="center", va="bottom", fontsize=8.5)
        ax.text(i, 0.505, f"{m:.3f}", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
    ax.axhline(rho, color="gray", ls="--", lw=1, label=f"RHO {rho:.4f}")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(mname); ax.set_xticks(range(len(data))); ax.set_xticklabels([d[0] for d in data], fontsize=8.5); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("service rate (pooled over the 9 validation / the test instances)")
axes[0].set_ylim(0.5, 0.86)
fig.suptitle("actor_lr=0.0031923  critic_lr=0.002 (last bar 0.004)  epochs=20  gamma=0.99  polyak_tau=0.005  sigma=0.2  num_samples=20  twin_critic  batch_interval=200  step_size=100  pickup_slack\n"
             "old: reward=local_positive  td_steps=1  actor_batch=16  pretrain=10 epochs     new: reward=local_served  td_steps=1 / 2  actor_batch=20  critic_batch=20  buffer=10 episodes  pretrain=10 instances x 150 steps", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = REPO / "figures_export" / "SRL"
for ext in ("png", "pdf"):
    fig.savefig(out / f"episodic_comparison_bars.{ext}", dpi=160)

# table
fig2, ax2 = plt.subplots(figsize=(17, 3.4)); ax2.axis("off")
rows = []
for name, rs, col in data:
    cells = [name.replace("\n", " ")]
    for mname, fn, _ in metrics:
        vals = [fn(r) for r in rs if fn(r) is not None]
        cells.append((f"{st.mean(vals):.3f} ± {st.pstdev(vals):.3f}" if len(vals) > 1 else f"{st.mean(vals):.3f}") + f"\n(n={len(vals)}: " + ", ".join(f"{v:.3f}" for v in vals) + ")")
    cells.append(", ".join(str(r["seed"]) for r in rs))
    rows.append(cells)
t = ax2.table(cellText=rows, colLabels=["variant", "best val", "val at epoch 20", "test", "seeds"], loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 3.2); t.auto_set_column_width(list(range(5)))
fig2.suptitle("mean ± std over seeds (single seeds in brackets, in seed order)", fontsize=9)
for ext in ("png", "pdf"):
    fig2.savefig(out / f"episodic_comparison_table.{ext}", dpi=160, bbox_inches="tight")
for name, rs, _ in data:
    print(name.split("\n")[0], [r["seed"] for r in rs])
