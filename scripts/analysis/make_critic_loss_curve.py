"""
2026-10-17: critic loss curve of one SRL run. Reads outputs/srl_training_sweep/<run>/critic_loss_log.json (one record per episode = one instance rollout:
critic 1 / critic 2 MSE, mean live-critic prediction q_mean, mean TD target; written by srl_training_loop._record_critic_stats). Left: MSE per episode
(thin = single episodes, thick = mean per epoch, critic 1 and 2, log scale) over critic pretrain (target = r_t) and main training (TD target
r_t + gamma*min Q'(s_t+1)); dashed line = phase change. Right: mean Q prediction vs mean target per epoch. Usage: python make_critic_loss_curve.py <run_name>
(default served_D_s301). PNG + PDF in figures_export/SRL/.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
run = sys.argv[1] if len(sys.argv) > 1 else "served_D_s301"
log = json.load(open(REPO / "outputs" / "srl_training_sweep" / run / "critic_loss_log.json"))
ok = lambda v: v is not None and not math.isnan(v)
n_pre = sum(r["phase"] == "pretrain" for r in log)
x_of = {i: i + 1 for i in range(len(log))}

# per-epoch means in episode-index space (x = last episode of the epoch block)
blocks = defaultdict(list)
for i, r in enumerate(log):
    blocks[(r["phase"], r["epoch"])].append((i + 1, r))
def mean(vals):
    vals = [v for v in vals if ok(v)]
    return sum(vals) / len(vals) if vals else float("nan")
ep = [(k, sum(x for x, _ in v) / len(v), mean([r["critic_loss"] for _, r in v]), mean([r.get("critic2_loss") for _, r in v]),
       mean([r["q_mean"] for _, r in v]), mean([r["target_mean"] for _, r in v])) for k, v in blocks.items()]

# 2026-10-17 (v2): x = epoch (pretrain epochs first, then training epochs), one point = mean over the ~39 episodes of that epoch (+- std as band)
keys = sorted(blocks, key=lambda k: (k[0] != "pretrain", k[1]))
xe = {k: i + 1 for i, k in enumerate(keys)}
n_pre_ep = sum(k[0] == "pretrain" for k in keys)
def stat(k, name):
    vals = [r[name] for _, r in blocks[k] if ok(r.get(name))]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    return m, (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
XLAB = "training epoch" if n_pre_ep == 0 else f"epoch (1-{n_pre_ep} pretrain, then training epochs)"  # 2026-10-18: buffer pretrain (episodic runs) is not in this log
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
for key, col, lab in (("critic_loss", "#0072B2", "critic 1"), ("critic2_loss", "#D55E00", "critic 2")):
    pts = [(xe[k],) + stat(k, key) for k in keys if ok(stat(k, key)[0])]
    if not pts:
        continue  # 2026-10-18: e.g. critic 2 loss is not logged in the episodic update step (critic_td_update_step returns critic 1 only)
    a.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2.2, marker="o", ms=4, label=lab)
a.set_yscale("log"); a.set_xlabel(XLAB); a.set_ylabel("critic MSE (mean over the 38 episodes of the epoch)"); a.grid(alpha=0.3); a.legend(frameon=False)
a.set_title("Critic loss per epoch")
# 2026-10-18 (v3): newer runs log q2_mean and step_target_mean (TD target on the SAME steps as q); episodic runs have no batch target_mean (critic updates are deferred) -> fall back to step_target_mean
has_step_target = any(ok(r.get("step_target_mean")) for r in log)
tgt = "step_target_mean" if has_step_target else "target_mean"
series = [("q_mean", "#009E73", "mean Q critic 1"), ("q2_mean", "#E69F00", "mean Q critic 2"), (tgt, "#CC79A7", "mean TD target (same steps)" if has_step_target else "mean target")]
for name, col, lab in series:
    pts = [(xe[k], stat(k, name)[0]) for k in keys if ok(stat(k, name)[0])]
    if not pts:
        continue
    b.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2, marker="o", ms=4, label=lab)
b.set_xlabel(XLAB); b.set_ylabel("value"); b.grid(alpha=0.3); b.legend(frameon=False)
b.set_title("Q prediction vs target per epoch")
for ax in ((a, b) if n_pre_ep else ()):
    ax.axvline(n_pre_ep + 0.5, color="gray", ls="--", lw=1)
    ax.text(n_pre_ep + 0.5, ax.get_ylim()[1], " training", va="top", ha="left", fontsize=9, color="gray")
    ax.text(n_pre_ep + 0.5, ax.get_ylim()[1], "pretrain ", va="top", ha="right", fontsize=9, color="gray")
fig.suptitle(f"{run}: critic loss (source: outputs/srl_training_sweep/{run}/critic_loss_log.json)", fontsize=10)
fig.tight_layout()
out = REPO / "figures_export" / "SRL"
for ext in ("png", "pdf"):
    fig.savefig(out / f"critic_loss_curve_{run}.{ext}", dpi=160)
print("saved", out / f"critic_loss_curve_{run}.png", "episodes:", len(log), "pretrain:", n_pre)
