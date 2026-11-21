"""
2026-11-18: critic pretraining check on SF (user: "critic pretraining erstmal machen", option A local). Compares the
per-episode critic pretraining (twin critic, 10 epochs, random-init actor, target = immediate reward r_t, 1 gradient step
per episode) of
  * SF:     outputs/srl_sf_lodes/sf_uniformroad_bi900_ss450_mc2_v150_featfix_pretrainonly_s42 (1 train instance, seed 1)
  * Li&Lim: outputs/srl_training_sweep/served_D_s301 (38 train instances, batch_interval=200/step_size=100, critic_lr 0.002, reward local_served - not identical)
Metric: Q mean / target mean per pretrain epoch (1.0 = critic has learned the reward level) + MSE.
Output: figures_export/SF/sf_critic_pretrain_vs_lilim.{png,pdf,csv}
"""
import collections, csv, json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = {"SF: 1 instance, batch_interval=900/step_size=450, critic_lr=0.00108, reward local_positive": ROOT / "outputs/srl_sf_lodes/sf_uniformroad_bi900_ss450_mc2_v150_featfix_pretrainonly_s42/critic_loss_log.json",
        "Li&Lim served_D_s301: 38 instances, batch_interval=200/step_size=100, critic_lr=0.002, reward local_served": ROOT / "outputs/srl_training_sweep/served_D_s301/critic_loss_log.json"}
OUT = ROOT / "figures_export" / "SF" / "sf_critic_pretrain_vs_lilim"
rows = []
for name, p in RUNS.items():
    e = collections.defaultdict(list)
    for r in json.load(open(p)):
        if r["phase"] == "pretrain":
            e[r["epoch"]].append(r)
    for k, v in sorted(e.items()):
        m = lambda key: sum(r[key] for r in v) / len(v)
        rows.append({"run": name, "epoch": k, "episodes": len(v), "updates_cum": sum(len(e[j]) for j in range(k + 1)),
                     "critic_mse": m("critic_loss"), "q_mean": m("q_mean"), "target_mean": m("target_mean"),
                     "q_over_target": m("q_mean") / m("target_mean")})
with open(OUT.with_suffix(".csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for i, name in enumerate(RUNS):
    rr = [r for r in rows if r["run"] == name]
    c = ["#2a6fdb", "#e07b39"][i]
    ax[0].plot([r["updates_cum"] for r in rr], [r["q_over_target"] for r in rr], "o-", lw=2, ms=8, color=c, label=name)
    ax[1].plot([r["epoch"] + 1 for r in rr], [r["critic_mse"] / r["target_mean"] ** 2 for r in rr], "o-", lw=2, ms=8, color=c, label=name)
ax[0].axhline(1, color="gray", ls="--", lw=1); ax[0].set_xscale("log")
ax[0].set_xlabel("cumulative critic gradient steps"); ax[0].set_ylabel("Q mean / target mean"); ax[0].set_title("Has the critic learned the reward level?")
ax[1].set_xlabel("pretrain epoch"); ax[1].set_ylabel("critic MSE / target_mean²"); ax[1].set_title("Relative critic error")
for a in ax: a.grid(alpha=.3)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=1, fontsize=8)
fig.suptitle("Critic pretraining: 10 epochs, twin critic, random-init actor, target = r_t, 1 gradient step per episode (NOTE: Li&Lim reference differs in critic_lr and reward mode)", fontsize=9)
fig.tight_layout(rect=(0, 0.12, 1, 0.95)); fig.savefig(OUT.with_suffix(".png"), dpi=150); fig.savefig(OUT.with_suffix(".pdf"))
for r in rows: print(r)
