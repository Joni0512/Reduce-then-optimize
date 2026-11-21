"""
2026-11-19: evaluation of the SF buffer critic pretraining (user: "Buffer mit collect 3 / sigma 0.2 / 500 steps / batch 20,
danach 10 Episoden Critic-Training mit frozen Actor", target r_t). Run:
  outputs/srl_sf_lodes/sf_uniformroad_bi900_ss450_mc2_v150_featfix_split10_bufpre_pretrainonly_s42
  (script rtv_solver/pipeline/run_srl_sf_uniformroad_bi900_ss450.py --pretrain_only --buffer_pretrain, seed 42,
   10 train instances seeds 1,3-11, batch_interval=900/step_size=450, mc2, 150 vehicles, twin critic, critic_lr baseline)
Phase 1 (critic_pretrain_log.json): 510 windows collected (3 passes x 10 instances x 17 windows, random-init actor + sigma 0.2
  noise), 500 gradient steps (batch 20), target r_t; MSE / mean Q evaluated on all 510 windows every 25 steps.
  Constant-predictor baseline: Var(r) = MSE at step 0 (Q ~ 0, i.e. E[r^2]) - mean(r)^2.
Phase 2 (critic_loss_log.json, phase=pretrain): 10 episodes (1 per train instance), frozen actor WITHOUT noise
  (srl_training_loop: pretrain never samples), 1 gradient step per episode, target r_t.
Output: figures_export/SF/sf_critic_buffer_pretrain.{png,pdf} + _buffer.csv + _episodes.csv
"""
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/srl_sf_lodes/sf_uniformroad_bi900_ss450_mc2_v150_featfix_split10_bufpre_pretrainonly_s42"
OUT = ROOT / "figures_export" / "SF" / "sf_critic_buffer_pretrain"

buf = json.load(open(RUN / "critic_pretrain_log.json"))
ev = [r for r in buf["log"] if r.get("mse1") is not None]
r_mean = ev[0]["target_mean"]
var_r = ev[0]["mse1"] - r_mean ** 2 if abs(ev[0]["q1"]) < 1 else float("nan")
brows = [{"step": r["step"], "mse1": r["mse1"], "mse2": r["mse2"], "q1": r["q1"], "q2": r["q2"], "target_mean": r["target_mean"],
          "var_r_const_baseline": var_r, "r2_critic1": 1 - r["mse1"] / var_r, "r2_critic2": 1 - r["mse2"] / var_r} for r in ev]
eps = [r for r in json.load(open(RUN / "critic_loss_log.json")) if r["phase"] == "pretrain"]
erows = [{"episode": i + 1, "instance": r["instance"].split("_")[2], "mse1": r["critic_loss"], "mse2": r["critic2_loss"],
          "q1": r["q_mean"], "q2": r["q2_mean"], "target_mean": r["target_mean"]} for i, r in enumerate(eps)]
for name, rows in (("_buffer", brows), ("_episodes", erows)):
    with open(str(OUT) + name + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

C1, C2, CT = "#2a6fdb", "#e07b39", "#444444"
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
s = [r["step"] for r in brows]
ax[0, 0].plot(s, [r["q1"] for r in brows], "-", lw=2, color=C1, label="critic 1 mean Q")
ax[0, 0].plot(s, [r["q2"] for r in brows], "-", lw=2, color=C2, label="critic 2 mean Q")
ax[0, 0].axhline(r_mean, color=CT, ls="--", lw=1.5, label=f"mean target r_t = {r_mean:.1f}")
ax[0, 0].set(xlabel="gradient step (batch 20)", ylabel="mean over 510 windows", title="Phase 1 buffer: mean Q vs target")
ax[0, 1].plot(s, [r["mse1"] for r in brows], "-", lw=2, color=C1, label="critic 1 MSE")
ax[0, 1].plot(s, [r["mse2"] for r in brows], "-", lw=2, color=C2, label="critic 2 MSE")
ax[0, 1].axhline(var_r, color=CT, ls="--", lw=1.5, label=f"constant predictor = Var(r) = {var_r:.0f}")
ax[0, 1].set_yscale("log"); ax[0, 1].set(xlabel="gradient step (batch 20)", ylabel="MSE on all 510 windows (log)", title="Phase 1 buffer: MSE")
e = [r["episode"] for r in erows]
ax[1, 0].plot(e, [r["q1"] for r in erows], "o-", lw=2, ms=8, color=C1, label="critic 1 mean Q")
ax[1, 0].plot(e, [r["q2"] for r in erows], "o-", lw=2, ms=8, color=C2, label="critic 2 mean Q")
ax[1, 0].plot(e, [r["target_mean"] for r in erows], "s--", lw=1.5, ms=7, color=CT, label="episode mean r_t")
ax[1, 0].set(xlabel="episode (1 per train instance, 1 gradient step each)", ylabel="mean over episode windows",
             title="Phase 2 per-episode, frozen actor (no noise)")
ax[1, 1].bar([x - 0.2 for x in e], [r["mse1"] for r in erows], 0.4, color=C1, label="critic 1 MSE")
ax[1, 1].bar([x + 0.2 for x in e], [r["mse2"] for r in erows], 0.4, color=C2, label="critic 2 MSE")
ax[1, 1].set(xlabel="episode", ylabel="MSE (before the update)", title="Phase 2: MSE per episode")
for a in ax.flat: a.grid(alpha=.3); a.legend(fontsize=8)
fig.suptitle("SF critic pretraining, batch_interval=900/step_size=450, max_cardinality=2, 150 vehicles, 10 train instances (seeds 1,3-11), seed 42\n"
             "Phase 1: 3 collect passes, exploration sigma=0.2, 500 steps, batch 20, target r_t (reward local_positive) | "
             "Phase 2: 1 epoch per-episode, frozen random-init actor, target r_t\nsource: " + RUN.relative_to(ROOT).as_posix(), fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(str(OUT) + ".png", dpi=150); fig.savefig(str(OUT) + ".pdf")
print(f"mean r {r_mean:.2f}, Var(r) {var_r:.1f}"); print(brows[0]); print(brows[-1])
