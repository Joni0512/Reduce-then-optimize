"""
2026-11-22: first SRL run on the NYC real-day instances (user request, see chat): batch_interval=900/step_size=450,
max_cardinality=3, time-rule request filter OFF (user decision 2026-11-22; at 1800/900 the filter cost 5.6-8.2 pp service
rate on NYC), 50 vehicles, am_capacity 3, 1,000-request days with 300 s pickup windows (solutions/nyc/manifests/*_v50.json).
Question behind it: does SRL need a shorter horizon than RHO, i.e. can an SRL actor at 900/450 reach what RHO needs
1800/900 for? Reference RHO (no pruner, cluster, seed 1, results.json): 900/450 Val 95.51 % / Train 94.00 %;
1800/900 Val 99.33 % / Train 99.10 %. RHO references for 2016-01-05 and 2016-01-06 do not exist yet.

Copy of run_srl_sf_lodes_bi900_ss450.py (same reason for a separate script: srl_training_loop.py is hard-wired to Li&Lim and
is NOT changed; this script patches its module globals at import time):
  * MANIFEST_DIR -> solutions/nyc/manifests; TRAIN/VAL/OVERFIT_CHECK instance lists -> NYC manifest stems
  * Config -> functools.partial(Config, MAX_CARDINALITY=3, RTV_TIMEOUT=1800) (loop's own Configs would fall back to
    MAX_CARDINALITY 2 and RTV_TIMEOUT 120; call-site keywords still win)
SRL protocol = srl_buffered_actor_lr_trial.py (buffered actor update, batch 16, twin critic, no warmstart, pickup_slack,
local_positive reward, gamma 0.99, tau 0.005, sigma 0.2, 20 samples, critic_lr baseline 0.0010833586558285635,
actor_lr baseline 0.0031923396291543868), all imported from there so the values cannot drift. No SIL warm start.
Split (user decision 2026-11-22): train = 2016-01-05, 2016-01-06, 2016-01-12; val = 2016-01-13 (the NYC Val day used for all
pruner results). Test days (2016-01-19/20) are NOT touched. Days are separate instances, never pooled.

--smoke: 1 main epoch, no critic pretraining, validation once at the end -> checks that SRL runs on NYC at all and
measures the time of one training episode + one validation episode (decides how many epochs/seeds are feasible).

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.run_srl_nyc_bi900_ss450 --smoke
"""
import argparse
import functools
import json
import time

# import first: re-patches FeatureBuilder pickup_slack=True AFTER srl_training_loop's own patch (see that module)
from rtv_solver.pipeline import srl_buffered_actor_lr_trial as trial
from rtv_solver.pipeline import srl_training_loop as loop
from rtv_solver.pipeline import srl_buffer_pretrain as buffer_pretrain
from rtv_solver.structure.config import Config

BATCH_INTERVAL, STEP_SIZE, MAX_CARDINALITY = 900, 450, 3
NYC_MANIFEST_DIR = loop.REPO_ROOT / "solutions" / "nyc" / "manifests"
TRAIN = ["nyc_real1000_20160105_0614_train_v50",
         "nyc_real1000_20160106_0614_val_v50",
         "nyc_real1000_20160112_0614_train_v50"]
VAL = ["nyc_real1000_20160113_0614_val_v50"]
ACTOR_LR = 0.0031923396291543868  # 1x baseline of the buffered actor_lr scan (srl_buffered_actor_lr_trial.py docstring)

_NYCConfig = functools.partial(Config, MAX_CARDINALITY=MAX_CARDINALITY, RTV_TIMEOUT=1800)
loop.MANIFEST_DIR = NYC_MANIFEST_DIR
loop.TRAIN_INSTANCES = list(TRAIN)
loop.VAL_INSTANCES = list(VAL)
loop.OVERFIT_CHECK_INSTANCES = list(TRAIN)
loop.Config = _NYCConfig
buffer_pretrain.Config = _NYCConfig  # only used with pretrain_mode="buffer"; patched so it cannot silently fall back


# 2026-11-23: = run_srl_sf_uniformroad_bi900_ss450.py --buffer_pretrain
BUFFER_PRETRAIN_KW = dict(pretrain_mode="buffer", pretrain_instances=["all"], pretrain_collect_epochs=3, pretrain_exploration_sigma=0.2,
                          pretrain_steps=500, pretrain_batch=20, per_episode_after_buffer_epochs=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    # 2026-11-22: short local trend runs (user request: does the loss go down?) - the default critic pretraining of 10 epochs
    # (3 train episodes x ~3.5 min each) would dominate a 3-epoch run, so it can be overridden; defaults = previous behaviour.
    p.add_argument("--pretrain_epochs", type=int, default=None, help="critic pretrain epochs (default: trial.CRITIC_PRETRAIN_EPOCHS, 0 with --smoke)")
    p.add_argument("--val_every", type=int, default=None, help="validate every N epochs (default: trial.VAL_EVERY_N_EPOCHS)")
    # 2026-11-22: the 3-epoch run without critic pretraining collapsed (all trip scores negative -> ILP rejects everything,
    # actor gradient ~7e-5); the actor inputs were large (mean |x| 50-65, max up to 4,500: raw seconds, see chat). The three
    # feat_builder.py switches from the SF feature check (2026-11-17) rescale them; same switches as run_srl_sf_uniformroad_bi900_ss450.py.
    # Default OFF = the features of the earlier NYC runs, so runs with/without the flag differ only in the features.
    p.add_argument("--feature_fixes", action="store_true",
                   help="FeatureBuilder.FIX_CAPACITY_FEATURES / FIX_TIME_NORMALIZATION / FIX_FEATURE_SCALING = True")
    # 2026-11-23: optional suffix for the output folder (e.g. "_scoretrace") so a rerun with the same flags never overwrites an earlier run.
    p.add_argument("--tag", type=str, default="", help="suffix appended to run_id before the seed")
    # 2026-11-23: actor learning rate override (user request: lower actor_lr as cluster runs after the actor sign collapse, see chat);
    # default None = ACTOR_LR (0.00319, the baseline used by all earlier NYC runs), so existing invocations are unchanged.
    p.add_argument("--actor_lr", type=float, default=None, help="actor learning rate (default: ACTOR_LR baseline 0.0031923...)")
    # 2026-11-23: buffer critic pretraining, same kwargs as run_srl_sf_uniformroad_bi900_ss450.py --buffer_pretrain (all train
    # instances, 3 collect passes with sigma 0.2, 500 steps, batch 20, then 1 per-episode epoch). Reason: diag run
    # outputs/srl_nyc/diag_target_action_ep1_bufpre_val1_ff_save_s42 (seed 42, 1 epoch): val 91.8 %, target critics corr(Q, requests)
    # +0.73/+0.76, whereas the 5 per-episode pretrain epochs (15 critic steps) gave a random-sign critic and a 0 % collapse.
    p.add_argument("--buffer_pretrain", action="store_true")
    a = p.parse_args()
    if a.buffer_pretrain and (a.pretrain_epochs is not None or a.smoke):
        p.error("--buffer_pretrain cannot be combined with --pretrain_epochs / --smoke")  # 2026-11-23
    actor_lr = ACTOR_LR if a.actor_lr is None else a.actor_lr

    if a.feature_fixes:
        _FB = trial._feat_builder_module.FeatureBuilder
        _FB.FIX_CAPACITY_FEATURES = True
        _FB.FIX_TIME_NORMALIZATION = True
        _FB.FIX_FEATURE_SCALING = True

    epochs = 1 if a.smoke else a.epochs
    pretrain = 0 if a.smoke else (trial.CRITIC_PRETRAIN_EPOCHS if a.pretrain_epochs is None else a.pretrain_epochs)
    if a.buffer_pretrain:
        pretrain = 0  # 2026-11-23: ignored by the loop in buffer mode (it runs per_episode_after_buffer_epochs instead); 0 so result.json is not misleading
    val_every = trial.VAL_EVERY_N_EPOCHS if a.val_every is None else a.val_every
    ff = "_ff" if a.feature_fixes else ""  # 2026-11-22: feature-fix runs get their own output folder
    pre_tag = "bufpre" if a.buffer_pretrain else f"pre{pretrain}"  # 2026-11-23: bufpre runs get their own folder name
    run_id = f"nyc_bi900_ss450_mc3_{'smoke' if a.smoke else f'ep{epochs}_{pre_tag}_val{val_every}'}{ff}{a.tag}_s{a.seed}"
    output_dir = loop.REPO_ROOT / "outputs" / "srl_nyc" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for m in (TRAIN + VAL):
        assert (NYC_MANIFEST_DIR / f"{m}.json").exists(), m
    print(f"=== {run_id}: train={TRAIN} val={VAL} epochs={epochs} pretrain={pretrain} actor_lr={actor_lr} "
          f"critic_lr={trial.CRITIC_LR} feature_size={trial._feat_builder_module.FeatureBuilder.FEATURE_SIZE} ===", flush=True)

    t0 = time.time()
    result = loop.run_srl_training_loop(
        reward_mode=trial.REWARD_MODE, actor_lr=actor_lr, critic_lr=trial.CRITIC_LR,
        output_dir=output_dir, actor_checkpoint=trial.ACTOR_CHECKPOINT,
        gamma=trial.GAMMA, tau=trial.TAU, sigma=trial.SIGMA, num_samples=trial.NUM_SAMPLES,
        actor_clip_norm=trial.ACTOR_CLIP, critic_clip_norm=trial.CRITIC_CLIP,
        epochs=epochs, val_every_n_epochs=val_every, critic_pretrain_epochs=pretrain,
        batch_interval=BATCH_INTERVAL, step_size=STEP_SIZE, seed=a.seed, max_cardinality=MAX_CARDINALITY,
        use_twin_critic=True, use_actor_warmstart=False,
        actor_update_mode="buffered", actor_batch_size=trial.ACTOR_BATCH_SIZE,
        **(BUFFER_PRETRAIN_KW if a.buffer_pretrain else {}),  # 2026-11-23
    )
    wall = time.time() - t0
    out = {
        "run_id": run_id, "train": TRAIN, "val": VAL, "batch_interval": BATCH_INTERVAL, "step_size": STEP_SIZE,
        "max_cardinality": MAX_CARDINALITY, "time_rule_filter": False, "rtv_timeout": 1800,
        "epochs": epochs, "critic_pretrain_epochs": pretrain,
        "buffer_pretrain": BUFFER_PRETRAIN_KW if a.buffer_pretrain else None,  # 2026-11-23 "val_every_n_epochs": val_every, "feature_fixes": bool(a.feature_fixes), "actor_lr": actor_lr, "critic_lr": trial.CRITIC_LR,
        "seed": a.seed, "best_epoch": result.best_epoch, "best_val_service_rate": result.best_val_service_rate,
        "val_curve": result.val_curve, "overfit_curve": result.overfit_curve, "wall_s": round(wall, 1),
    }
    (output_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"=== {run_id} DONE: best val {result.best_val_service_rate:.4f} @ epoch {result.best_epoch}, wall {wall:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
