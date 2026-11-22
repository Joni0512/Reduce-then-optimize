"""
2026-11-18: SRL on the CURRENT SF setup (user: "SRL bei diesen Daten", window "900/450 lokal"): batch_interval=900/
step_size=450, 150 vehicles with uniform road-node starts (start seed 42), max_cardinality=2, am_capacity 2, time-rule
request filter OFF, no depot return, rtv_timeout 1800 s = the SF RHO / pair-pruner setup (RHO on test seed 42: 90.3 %).
Copy of run_srl_sf_lodes_bi900_ss450.py (old setup: 50 vehicles at the depot, mc3, filter ON - that run collapsed to 0 %
val service rate from epoch 10 on; the old script stays as the record). Differences only:
  * manifests: sf_lodes1000_0608_seed{1,2}_v150_cap2_vstart_uniformroad_s42_road_solve (train seed 1, val seed 2;
    test instance seed 42 is NOT touched by this script)
  * Config partial: MAX_CARDINALITY=2, USE_REQUEST_TIME_RULE_FILTER=False, RETURN_DEPOT=False, RTV_TIMEOUT=1800
  * actor feature fixes ON (feat_builder.py, 2026-11-17, SF feature check): FIX_CAPACITY_FEATURES,
    FIX_TIME_NORMALIZATION, FIX_FEATURE_SCALING - set right after the trial import (which re-patches pickup_slack)
SRL protocol unchanged = srl_buffered_actor_lr_trial.py (buffered actor update, batch 16, twin critic, no warmstart,
pickup_slack, local_positive reward, gamma 0.99, tau 0.005, sigma 0.2, 20 samples, critic_lr / actor_lr baselines), all
imported from there so the values cannot drift.
--smoke: 1 main epoch, no critic pretraining, validation once at the end -> does SRL run on this setup, and how long
does one training + one validation episode take (decides epochs / instances for real runs).
Usage: ./venv/bin/python3 -m rtv_solver.pipeline.run_srl_sf_uniformroad_bi900_ss450 --smoke
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

# 2026-11-18: feature fixes on for this (new, from-scratch) SF line - default off elsewhere for old checkpoints
_FB = trial._feat_builder_module.FeatureBuilder
_FB.FIX_CAPACITY_FEATURES = True
_FB.FIX_TIME_NORMALIZATION = True
_FB.FIX_FEATURE_SCALING = True

BATCH_INTERVAL, STEP_SIZE, MAX_CARDINALITY = 900, 450, 2
SF_MANIFEST_DIR = loop.REPO_ROOT / "solutions" / "sf" / "manifests"
# 2026-11-18: split 10 train / 4 val / 4 test (user), disjoint request draws; test = seeds 42, 15-17 (never used here)
_M = "sf_lodes1000_0608_seed{}_v150_cap2_vstart_uniformroad_s42_road_solve"
TRAIN = [_M.format(s) for s in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11)]
VAL = [_M.format(s) for s in (2, 12, 13, 14)]
ACTOR_LR = 0.0031923396291543868  # 1x baseline of the buffered actor_lr scan (srl_buffered_actor_lr_trial.py docstring)

_SFConfig = functools.partial(Config, USE_REQUEST_TIME_RULE_FILTER=False, MAX_CARDINALITY=MAX_CARDINALITY,
                              RETURN_DEPOT=False, RTV_TIMEOUT=1800)
loop.MANIFEST_DIR = SF_MANIFEST_DIR
loop.TRAIN_INSTANCES = list(TRAIN)
loop.VAL_INSTANCES = list(VAL)
loop.OVERFIT_CHECK_INSTANCES = list(TRAIN)
loop.Config = _SFConfig
buffer_pretrain.Config = _SFConfig  # only used with pretrain_mode="buffer"; patched so it cannot silently fall back


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    # 2026-11-18: critic pretraining only (CRITIC_PRETRAIN_EPOCHS per-episode epochs, twin critic, 0 actor epochs) to
    # check on SF whether the critic loss falls and stays stable before a full run (user: "A lokal")
    p.add_argument("--pretrain_only", action="store_true")
    # 2026-11-18: buffer critic pretrain (all 10 train instances, 3 collect passes with sigma 0.2 noise, 500 steps, batch 20),
    # then 1 epoch of the old per-episode pretrain (frozen actor, target r_t) = 10 episodes per critic (user request)
    p.add_argument("--buffer_pretrain", action="store_true")
    # 2026-11-22: no critic pretraining at all, main training starts directly with a randomly initialized critic
    # (user: "kein Critic-Pretraining, sondern gleich 10 Epochen Haupttraining", wants a faster run)
    p.add_argument("--no_pretrain", action="store_true")
    a = p.parse_args()
    if a.no_pretrain and (a.buffer_pretrain or a.pretrain_only):
        p.error("--no_pretrain cannot be combined with --buffer_pretrain / --pretrain_only")  # 2026-11-22

    epochs = 1 if a.smoke else (0 if a.pretrain_only else a.epochs)
    pretrain = 0 if (a.smoke or a.no_pretrain) else trial.CRITIC_PRETRAIN_EPOCHS  # 2026-11-22: + no_pretrain
    # 2026-11-22: num_samples tag only when != 20 (default), so existing run dirs keep their names and the num_samples=10 run
    # (user: "num_samples 20 -> 10", seed 42) does not overwrite the dir of the still running num_samples=20 job 5743969
    ns_tag = "" if trial.NUM_SAMPLES == 20 else f"ns{trial.NUM_SAMPLES}_"
    run_id = (f"sf_uniformroad_bi900_ss450_mc2_v150_featfix_split10_{'bufpre_' if a.buffer_pretrain else ''}{'nopre_' if a.no_pretrain else ''}{ns_tag}"
              f"{'smoke' if a.smoke else ('pretrainonly' if a.pretrain_only else f'ep{epochs}')}_s{a.seed}")  # 2026-11-18: split10 / bufpre tags
    output_dir = loop.REPO_ROOT / "outputs" / "srl_sf_lodes" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for m in (TRAIN if a.pretrain_only else TRAIN + VAL):  # 2026-11-18: pretrain-only needs no val manifests
        assert (SF_MANIFEST_DIR / f"{m}.json").exists(), m
    print(f"=== {run_id}: train={TRAIN} val={VAL} epochs={epochs} pretrain={pretrain} actor_lr={ACTOR_LR} "
          f"critic_lr={trial.CRITIC_LR} feature_size={_FB.FEATURE_SIZE} fixes=capacity,time_norm,scaling ===", flush=True)

    t0 = time.time()
    result = loop.run_srl_training_loop(
        reward_mode=trial.REWARD_MODE, actor_lr=ACTOR_LR, critic_lr=trial.CRITIC_LR,
        output_dir=output_dir, actor_checkpoint=trial.ACTOR_CHECKPOINT,
        gamma=trial.GAMMA, tau=trial.TAU, sigma=trial.SIGMA, num_samples=trial.NUM_SAMPLES,
        actor_clip_norm=trial.ACTOR_CLIP, critic_clip_norm=trial.CRITIC_CLIP,
        epochs=epochs, val_every_n_epochs=trial.VAL_EVERY_N_EPOCHS, critic_pretrain_epochs=pretrain,
        batch_interval=BATCH_INTERVAL, step_size=STEP_SIZE, seed=a.seed, max_cardinality=MAX_CARDINALITY,
        use_twin_critic=True, use_actor_warmstart=False,
        actor_update_mode="buffered", actor_batch_size=trial.ACTOR_BATCH_SIZE,
        **(dict(pretrain_mode="buffer", pretrain_instances=["all"], pretrain_collect_epochs=3, pretrain_exploration_sigma=0.2,
                pretrain_steps=500, pretrain_batch=20, per_episode_after_buffer_epochs=1) if a.buffer_pretrain else {}),  # 2026-11-18
    )
    wall = time.time() - t0
    out = {
        "run_id": run_id, "train": TRAIN, "val": VAL, "batch_interval": BATCH_INTERVAL, "step_size": STEP_SIZE,
        "max_cardinality": MAX_CARDINALITY, "vehicles": 150, "vehicle_starts": "uniform road nodes, start seed 42",
        "time_rule_filter": False, "return_depot": False, "rtv_timeout": 1800,
        "feature_fixes": ["FIX_CAPACITY_FEATURES", "FIX_TIME_NORMALIZATION", "FIX_FEATURE_SCALING"],
        "epochs": epochs, "critic_pretrain_epochs": pretrain,
        "buffer_pretrain": (dict(collect_epochs=3, exploration_sigma=0.2, steps=500, batch=20, per_episode_after_buffer_epochs=1)
                            if a.buffer_pretrain else None),  # 2026-11-18
        "actor_lr": ACTOR_LR, "critic_lr": trial.CRITIC_LR,
        "num_samples": trial.NUM_SAMPLES,  # 2026-11-22: recorded explicitly (SRL_NUM_SAMPLES env, ns10 run)
        "seed": a.seed, "best_epoch": result.best_epoch, "best_val_service_rate": result.best_val_service_rate,
        "val_curve": result.val_curve, "overfit_curve": result.overfit_curve, "wall_s": round(wall, 1),
    }
    (output_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"=== {run_id} DONE: best val {result.best_val_service_rate:.4f} @ epoch {result.best_epoch}, wall {wall:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
