"""
2026-11-11: first SRL run on the SF LODES instances (user request, see chat): batch_interval=900/step_size=450,
time-rule request filter ON, max_cardinality=3 (= the SF RHO reference run bi900_ss450 with filter, 47.9 % served,
97 s solver time), 50 vehicles, am_capacity 2, OSMnx free-flow road matrix.

Reason for a separate script: srl_training_loop.py is hard-wired to Li&Lim (MANIFEST_DIR, instance lists) and builds its
Configs without USE_REQUEST_TIME_RULE_FILTER, with MAX_CARDINALITY only in the main-loop training Config (pretrain and
validation fall back to Config's default 2) and with Config's RTV_TIMEOUT default 120 s (main.py/RHO uses 1800 s).
srl_training_loop.py itself is NOT changed; this script patches its module globals at import time (same pattern as the
FeatureBuilder pickup_slack patch in srl_buffered_actor_lr_trial.py):
  * MANIFEST_DIR -> solutions/sf/manifests; TRAIN/VAL/OVERFIT_CHECK instance lists -> SF manifest stems
  * Config -> functools.partial(Config, USE_REQUEST_TIME_RULE_FILTER=True, MAX_CARDINALITY=3, RTV_TIMEOUT=1800)
    (call-site keywords still win, e.g. the main loop's own MAX_CARDINALITY=max_cardinality, which is also passed as 3)
SRL protocol = srl_buffered_actor_lr_trial.py (buffered actor update, batch 16, twin critic, no warmstart, pickup_slack,
local_positive reward, gamma 0.99, tau 0.005, sigma 0.2, 20 samples, critic_lr baseline 0.0010833586558285635,
actor_lr baseline 0.0031923396291543868), all imported from there so the values cannot drift.
Split: train = SF seed 1 (more seeds later), val = SF seed 2; test instance seed 42 is NOT touched by this script.

--smoke: 1 main epoch, no critic pretraining, validation once at the end -> checks that SRL runs on SF at all and
measures the time of one training episode + one validation episode (decides how many train instances/epochs are feasible).

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.run_srl_sf_lodes_bi900_ss450 --smoke
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
SF_MANIFEST_DIR = loop.REPO_ROOT / "solutions" / "sf" / "manifests"
TRAIN = ["sf_lodes1000_0608_seed1_v50_cap2_road_solve"]
VAL = ["sf_lodes1000_0608_seed2_v50_cap2_road_solve"]
ACTOR_LR = 0.0031923396291543868  # 1x baseline of the buffered actor_lr scan (srl_buffered_actor_lr_trial.py docstring)

_SFConfig = functools.partial(Config, USE_REQUEST_TIME_RULE_FILTER=True, MAX_CARDINALITY=MAX_CARDINALITY, RTV_TIMEOUT=1800)
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
    a = p.parse_args()

    epochs = 1 if a.smoke else a.epochs
    pretrain = 0 if a.smoke else trial.CRITIC_PRETRAIN_EPOCHS
    run_id = f"sf_lodes_bi900_ss450_mc3_timerule_{'smoke' if a.smoke else f'ep{epochs}'}_s{a.seed}"
    output_dir = loop.REPO_ROOT / "outputs" / "srl_sf_lodes" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for m in (TRAIN + VAL):
        assert (SF_MANIFEST_DIR / f"{m}.json").exists(), m
    print(f"=== {run_id}: train={TRAIN} val={VAL} epochs={epochs} pretrain={pretrain} actor_lr={ACTOR_LR} "
          f"critic_lr={trial.CRITIC_LR} feature_size={trial._feat_builder_module.FeatureBuilder.FEATURE_SIZE} ===", flush=True)

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
    )
    wall = time.time() - t0
    out = {
        "run_id": run_id, "train": TRAIN, "val": VAL, "batch_interval": BATCH_INTERVAL, "step_size": STEP_SIZE,
        "max_cardinality": MAX_CARDINALITY, "time_rule_filter": True, "rtv_timeout": 1800,
        "epochs": epochs, "critic_pretrain_epochs": pretrain, "actor_lr": ACTOR_LR, "critic_lr": trial.CRITIC_LR,
        "seed": a.seed, "best_epoch": result.best_epoch, "best_val_service_rate": result.best_val_service_rate,
        "val_curve": result.val_curve, "overfit_curve": result.overfit_curve, "wall_s": round(wall, 1),
    }
    (output_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"=== {run_id} DONE: best val {result.best_val_service_rate:.4f} @ epoch {result.best_epoch}, wall {wall:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
