"""
2026-10-13: per-instance TEST service rates of the config-D runs bestD_s301..s304 (MLP actor), recomputed from each run's best_actor_checkpoint.pt
(result.json only holds the pooled test mean). Same protocol as compute_srl_test_per_instance.py: one eval pass over the 9 TEST_INSTANCES per
run, batch_interval=200/step_size=100, KEEP_ACTIVE=False. Output: outputs/srl_test_per_instance_D/<run>.json (own folder, the older
compute script and its outputs are untouched). Usage: SRL_MAX_THREAD_CNT=2 ./venv/bin/python3 scripts/analysis/compute_bestD_test_per_instance.py
"""
import json
import time
from pathlib import Path

import torch

# import first: patches the FeatureBuilder pickup_slack flags (85 features)
from rtv_solver.pipeline import srl_buffered_actor_lr_trial  # noqa: F401
from rtv_solver.pipeline.srl_training_loop import _per_instance_service_rates, REPO_ROOT
from rtv_solver.pipeline.srl_train_val_test_split import TEST_INSTANCES
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline import select_feature_builder_class
from rtv_solver.structure.config import Config

SWEEP = REPO_ROOT / "outputs" / "srl_training_sweep"
OUT = REPO_ROOT / "outputs" / "srl_test_per_instance_D"
OUT.mkdir(parents=True, exist_ok=True)
scratch = OUT / "_eval_dirs"

for seed in (301, 302, 303, 304):
    name = f"bestD_s{seed}"
    target = OUT / f"{name}.json"
    if target.exists():
        continue
    d = json.load(open(SWEEP / name / "result.json"))
    t0 = time.time()
    cfg = Config(OUTPUT_DIR=scratch / name, BATCH_INTERVAL=200, STEP_SIZE=100, SEED=d["seed"], KEEP_ACTIVE=False)
    fb = select_feature_builder_class(cfg)
    model = build_scoring_model("mlp", feature_dim=fb.FEATURE_SIZE, hidden_dim=64)
    model.load_state_dict(torch.load(SWEEP / name / "best_actor_checkpoint.pt", map_location="cpu")["model_state_dict"])
    rates = _per_instance_service_rates(TEST_INSTANCES, model, cfg, scratch / name, epoch_num=d["best_epoch"], tag="test")
    pooled = sum(rates.values()) / max(len(rates), 1)
    json.dump({"run": name, "seed": d["seed"], "best_epoch": d["best_epoch"], "original_test_service_rate": d["test_service_rate"],
               "recomputed_test_service_rate": pooled, "per_instance": rates}, open(target, "w"), indent=1)
    print(f"DONE {name}: original {d['test_service_rate']:.4f} recomputed {pooled:.4f} ({len(rates)} instances, {time.time()-t0:.0f}s)", flush=True)
