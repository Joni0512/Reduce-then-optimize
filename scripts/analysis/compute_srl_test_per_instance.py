"""
2026-10-09: per-instance TEST service rates of the buffered SRL runs, recomputed from each run's saved best_actor_checkpoint.pt
(the original trial scripts only stored the pooled test mean in result.json). One eval pass over the 9 TEST_INSTANCES per run,
batch_interval=200/step_size=100, KEEP_ACTIVE=False (same as validation/test in srl_training_loop.py). Only runs whose checkpoint
is available locally (the 8 local runs); cluster runs would need to be done there. Results -> outputs/srl_test_per_instance/<run>.json.
The recomputed pooled mean is stored next to the original test_service_rate as a sanity check (Gurobi is not fully deterministic).
Usage: ./venv/bin/python3 scripts/analysis/compute_srl_test_per_instance.py
"""
import json
import sys
import time
from pathlib import Path

import torch

# import first: patches the FeatureBuilder pickup_slack flags (85 features) after srl_training_loop's own patch
from rtv_solver.pipeline import srl_buffered_actor_lr_trial  # noqa: F401
from rtv_solver.pipeline.srl_training_loop import _per_instance_service_rates, REPO_ROOT
from rtv_solver.pipeline.srl_train_val_test_split import TEST_INSTANCES
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline import select_feature_builder_class
from rtv_solver.structure.config import Config

SWEEP = REPO_ROOT / "outputs" / "srl_training_sweep"
OUT = REPO_ROOT / "outputs" / "srl_test_per_instance"
OUT.mkdir(parents=True, exist_ok=True)
scratch = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "_eval_dirs"

for result_path in sorted(SWEEP.glob("buf*/result.json")):
    name = result_path.parent.name
    target = OUT / f"{name}.json"
    if target.exists():
        continue
    d = json.load(open(result_path))
    ckpt_path = result_path.parent / "best_actor_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"SKIP {name}: no checkpoint", flush=True)
        continue
    t0 = time.time()
    cfg = Config(OUTPUT_DIR=scratch / name, BATCH_INTERVAL=200, STEP_SIZE=100, SEED=d["seed"], KEEP_ACTIVE=False)
    fb = select_feature_builder_class(cfg)
    model = build_scoring_model("mlp", feature_dim=fb.FEATURE_SIZE, hidden_dim=64)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state_dict"])
    rates = _per_instance_service_rates(TEST_INSTANCES, model, cfg, scratch / name, epoch_num=d["best_epoch"], tag="test")
    pooled = sum(rates.values()) / max(len(rates), 1)
    json.dump({"run": name, "actor_lr": d["actor_lr"], "critic_lr": d["critic_lr"], "seed": d["seed"], "best_epoch": d["best_epoch"],
               "original_test_service_rate": d["test_service_rate"], "recomputed_test_service_rate": pooled, "per_instance": rates,
               "n_instances": len(rates)}, open(target, "w"), indent=1)
    # 2026-10-09: compact per-instance line so the numbers can be copied out of a Slurm log (cluster runs)
    print(f"PERINST {name} alr {d['actor_lr']} clr {round(d['critic_lr'], 5)} seed {d['seed']} best_ep {d['best_epoch']} " + json.dumps({k: round(v, 4) for k, v in rates.items()}), flush=True)
    print(f"DONE {name}: original {d['test_service_rate']:.4f} recomputed {pooled:.4f} ({len(rates)} instances, {time.time()-t0:.0f}s)", flush=True)
