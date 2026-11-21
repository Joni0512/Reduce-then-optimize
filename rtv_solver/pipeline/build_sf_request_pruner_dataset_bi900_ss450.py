"""
2026-11-19: SF request-pruner dataset, batch_interval=900/step_size=450 (user: "Request Pruner fuer SF ... nur 900/450",
label "pickup vor Re-Solve"). Same builder as the NYC pickup datasets (dataset/request_pruning_nyc_pickup_bi*), i.e.
request_pruner_dataset_builder.build_instance_arrays(label_mode="pickup_before_next_resolve", geographic=True):
  label 1.0 iff the request is assigned in this window AND its pickup in the run's final/result_driver_runs.json is
  earlier than window t + step_size; else 0.0.
Label source = RHO runs WITHOUT any pruner, start seed 42 (uniform road-node starts), 150 vehicles, max_cardinality=2,
time rule off, no depot return, rtv_timeout 1800 - the existing pair-pruner label runs:
  train: instance seed 1, val: instance seed 2 (same split as the SF pair pruner; seed 42 test instance never touched).
Pre-check of these labels: scripts/analysis/sf_request_pruner_label_check.py (41-42 % positive, P(1 | time rule prunes) = 0).
More train instances (seeds 3-11) would need new RHO runs on the cluster - not done here.
Output: dataset/request_pruning_sf_pickup_bi900_ss450/{train,val}/<instance>.npz + manifest.json
Run: ./venv/bin/python3 -m rtv_solver.pipeline.build_sf_request_pruner_dataset_bi900_ss450
"""
import glob
import json
import time
from pathlib import Path

import numpy as np

from rtv_solver.pipeline.request_pruner_dataset_builder import FEATURE_NAMES, build_instance_arrays
from rtv_solver.structure.config import Config

BATCH_INTERVAL, STEP_SIZE = 900, 450
MANIFEST_DIR = Path("solutions/sf/manifests")
OUT = Path(f"dataset/request_pruning_sf_pickup_bi{BATCH_INTERVAL}_ss{STEP_SIZE}")
_M = "sf_lodes1000_0608_seed{}_v150_cap2_vstart_uniformroad_s42_road_solve"
RUNS = {  # split, instance -> run dir of the RHO baseline (assignment_data.jsonl + final/result_driver_runs.json)
    ("train", _M.format(1)): "outputs/new_tests/sf_lodes_nodepot/inst1_bi900_ss450_mc2_v150_notimerule_noreturn_uniformroad_s42",
    ("val", _M.format(2)): "outputs/new_tests/sf_lodes_nodepot/inst2_bi900_ss450_mc2_v150_notimerule_noreturn_uniformroad_s42",
}
# same SF Config as the SF RHO / SRL runs (run_srl_sf_uniformroad_bi900_ss450.py)
CONFIG = Config(USE_REQUEST_TIME_RULE_FILTER=False, MAX_CARDINALITY=2, RETURN_DEPOT=False, RTV_TIMEOUT=1800)


def main() -> None:
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geographic": True,
        "config": f"batch_interval={BATCH_INTERVAL}, step_size={STEP_SIZE}, max_cardinality=2, 150 vehicles, uniform road-node "
                  "starts start seed 42, time rule off, no depot return, rtv_timeout 1800, RHO baseline without any pruner",
        "label_definition": "1.0 iff the request is assigned in this window AND its pickup (scheduled_time of the pickup stop in "
                            "final/result_driver_runs.json) < window timestamp + step_size; else 0.0 (unserved in this window, or "
                            "assigned but picked up only after the next re-solve, or no pickup stop in the final plan).",
        "feature_names": FEATURE_NAMES,
        "normalization": "NOT applied - raw features; normalize at train time with TRAIN statistics only.",
        "note": "Test instance seed 42 (and 15-17) not built. Only 1 train / 1 val instance (existing pair-pruner label runs).",
        "instances": {},
    }
    for (split, inst), run_top in RUNS.items():
        logs = glob.glob(f"{run_top}/run_offline_*/{inst}_*/assignment_data.jsonl")
        assert len(logs) == 1, (inst, logs)
        log = Path(logs[0])
        assert (log.parent / "final" / "result_driver_runs.json").exists(), log
        arr = build_instance_arrays(inst, CONFIG, geographic=True, manifest_dir=MANIFEST_DIR, log_path=log,
                                    label_mode="pickup_before_next_resolve", step_size=STEP_SIZE)
        (OUT / split).mkdir(parents=True, exist_ok=True)
        np.savez(OUT / split / f"{inst}.npz", node_ids=arr["node_ids"], window_index=arr["window_index"],
                 timestamp=arr["timestamp"], label=arr["label"], features=arr["features"])
        manifest["instances"][inst] = {"split": split, "source_run": str(log.parent), "num_windows": arr["num_windows"],
                                       "num_rows": int(len(arr["node_ids"])), "positive_rate": float(arr["label"].mean())}
        print(split, inst, manifest["instances"][inst])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
