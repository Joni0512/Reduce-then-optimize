"""
2026-11-05: NYC request-pruner dataset with the label "pickup_before_next_resolve", one dataset per
window config (rule: one request pruner per batch_interval/step_size, never reuse another config's).

Reason: the old label "assigned in this window" is 94-97 % positive at batch_interval=3600/step_size=1800
(99.4 % at 1800/900, 95.6-96.5 % at 900/450), i.e. nothing worth pruning - the same problem that made
build_nyc_request_pruner_dataset.py switch to 480/240 in August. New label (see
request_pruner_dataset_builder.build_instance_arrays): 1.0 iff the request is assigned in this window AND
its pickup in the run's final plan is earlier than timestamp + step_size.

Only Train (2016-01-12) and Val (2016-01-13) are built. The Test day (2016-01-19) stays untouched.
The source runs are the no-pruner RHO baselines (offline, cardinality 3, seed 1) of that config; the log
directory of each run must contain assignment_data.jsonl and final/result_driver_runs.json.

Run, e.g. (3600/1800, default run location of the local baseline):
    python3 -m build_nyc_request_pruner_dataset_pickup_label --batch_interval 3600 --step_size 1800
For another config pass --run_glob with {split}, e.g. for data copied from the cluster:
    --run_glob "outputs/outputs/new_tests/nyc_rho_baseline_bi900_ss450/{split}_seed1/*/*"
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from rtv_solver.pipeline.request_pruner_dataset_builder import build_instance_arrays, FEATURE_NAMES
from rtv_solver.structure.config import Config

MANIFEST_DIR = Path("solutions/nyc/manifests")
INSTANCES = {
    "train": "nyc_real1000_20160112_0614_train_v50_expert",
    "val": "nyc_real1000_20160113_0614_val_v50_expert",
}
DEFAULT_RUN_GLOB = ("outputs/new_tests/nyc_30min_60min_baseline/{split}_seed1/"
                    "run_offline_mc3_bi3600_ss1800/*")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_interval", type=int, required=True)
    parser.add_argument("--step_size", type=int, required=True)
    parser.add_argument("--run_glob", type=str, default=DEFAULT_RUN_GLOB,
                        help="glob with {split} -> directory that contains assignment_data.jsonl and final/")
    args = parser.parse_args()

    output_dir = Path(f"dataset/request_pruning_nyc_pickup_bi{args.batch_interval}_ss{args.step_size}")
    config = Config()
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geographic": True,
        "config": f"batch_interval={args.batch_interval}, step_size={args.step_size}, max_cardinality=3, seed 1, "
                  "RHO baseline without any pruner",
        "label_definition": (
            "1.0 iff the request is assigned in this window AND its pickup (scheduled_time of the pickup stop in "
            "final/result_driver_runs.json) < window timestamp + step_size; else 0.0 (unserved in this window, "
            "or assigned but picked up only after the next re-solve, or no pickup stop in the final plan)."
        ),
        "feature_names": FEATURE_NAMES,
        "note": "Test day not built (kept untouched). local_request_count_10min is constant 0.0 (geographic=True).",
        "instances": {},
    }

    for split, instance_name in INSTANCES.items():
        run_dirs = [p for p in glob.glob(args.run_glob.format(split=split))
                    if Path(p, "assignment_data.jsonl").exists()]
        if len(run_dirs) != 1:
            raise SystemExit(f"{split}: expected exactly 1 run dir with assignment_data.jsonl, found {run_dirs}")
        log_path = Path(run_dirs[0]) / "assignment_data.jsonl"

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"  building {split}/{instance_name} from {log_path} ...")
        arrays = build_instance_arrays(
            instance_name, config, geographic=True, manifest_dir=MANIFEST_DIR, log_path=log_path,
            label_mode="pickup_before_next_resolve", step_size=args.step_size,
        )
        if arrays is None:
            raise SystemExit(f"no arrays produced for {instance_name}")

        np.savez(split_dir / f"{instance_name}.npz", node_ids=arrays["node_ids"],
                 window_index=arrays["window_index"], timestamp=arrays["timestamp"],
                 label=arrays["label"], features=arrays["features"])
        manifest["instances"][instance_name] = {
            "split": split,
            "source_run": str(log_path.parent),
            "num_windows": arrays["num_windows"],
            "num_rows": len(arrays["node_ids"]),
            "positive_rate": float(arrays["label"].mean()),
        }
        info = manifest["instances"][instance_name]
        print(f"    -> {info['num_windows']} windows, {info['num_rows']} rows, positive_rate={info['positive_rate']:.3f}")

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
