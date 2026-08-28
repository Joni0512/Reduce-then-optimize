"""
2026-08-27: NYC request-pruner dataset build (Phase 0, step 2 of the NYC
RequestPruner readiness plan). Analogous to build_and_train_request_pruner_bi*.py
but for NYC's 3-day split instead of Li&Lim's 56-instance mixed_all split, and
using geographic=True (haversine) node features - see
request_pruner_dataset_builder.py's manifest_dir/log_path parameters, added
for exactly this caller.

Baseline logs used (USE_REQUEST_GRAPH_PRUNER=false, USE_REQUEST_PRUNER=false,
max_cardinality=3, step_size=240s/4min, batch_interval=480s/8min).

2026-08-27: switched from the initially-planned 10min/40min config (best
service rate, 98.82%) to 4min/8min after building the dataset once and
finding the label ("assigned this window") was ~99% positive at 10min/40min
- too little negative-class signal for the pruner to learn a meaningful
decision boundary. 4min/8min's lower service rate (~78.5%, see the
robustness sweep) means far more requests roll over to the next window
unserved, giving a much more balanced label distribution - see the
positive_rate values printed/stored below.
    Train (2016-01-12): seed 1 of the existing robustness-sweep runs
    Val   (2016-01-13): fresh baseline run, seed 1
    Test  (2016-01-19): fresh baseline run, seed 1
"""
from pathlib import Path

from rtv_solver.pipeline.request_pruner_dataset_builder import build_instance_arrays, FEATURE_NAMES
from rtv_solver.structure.config import Config

import numpy as np
import json
import time

MANIFEST_DIR = Path("solutions/nyc/manifests")
OUTPUT_DIR = Path("dataset/request_pruning_nyc")

INSTANCES = {
    "train": (
        "nyc_real1000_20160112_0614_train_v50_expert",
        Path("outputs/new_tests/nyc_stepsize_sweep/robustness/ss240_bi480_seed1_train/"
             "run_offline_mc3_bi480_ss240/nyc_real1000_20160112_0614_train_v50_expert_20260819_235630/"
             "assignment_data.jsonl"),
    ),
    "val": (
        "nyc_real1000_20160113_0614_val_v50_expert",
        Path("outputs/new_tests/nyc_stepsize_sweep/robustness/ss240_bi480_seed1_val/"
             "run_offline_mc3_bi480_ss240/nyc_real1000_20160113_0614_val_v50_expert_20260827_220540/"
             "assignment_data.jsonl"),
    ),
    "test": (
        "nyc_real1000_20160119_0614_test_v50_expert",
        Path("outputs/new_tests/nyc_stepsize_sweep/robustness/ss240_bi480_seed1_test/"
             "run_offline_mc3_bi480_ss240/nyc_real1000_20160119_0614_test_v50_expert_20260827_220547/"
             "assignment_data.jsonl"),
    ),
}


def main():
    config = Config()
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geographic": True,
        "config": "step_size=240 (4min), batch_interval=480 (8min), max_cardinality=3",
        "feature_names": FEATURE_NAMES,
        "note": (
            "local_request_count_10min is a constant 0.0 for this dataset "
            "(geographic=True) - Li&Lim's fixed 10.0-unit spatial threshold "
            "for that feature isn't meaningful for NYC's real coordinates, "
            "no meter-based cutoff has been agreed on yet. See "
            "request_graph_feature_builder.py add_node_features() TODO."
        ),
        "instances": {},
    }

    for split, (instance_name, log_path) in INSTANCES.items():
        split_dir = OUTPUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"  building {split}/{instance_name} ...")
        arrays = build_instance_arrays(
            instance_name,
            config,
            geographic=True,
            manifest_dir=MANIFEST_DIR,
            log_path=log_path,
        )
        if arrays is None:
            print(f"    [skip] no arrays produced for {instance_name}")
            continue

        out_path = split_dir / f"{instance_name}.npz"
        np.savez(
            out_path,
            node_ids=arrays["node_ids"],
            window_index=arrays["window_index"],
            timestamp=arrays["timestamp"],
            label=arrays["label"],
            features=arrays["features"],
        )

        num_rows = len(arrays["node_ids"])
        manifest["instances"][instance_name] = {
            "split": split,
            "num_windows": arrays["num_windows"],
            "num_rows": num_rows,
            "positive_rate": float(arrays["label"].mean()),
        }
        print(f"    -> {out_path} ({arrays['num_windows']} windows, {num_rows} rows, "
              f"positive_rate={manifest['instances'][instance_name]['positive_rate']:.3f})")

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_rows = sum(v["num_rows"] for v in manifest["instances"].values())
    total_windows = sum(v["num_windows"] for v in manifest["instances"].values())
    print(f"\nDONE: {len(manifest['instances'])} instances, {total_windows} windows, {total_rows} rows")
    print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
