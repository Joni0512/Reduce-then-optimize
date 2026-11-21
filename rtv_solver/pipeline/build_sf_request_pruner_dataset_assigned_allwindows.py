"""
2026-11-20: SF request-pruner datasets with the "assigned" label for all 5 SF window configs (user: "probiere dieses
Label bei SF mehrere Time Windows"; decision 2026-11-20: all 5 windows, threshold 0.5, min_retention_fraction 0.0).
Reason: the pickup_before_next_resolve label at batch_interval=900/step_size=450 cost -21.6 pp in RHO (68.66 % vs
90.26 % baseline), worse than the time rule (74.32 %); on SF 95.4 % of the requests the time rule prunes are still
assigned by RHO in that window (figures_export/SF/sf_request_pruner_label_check.csv), and the pickup label calls them 0.
Label (request_pruner_dataset_builder default label_mode="assigned", same as Li&Lim): 1.0 iff the request is in
status["assigned_requests"] of this window's RHO solve, else 0.0.
Label source = the plain-RHO pair-pruner label runs (no pruner, start seed 42 uniform road-node starts, 150 vehicles,
max_cardinality=2, time rule off, no depot return; rtv_timeout 1800 except 3600/1800 with 21600):
  train: instance seed 1, val: instance seed 2 (test instance seed 42 never touched).
If a run dir holds several assignment_data.jsonl (aborted earlier attempts), the largest file is used (same choice as
scripts/analysis/sf_request_pruner_label_check.py, which takes the one with the most Status lines).
Windows whose label logs are missing locally are skipped with a message (1800/900: logs only on the cluster).
Output: dataset/request_pruning_sf_assigned_bi{BI}_ss{SS}/{train,val}/<instance>.npz + manifest.json
Run: ./venv/bin/python3 -m rtv_solver.pipeline.build_sf_request_pruner_dataset_assigned_allwindows
"""
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from rtv_solver.pipeline.request_pruner_dataset_builder import FEATURE_NAMES, build_instance_arrays
from rtv_solver.structure.config import Config

MANIFEST_DIR = Path("solutions/sf/manifests")
TOP = "outputs/new_tests/sf_lodes_nodepot"
_M = "sf_lodes1000_0608_seed{}_v150_cap2_vstart_uniformroad_s42_road_solve"
WINDOWS = [(300, 150, 1800), (600, 300, 1800), (900, 450, 1800), (1800, 900, 1800), (3600, 1800, 21600)]
SPLITS = {"train": 1, "val": 2}


def main() -> None:
    for bi, ss, rtv in WINDOWS:
        suf = "_rtv21600" if rtv == 21600 else ""
        out = Path(f"dataset/request_pruning_sf_assigned_bi{bi}_ss{ss}")
        logs = {}
        for split, inst_seed in SPLITS.items():
            cand = glob.glob(f"{TOP}/inst{inst_seed}_bi{bi}_ss{ss}_mc2_v150_notimerule_noreturn_uniformroad{suf}_s42/"
                             f"run_offline_*/{_M.format(inst_seed)}_*/assignment_data.jsonl")
            if cand:
                logs[split] = Path(max(cand, key=os.path.getsize))
        if len(logs) < 2:
            print(f"SKIP batch_interval={bi}/step_size={ss}: label logs missing locally ({sorted(logs)})")
            continue
        # same SF Config as the SF RHO runs of this window
        config = Config(USE_REQUEST_TIME_RULE_FILTER=False, MAX_CARDINALITY=2, RETURN_DEPOT=False, RTV_TIMEOUT=rtv)
        manifest = {
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "geographic": True,
            "config": f"batch_interval={bi}, step_size={ss}, max_cardinality=2, 150 vehicles, uniform road-node starts start "
                      f"seed 42, time rule off, no depot return, rtv_timeout {rtv}, RHO baseline without any pruner",
            "label_definition": "1.0 iff the request is in assigned_requests of this window's RHO solve, else 0.0 (label_mode='assigned').",
            "feature_names": FEATURE_NAMES,
            "normalization": "NOT applied - raw features; normalize at train time with TRAIN statistics only.",
            "note": "Test instance seed 42 not built. 1 train (seed 1) / 1 val (seed 2) instance.",
            "instances": {},
        }
        for split, log in logs.items():
            inst = _M.format(SPLITS[split])
            arr = build_instance_arrays(inst, config, geographic=True, manifest_dir=MANIFEST_DIR, log_path=log,
                                        label_mode="assigned")
            (out / split).mkdir(parents=True, exist_ok=True)
            np.savez(out / split / f"{inst}.npz", node_ids=arr["node_ids"], window_index=arr["window_index"],
                     timestamp=arr["timestamp"], label=arr["label"], features=arr["features"])
            manifest["instances"][inst] = {"split": split, "source_run": str(log.parent), "num_windows": arr["num_windows"],
                                           "num_rows": int(len(arr["node_ids"])), "positive_rate": float(arr["label"].mean())}
            print(f"batch_interval={bi}/step_size={ss}", split, manifest["instances"][inst])
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
