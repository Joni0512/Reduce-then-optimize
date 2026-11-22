"""
2026-11-22: SF Request Pruner dataset WITH the four vehicle-related candidate-insertion features, label "pickup before next re-solve" (user 2026-11-22: "pickup erstmal"),
10 train / 4 val instances instead of 1 / 1 (user 2026-11-22: "wir koennen ja aus den daten mehr instanzen bauen").
Per row (window, request): the 15 existing columns of request_pruner_dataset_builder.build_instance_arrays (UNCHANGED, label_mode="pickup_before_next_resolve", geographic=True,
same call as build_sf_request_pruner_dataset_bi900_ss450.py) PLUS cand_features [n, 4] from the "CandidateInsertions" records (request_pruner_dataset_builder_cand._candidates_per_window,
UNCHANGED; names cand_min_veh_to_pickup, cand_min_added_cost, cand_min_cost_ratio, cand_n_feasible_vehicles; NaN / 0 when a request has no feasible vehicle, imputed only at training time with TRAIN medians).
Input : RHO baseline runs WITHOUT pruner with --log_candidate_insertions True from submit_sf_lodes_rho_uniformroad_cand_logging.sbatch:
        outputs/new_tests/sf_lodes_cand/inst<N>_bi<BI>_ss<SS>_mc2_v150_notimerule_noreturn_uniformroad_s42/run_offline_*/<instance>_*/{assignment_data.jsonl, final/result_driver_runs.json}
Split : same as run_srl_sf_uniformroad_bi900_ss450.py: train seeds 1,3-11, val seeds 2,12-14; test seeds 42,15-17 are NOT built here (no leakage).
Output: dataset/request_pruning_sf_pickup_bi<BI>_ss<SS>_cand/{train,val}/<instance>.npz (node_ids, window_index, timestamp, label, features [n,15], cand_features [n,4]) + manifest.json
Run   : ./venv/bin/python3 -m rtv_solver.pipeline.build_sf_request_pruner_dataset_cand --bi 900 --ss 450
Note  : the pickup label at 600/300 has NOT been pre-checked yet (sf_request_pruner_label_check.py was run for 900/450); check positive rate / P(1 | time rule prunes) before training.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from rtv_solver.pipeline.request_pruner_dataset_builder import FEATURE_NAMES, build_instance_arrays
from rtv_solver.pipeline.request_pruner_dataset_builder_cand import CAND_FEATURE_NAMES, _candidates_per_window
from rtv_solver.structure.config import Config

MANIFEST_DIR = Path("solutions/sf/manifests")
_M = "sf_lodes1000_0608_seed{}_v150_cap2_vstart_uniformroad_s42_road_solve"
SPLIT = {"train": (1, 3, 4, 5, 6, 7, 8, 9, 10, 11), "val": (2, 12, 13, 14)}
# same SF Config as the SF RHO / SRL runs (run_srl_sf_uniformroad_bi900_ss450.py)
CONFIG = Config(USE_REQUEST_TIME_RULE_FILTER=False, MAX_CARDINALITY=2, RETURN_DEPOT=False, RTV_TIMEOUT=1800)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", type=int, required=True)
    ap.add_argument("--ss", type=int, required=True)
    a = ap.parse_args()
    out_dir = Path(f"dataset/request_pruning_sf_pickup_bi{a.bi}_ss{a.ss}_cand")
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geographic": True,
        "config": f"batch_interval={a.bi}, step_size={a.ss}, max_cardinality=2, 150 vehicles, uniform road-node starts start seed 42, time rule off, no depot return, "
                  "rtv_timeout 1800, RHO baseline without any pruner, --log_candidate_insertions True",
        "label_definition": "1.0 iff the request is assigned in this window AND its pickup (scheduled_time of the pickup stop in final/result_driver_runs.json) < window timestamp + step_size; "
                            "else 0.0 (unserved in this window, or assigned but picked up only after the next re-solve, or no pickup stop in the final plan).",
        "feature_names": FEATURE_NAMES,
        "cand_feature_names": CAND_FEATURE_NAMES,
        "normalization": "NOT applied - raw features; cand_features contain NaN (no feasible vehicle / no ILP record): impute and normalise at train time with TRAIN statistics only.",
        "note": "Test instances seed 42, 15-17 not built.",
        "instances": {},
    }
    for split, seeds in SPLIT.items():
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            inst = _M.format(seed)
            run_top = f"outputs/new_tests/sf_lodes_cand/inst{seed}_bi{a.bi}_ss{a.ss}_mc2_v150_notimerule_noreturn_uniformroad_s42"
            logs = glob.glob(f"{run_top}/run_offline_*/{inst}_*/assignment_data.jsonl")
            assert len(logs) == 1, (inst, logs)
            log = Path(logs[0])
            assert (log.parent / "final" / "result_driver_runs.json").exists(), log
            arr = build_instance_arrays(inst, CONFIG, geographic=True, manifest_dir=MANIFEST_DIR, log_path=log,
                                        label_mode="pickup_before_next_resolve", step_size=a.ss)
            by_window = _candidates_per_window(log)
            n = len(arr["node_ids"])
            cand = np.full((n, len(CAND_FEATURE_NAMES)), np.nan, dtype=np.float32)
            cand[:, 3] = 0.0  # cand_n_feasible_vehicles = 0 marks rows without a feasible vehicle
            for i in range(n):
                row = by_window.get(int(arr["window_index"][i]), {}).get(int(arr["node_ids"][i]))
                if row is not None:
                    cand[i] = row
            with_cand = int((cand[:, 3] > 0).sum())
            np.savez(out_dir / split / f"{inst}.npz", node_ids=arr["node_ids"], window_index=arr["window_index"], timestamp=arr["timestamp"],
                     label=arr["label"], features=arr["features"], cand_features=cand)
            manifest["instances"][inst] = {"split": split, "source_run": str(log.parent), "num_windows": arr["num_windows"], "num_rows": n,
                                           "positive_rate": float(arr["label"].mean()), "rows_with_candidates": with_cand, "rows_without_candidates": n - with_cand}
            print(f"{split} seed {seed}: {arr['num_windows']} windows, {n} rows, positive rate {arr['label'].mean():.3f}, {with_cand} rows with candidates ({100 * with_cand / n:.1f} %)", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    tot = sum(v["num_rows"] for v in manifest["instances"].values())
    print(f"DONE: {len(manifest['instances'])} instances, {tot} rows. Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
