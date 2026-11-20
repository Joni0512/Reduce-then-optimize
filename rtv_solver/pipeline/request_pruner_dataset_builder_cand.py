"""
request_pruner_dataset_builder_cand.py
======================================
2026-11-19: Request Pruner dataset = the existing per-window features (request_pruner_dataset_builder.build_instance_arrays, UNCHANGED: same 15 columns, same label, same
train/val/test split "mixed_all") PLUS four vehicle-related features per (window, request) built from the "CandidateInsertions" records of the baseline log
(Config.LOG_CANDIDATE_INSERTIONS, co_base.CO._log_candidate_insertions). Reason: user wants to test whether the extra trip time / pickup time of a request (relative to the
vehicles) helps the Request Pruner as an ADDITIONAL feature; pruners with and without these columns are to be trained on the SAME data (only the feature differs).

New columns (stored in the extra array `cand_features`, names in manifest["cand_feature_names"]; per request = over all vehicles with a FEASIBLE single-request insertion in that window):
    cand_min_veh_to_pickup   min over vehicles of the travel time vehicle (current/next node) -> pickup of the request
    cand_min_added_cost      min over vehicles of added_cost (= new optimal route cost - cost of the vehicle's current route, i.e. the ILP's own trip cost; extra trip time)
    cand_min_cost_ratio      min over vehicles of added_cost / max(veh_to_pickup, 1.0)   (user's idea: extra trip time relative to the time it takes to reach the request; the clip avoids
                             division by ~0 for a vehicle that is already at the pickup; same clip as in analyze_candidate_insertions.py)
    cand_n_feasible_vehicles number of vehicles with a feasible single-request insertion
Requests without any feasible vehicle in that window (or windows without an ILP solve record) get NaN in the first three columns and 0 in cand_n_feasible_vehicles; the values are stored RAW
(NaN not imputed): imputation and normalisation are training-time decisions (statistics from the TRAIN split only), same principle as the base dataset.

Alignment (verified on lr109): in assignment_data.jsonl the "CandidateInsertions" record of a window comes BEFORE that window's "Status" record; window_index of the base builder = index among all
"Status" records, so the pending CandidateInsertions record is attached to the next Status record. Every candidate request id must be among the window's active requests (checked, assertion).

Input : outputs/eval_rh_no_learning_bi<BI>_ss<SS>_c2cand/<instance>/rh_no_learning/assignment_data.jsonl (run_baseline_data_c2_candidates.py; C=2, cardinality 2, no pruner)
Output: dataset/request_pruning_bi<BI>_ss<SS>_c2cand/{train,val,test}/<instance>.npz (node_ids, window_index, timestamp, label, features [n,15], cand_features [n,4]) + manifest.json
Run   : ./venv/bin/python3 -m rtv_solver.pipeline.request_pruner_dataset_builder_cand --bi 40 --ss 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rtv_solver.pipeline.request_pruner_dataset_builder import (
    FEATURE_NAMES,
    REQUEST_PRUNER_SPLIT,
    build_instance_arrays,
)
from rtv_solver.structure.config import Config

CAND_FEATURE_NAMES = ["cand_min_veh_to_pickup", "cand_min_added_cost", "cand_min_cost_ratio", "cand_n_feasible_vehicles"]
PICKUP_CLIP = 1.0  # see docstring (cand_min_cost_ratio)


def _candidates_per_window(log_path: Path) -> dict[int, dict[int, list[float]]]:
    """{window_index (= index among all 'Status' records): {request_id: [min_veh_to_pickup, min_added_cost, min_cost_ratio, n_feasible]}}"""
    by_window: dict[int, dict[int, list[float]]] = {}
    pending = None
    window_index = -1
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            msg = rec.get("message")
            if msg == "CandidateInsertions":
                pending = rec["extra"]
            elif msg == "Status":
                window_index += 1
                if pending is not None:
                    cols = pending["columns"]
                    ix = {c: i for i, c in enumerate(cols)}
                    per_req: dict[int, list[list[float]]] = {}
                    for c in pending["candidates"]:
                        per_req.setdefault(int(c[ix["request_id"]]), []).append(c)
                    status = rec["extra"]["status"]
                    active = {int(r) for r in status["assigned_requests"]} | {int(r) for r in status["unserved_requests"]}
                    assert set(per_req) <= active, f"{log_path} window {window_index}: candidate request ids not among the active requests"
                    out = {}
                    for rid, cs in per_req.items():
                        pick = np.array([np.nan if c[ix["veh_to_pickup"]] is None else c[ix["veh_to_pickup"]] for c in cs], dtype=float)
                        added = np.array([c[ix["added_cost"]] for c in cs], dtype=float)
                        ratio = added / np.maximum(pick, PICKUP_CLIP)
                        out[rid] = [np.nanmin(pick) if np.isfinite(pick).any() else np.nan, float(np.min(added)),
                                    np.nanmin(ratio) if np.isfinite(ratio).any() else np.nan, float(len(cs))]
                    by_window[window_index] = out
                pending = None
    return by_window


def build_dataset_cand(bi: str, ss: str, rh_baseline_dir: Path | None = None, output_dir: Path | None = None) -> None:
    rh_baseline_dir = rh_baseline_dir or Path(f"outputs/eval_rh_no_learning_bi{bi}_ss{ss}_c2cand")
    output_dir = output_dir or Path(f"dataset/request_pruning_bi{bi}_ss{ss}_c2cand")
    config = Config()
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rh_baseline_dir": str(rh_baseline_dir),
        "feature_names": FEATURE_NAMES,
        "cand_feature_names": CAND_FEATURE_NAMES,
        "label_definition": "1.0 if the request appears in assigned_requests for that window (got a vehicle THIS window), else 0.0 (same as the base builder)",
        "normalization": "NOT applied; cand_features contain NaN (no feasible vehicle / no ILP record): impute and normalise at train time with TRAIN-split statistics only",
        "split_source": "train_models_v2.MODEL_CONFIGS['mixed_all']",
        "setting": f"RHO offline, batch_interval={bi}, step_size={ss}, C=2 on board (unit-demand copy), max_cardinality 2, no pruner, seed 42",
        "instances": {},
    }
    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        for inst in REQUEST_PRUNER_SPLIT[split]:
            arrays = build_instance_arrays(inst, config, rh_baseline_dir=rh_baseline_dir)
            if arrays is None:
                continue
            by_window = _candidates_per_window(rh_baseline_dir / inst / "rh_no_learning" / "assignment_data.jsonl")
            n = len(arrays["node_ids"])
            cand = np.full((n, len(CAND_FEATURE_NAMES)), np.nan, dtype=np.float32)
            cand[:, 3] = 0.0
            for i in range(n):
                row = by_window.get(int(arrays["window_index"][i]), {}).get(int(arrays["node_ids"][i]))
                if row is not None:
                    cand[i] = row
            with_cand = int((cand[:, 3] > 0).sum())
            np.savez(output_dir / split / f"{inst}.npz", node_ids=arrays["node_ids"], window_index=arrays["window_index"], timestamp=arrays["timestamp"],
                     label=arrays["label"], features=arrays["features"], cand_features=cand)
            manifest["instances"][inst] = {"split": split, "num_windows": arrays["num_windows"], "num_rows": n, "positive_rate": float(arrays["label"].mean()),
                                           "rows_with_candidates": with_cand, "rows_without_candidates": n - with_cand}
            print(f"  {split}/{inst}: {arrays['num_windows']} windows, {n} rows, {with_cand} with candidates ({100 * with_cand / n:.1f} %)", flush=True)
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    tot = sum(v["num_rows"] for v in manifest["instances"].values())
    wc = sum(v["rows_with_candidates"] for v in manifest["instances"].values())
    print(f"DONE: {len(manifest['instances'])} instances, {tot} rows, {wc} with candidates ({100 * wc / tot:.1f} %). Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", default="40")
    ap.add_argument("--ss", default="10")
    a = ap.parse_args()
    build_dataset_cand(a.bi, a.ss)
