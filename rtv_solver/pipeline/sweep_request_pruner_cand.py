"""
sweep_request_pruner_cand.py
============================
2026-11-19: Request Pruner sweep WITH vs. WITHOUT the four vehicle-related candidate-insertion features, on the SAME data and with the SAME protocol (user wants to test whether the extra
trip time / pickup time of a request, relative to the vehicles, helps the Request Pruner as an additional feature).

Data   : dataset/request_pruning_bi<BI>_ss<SS>_c2cand/ (request_pruner_dataset_builder_cand.py; C=2 baseline logs, split "mixed_all": 36 train / 8 val / 12 test instances).
Variants (both trained on exactly the same rows, grid and seeds):
    base  13 deployable columns (= train_request_pruner.NUM_DEPLOYABLE_FEATURES, the idle-proxy columns are dropped exactly like in the existing pruner training)
    plus  the same 13 columns + cand_min_veh_to_pickup, cand_min_added_cost, cand_min_cost_ratio, cand_n_feasible_vehicles (17 columns)
Missing candidate values (request without a feasible vehicle / window without an ILP record): NaN in the first three cand columns are replaced by the MEDIAN of the TRAIN split's column
(statistics from train only, same principle as the normalisation); cand_n_feasible_vehicles = 0 marks these rows. The normalisation (train mean/std) happens inside train_one_config as before.

The sweep itself is sweep_request_pruner.run_sweep UNCHANGED (432 configs x 5 seeds: hidden 16/32/64, layers 1/2/3, dropout 0/0.1/0.2, pos_weight 1/1.5/2/3, lr 0.001/0.003/0.01/0.03; seeds 42,1,2,3,4;
ranking by mean val ROC-AUC, test only reported, winner retrained once with seed 42). To feed it the variant arrays, load_all_splits (name inside sweep_request_pruner) and the feature names stored
in the checkpoint (train_request_pruner.DEPLOYABLE_FEATURE_NAMES) are replaced for the duration of each variant - no existing file is modified.

Output: outputs/request_pruner_mlp_bi<BI>_ss<SS>_c2cand_{base,plus}/{sweep_detailed_5seed.csv, sweep_summary_5seed.csv, <winner>/...}
Run   : ./venv/bin/python3 -m rtv_solver.pipeline.sweep_request_pruner_cand --bi 40 --ss 10 [--variants base plus]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rtv_solver.pipeline import sweep_request_pruner as srp
from rtv_solver.pipeline import train_request_pruner as trp
from rtv_solver.pipeline.request_pruner_dataset_builder_cand import CAND_FEATURE_NAMES

BASE_NAMES = list(trp.DEPLOYABLE_FEATURE_NAMES)


def _load_split_raw(split: str, dataset_dir: Path) -> dict:
    feats, cand, labels, inst = [], [], [], []
    for p in sorted((dataset_dir / split).glob("*.npz")):
        z = np.load(p)
        feats.append(z["features"][:, : trp.NUM_DEPLOYABLE_FEATURES])
        cand.append(z["cand_features"])
        labels.append(z["label"])
        inst.append(np.full(len(z["label"]), p.stem))
    return {"features": np.concatenate(feats).astype(np.float32), "cand": np.concatenate(cand).astype(np.float32),
            "labels": np.concatenate(labels).astype(np.float32), "instances": np.concatenate(inst)}


def run_variant(variant: str, bi: str, ss: str) -> None:
    dataset_dir = Path(f"dataset/request_pruning_bi{bi}_ss{ss}_c2cand")
    output_dir = Path(f"outputs/request_pruner_mlp_bi{bi}_ss{ss}_c2cand_{variant}")
    raw = {s: _load_split_raw(s, dataset_dir) for s in ("train", "val", "test")}
    medians = np.nanmedian(raw["train"]["cand"], axis=0)  # TRAIN statistics only; the n_feasible column has no NaN
    print(f"[{variant}] train medians of the cand columns {dict(zip(CAND_FEATURE_NAMES, np.round(medians, 2).tolist()))}")

    def build(split: str) -> dict:
        r = raw[split]
        if variant == "base":
            x = r["features"]
        else:
            c = np.where(np.isnan(r["cand"]), medians[None, :], r["cand"])
            x = np.hstack([r["features"], c]).astype(np.float32)
        return {"features": x, "labels": r["labels"], "instances": r["instances"]}

    train, val, test = build("train"), build("val"), build("test")
    print(f"[{variant}] train={len(train['labels'])} val={len(val['labels'])} test={len(test['labels'])} rows, {train['features'].shape[1]} feature columns")

    old_loader, old_names = srp.load_all_splits, trp.DEPLOYABLE_FEATURE_NAMES
    srp.load_all_splits = lambda _d: (train, val, test)  # feed the variant arrays to the unchanged sweep
    trp.DEPLOYABLE_FEATURE_NAMES = BASE_NAMES if variant == "base" else BASE_NAMES + CAND_FEATURE_NAMES  # checkpoint/metadata feature names
    try:
        srp.run_sweep(dataset_dir=dataset_dir, output_dir=output_dir)
    finally:
        srp.load_all_splits, trp.DEPLOYABLE_FEATURE_NAMES = old_loader, old_names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", default="40")
    ap.add_argument("--ss", default="10")
    ap.add_argument("--variants", nargs="+", default=["base", "plus"], choices=["base", "plus"])
    a = ap.parse_args()
    for v in a.variants:
        run_variant(v, a.bi, a.ss)
