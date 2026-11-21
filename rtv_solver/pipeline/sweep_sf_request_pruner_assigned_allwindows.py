"""
2026-11-20: SF request-pruner training with the "assigned" label, one pruner per window config (CLAUDE.md 2026-10-20:
always window-specific). Same procedure as sweep_sf_request_pruner_bi900_ss450.py (pickup label): standard Li&Lim sweep
sweep_request_pruner.run_sweep unchanged (MLP grid hidden 16/32/64 x layers 1/2/3 x dropout 0/0.1/0.2 x pos_weight
1/1.5/2/3 x lr 0.001/0.003/0.01/0.03, 5 seeds, ranked by mean val ROC-AUC, winner retrained with seed 42 and saved).
Datasets: dataset/request_pruning_sf_assigned_bi{BI}_ss{SS} (build_sf_request_pruner_dataset_assigned_allwindows.py;
train = instance seed 1, val = instance seed 2). No test split -> val passed as "test" too, test_* columns dropped.
Windows without a dataset are skipped. Optional CLI args: list of batch_interval values to run (default: all present).
Output: outputs/request_pruner_mlp_sf_assigned_bi{BI}_ss{SS}/ (sweep CSVs + <winner>/ checkpoint + metadata.json)
Run: ./venv/bin/python3 -m rtv_solver.pipeline.sweep_sf_request_pruner_assigned_allwindows [1800]
"""
import sys
from functools import partial
from pathlib import Path

import pandas as pd

from rtv_solver.pipeline import sweep_request_pruner as sweep
from rtv_solver.pipeline.train_request_pruner import load_split

WINDOWS = [(300, 150), (600, 300), (900, 450), (1800, 900), (3600, 1800)]


def _load_train_val_val(dataset_dir: Path, *_args, **_kwargs):
    train, val = load_split("train", dataset_dir), load_split("val", dataset_dir)
    print(f"{dataset_dir}: train={len(train['labels'])} rows, val={len(val['labels'])} rows (val passed as test, test_* dropped)")
    return train, val, val


if __name__ == "__main__":
    only = {int(a) for a in sys.argv[1:]}
    for bi, ss in WINDOWS:
        if only and bi not in only:
            continue
        ds = Path(f"dataset/request_pruning_sf_assigned_bi{bi}_ss{ss}")
        out = Path(f"outputs/request_pruner_mlp_sf_assigned_bi{bi}_ss{ss}")
        if not (ds / "manifest.json").exists():
            print(f"SKIP batch_interval={bi}/step_size={ss}: no dataset {ds}")
            continue
        sweep.load_all_splits = partial(_load_train_val_val, ds)
        sweep.run_sweep(dataset_dir=ds, output_dir=out)
        for name in ("sweep_detailed_5seed.csv", "sweep_summary_5seed.csv"):
            df = pd.read_csv(out / name)
            df.drop(columns=[c for c in df.columns if c.startswith("test_")]).to_csv(out / name, index=False)
