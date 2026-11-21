"""
2026-11-19: SF request-pruner training, batch_interval=900/step_size=450, label "pickup_before_next_resolve" (user:
"Ja, trainieren"). Runs the standard Li&Lim request-pruner sweep unchanged (sweep_request_pruner.run_sweep: MLP grid
hidden 16/32/64 x layers 1/2/3 x dropout 0/0.1/0.2 x pos_weight 1/1.5/2/3 x lr 0.001/0.003/0.01/0.03, 5 seeds, ranked
by mean val ROC-AUC, winner retrained with seed 42 and saved) on dataset/request_pruning_sf_pickup_bi900_ss450
(train = instance seed 1, val = instance seed 2; built by build_sf_request_pruner_dataset_bi900_ss450.py).
Only difference: that dataset has NO test split (test instance seed 42 is kept untouched), but train_one_config needs one
-> the val split is passed as "test" too, and all test_* columns are dropped from the written CSVs afterwards so
nobody reads them as test results.
Output: outputs/request_pruner_mlp_sf_pickup_bi900_ss450/ (sweep_detailed_5seed.csv, sweep_summary_5seed.csv,
<winner>/ checkpoint + metadata.json)
Run: ./venv/bin/python3 -m rtv_solver.pipeline.sweep_sf_request_pruner_bi900_ss450
"""
from pathlib import Path

import pandas as pd

from rtv_solver.pipeline import sweep_request_pruner as sweep
from rtv_solver.pipeline.train_request_pruner import load_split

DATASET = Path("dataset/request_pruning_sf_pickup_bi900_ss450")
OUT = Path("outputs/request_pruner_mlp_sf_pickup_bi900_ss450")


def _load_train_val_val(dataset_dir: Path = DATASET):
    train, val = load_split("train", dataset_dir), load_split("val", dataset_dir)
    print(f"train={len(train['labels'])} rows, val={len(val['labels'])} rows (no test split; val passed as test, test_* dropped)")
    return train, val, val


if __name__ == "__main__":
    sweep.load_all_splits = _load_train_val_val
    sweep.run_sweep(dataset_dir=DATASET, output_dir=OUT)
    for name in ("sweep_detailed_5seed.csv", "sweep_summary_5seed.csv"):
        df = pd.read_csv(OUT / name)
        df.drop(columns=[c for c in df.columns if c.startswith("test_")]).to_csv(OUT / name, index=False)
