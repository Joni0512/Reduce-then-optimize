"""
2026-11-19: add the imputation values (TRAIN medians of the first three cand_ columns) to the checkpoint of a Request Pruner that was trained with the vehicle-related candidate features
(sweep_request_pruner_cand.py, variant "plus"), so that RequestPruner can impute missing cand values at runtime exactly like the training sweep did. Writes a COPY "<name>_deploy.pt" next to the
original; the original checkpoint is not modified.
Run: ./venv/bin/python3 scripts/analysis/make_cand_pruner_deploy_checkpoint.py --bi 40 --ss 10
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--bi", default="40")
ap.add_argument("--ss", default="10")
a = ap.parse_args()
D = Path(f"dataset/request_pruning_bi{a.bi}_ss{a.ss}_c2cand")
cand = np.vstack([np.load(p)["cand_features"] for p in sorted((D / "train").glob("*.npz"))])
med = np.nanmedian(cand, axis=0)
src = glob.glob(f"outputs/request_pruner_mlp_bi{a.bi}_ss{a.ss}_c2cand_plus/*/*_best_val_f3.pt")[0]
ck = torch.load(src, map_location="cpu", weights_only=False)
assert any(n.startswith("cand_") for n in ck["feature_names"]), "not a cand_ checkpoint"
ck["cand_impute_values"] = [float(med[0]), float(med[1]), float(med[2])]
dst = src.replace("_best_val_f3.pt", "_best_val_f3_deploy.pt")
torch.save(ck, dst)
print("source:", src)
print("deploy:", dst)
print("feature_names:", ck["feature_names"])
print("cand_impute_values:", ck["cand_impute_values"])
