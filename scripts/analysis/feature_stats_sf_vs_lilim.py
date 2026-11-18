"""
2026-11-17: actor feature check on SF before planning SRL there (user: "features pruefen"). Read-only analysis, no solver
code changed: FeatureBuilder.build_matrix_from_trip_handler is wrapped at runtime to record every feature matrix of one
COAML eval episode with an UNTRAINED (randomly initialised) actor, then per-feature statistics are compared between
  * SF:     sf_lodes1000_0608_seed1_v150_cap2_vstart_uniformroad_s42_road_solve (train instance of the planned SRL),
            batch_interval=900/step_size=450, max_cardinality=2, time-rule filter off, no depot return, rtv_timeout 1800
            (= the current SF RHO setup)
  * Li&Lim: SRL training instances lc101, lr101, lrc101 at batch_interval=200/step_size=100 (SRL default), Config defaults
Caveat: a random actor visits other states than a trained one (e.g. fuller/emptier vehicles); this checks scales, NaN/inf,
constant columns and SF-vs-Li&Lim range shifts, not the on-policy distribution.
Output: outputs/feature_check/feature_stats_{sf,lilim}.csv + feature_compare.csv (per feature, side by side)
Usage: ./venv/bin/python3 -m scripts.analysis.feature_stats_sf_vs_lilim [--only sf|lilim]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline import feat_builder
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "feature_check"
SF = ("sf", ROOT / "solutions/sf/manifests/sf_lodes1000_0608_seed1_v150_cap2_vstart_uniformroad_s42_road_solve.json",
      dict(BATCH_INTERVAL=900, STEP_SIZE=450, MAX_CARDINALITY=2, USE_REQUEST_TIME_RULE_FILTER=False,
           RETURN_DEPOT=False, RTV_TIMEOUT=1800))
LILIM = [("lilim", ROOT / f"solutions/li_lim/manifests/{n}.json", dict(BATCH_INTERVAL=200, STEP_SIZE=100))
         for n in ("lc101", "lr101", "lrc101")]
ROWS_PER_STEP = 3000  # subsample per step to bound memory

_RECORD = []
_orig = feat_builder.FeatureBuilder.build_matrix_from_trip_handler


def _wrapped(self, trip_handler, current_time):
    m, names = _orig(self, trip_handler, current_time)
    if len(m):
        idx = np.random.default_rng(0).choice(len(m), size=min(len(m), ROWS_PER_STEP), replace=False)
        _RECORD.append((np.asarray(m, dtype=np.float64)[idx], list(names), len(m)))
    return m, names


feat_builder.FeatureBuilder.build_matrix_from_trip_handler = _wrapped


def run(tag, path, cfg):
    out = OUT / "episodes" / f"{tag}_{path.stem}"
    out.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=out, MODE="coaml", SEED=42, **cfg)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(42, config.DEBUG)
    payload = PayloadParser.clear_vehicle_manifests(PayloadParser.load_input_data(path))
    pipe = COAMLPipeline(config, payload, imitation_solution_path=path)
    t0 = time.time()
    pipe.solve_pdptw(payload, mode="eval", train_critic=False, reward_mode="local")
    print(f"{tag} {path.stem}: {len(_RECORD)} steps recorded so far, {time.time() - t0:.0f}s", flush=True)


def stats(tag):
    names = _RECORD[0][1]
    assert all(r[1] == names for r in _RECORD), "feature order differs between steps"
    X = np.concatenate([r[0] for r in _RECORD])
    rows = []
    for j, n in enumerate(names):
        c = X[:, j]
        fin = c[np.isfinite(c)]
        q = np.percentile(fin, [1, 50, 99]) if len(fin) else [np.nan] * 3
        rows.append({"feature": n, "n": len(c), "nan": int(np.isnan(c).sum()), "inf": int(np.isinf(c).sum()),
                     "min": fin.min() if len(fin) else np.nan, "p01": q[0], "median": q[1], "p99": q[2],
                     "max": fin.max() if len(fin) else np.nan, "mean": fin.mean() if len(fin) else np.nan,
                     "std": fin.std() if len(fin) else np.nan, "n_unique": len(np.unique(fin))})
    with open(OUT / f"feature_stats_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{tag}: {len(_RECORD)} steps, {sum(r[2] for r in _RECORD)} candidate trips in total, {X.shape[0]} rows kept, "
          f"{len(names)} features", flush=True)
    return {r["feature"]: r for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["sf", "lilim"], default=None)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for tag, jobs in (("lilim", LILIM), ("sf", [SF])):
        if a.only and a.only != tag:
            continue
        _RECORD.clear()
        for t, path, cfg in jobs:
            run(t, path, cfg)
        res[tag] = stats(tag)
    if len(res) == 2:
        keys = ["min", "p01", "median", "p99", "max", "std", "nan", "inf", "n_unique"]
        with open(OUT / "feature_compare.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["feature"] + [f"lilim_{k}" for k in keys] + [f"sf_{k}" for k in keys])
            for f in res["lilim"]:
                w.writerow([f] + [res["lilim"][f][k] for k in keys] + [res["sf"].get(f, {}).get(k) for k in keys])


if __name__ == "__main__":
    main()
