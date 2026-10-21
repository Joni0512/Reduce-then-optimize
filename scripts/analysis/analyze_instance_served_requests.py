"""
2026-10-13: which requests of validation instance lr202 does SRL (config with actor_lr 0.0031923396291543868 / critic_lr 0.002, seeds 301-304, best-val
checkpoints) serve that RHO (keep_active=False, batch_interval=200/step_size=100, cardinality 2) does not? (user question: why is SRL so much better on lr202,
0.645 vs 0.480.) Re-runs RHO once (lr202 RHO rate is identical across the 7 runs) and the 4 SRL checkpoints in eval mode on lr202, collects the set of
serviced booking ids per run (StatsParser serviced_requests = pickup AND dropoff completed, same definition as the service rate) and joins it with
the request data (time windows, load, direct travel time from the instance's travel_time_matrix). Output: outputs/lr202_analysis/served_requests.json + printout.
Usage: SRL_MAX_THREAD_CNT=2 ./venv/bin/python3 scripts/analysis/analyze_lr202_served_requests.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import torch

from rtv_solver.pipeline import srl_buffered_actor_lr_trial  # noqa: F401  (patches FeatureBuilder to 85 features)
from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.stats_parser import StatsParser
from rtv_solver.pipeline import select_feature_builder_class
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline.srl_training_loop import REPO_ROOT, MANIFEST_DIR
from rtv_solver.structure.config import Config
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

# 2026-10-13: instance id is now argv[1] (default lr202; used for lc202 too), output folder outputs/<instance>_analysis
INSTANCE = sys.argv[1] if len(sys.argv) > 1 else "lr202"
OUT = REPO_ROOT / "outputs" / f"{INSTANCE}_analysis"
OUT.mkdir(parents=True, exist_ok=True)
SW = REPO_ROOT / "outputs" / "srl_training_sweep"
input_path = MANIFEST_DIR / f"{INSTANCE}.json"


def served_ids(config, cleared, driver_runs):
    stats_payload = {
        PayloadKeys.DEPOT: cleared[PayloadKeys.DEPOT], PayloadKeys.REQUESTS: cleared[PayloadKeys.REQUESTS],
        PayloadKeys.DRIVERS: driver_runs, PayloadKeys.TIME_MATRIX: cleared.get(PayloadKeys.TIME_MATRIX, None),
    }
    _, stats, _ = StatsParser(config, payload=stats_payload).evaluate(stats_payload)
    return sorted(int(b) for b in stats.serviced_requests)


def run_one(name, model=None):
    cleared = PayloadParser.clear_vehicle_manifests(PayloadParser.load_input_data(input_path))
    out_dir = OUT / "_runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(MAX_THREAD_CNT=2, OUTPUT_DIR=out_dir, MODE="coaml", BATCH_INTERVAL=200, STEP_SIZE=100, SEED=42, MAX_CARDINALITY=2, KEEP_ACTIVE=False)
    setup_loggers(cfg.OUTPUT_DIR); set_seed(cfg.SEED, cfg.DEBUG)
    if model is None:
        runs = COAMLPipeline(cfg, cleared, imitation_solution_path=input_path).solve_pdptw(cleared, mode="offline")
    else:
        runs = COAMLPipeline(cfg, cleared, model=model, imitation_solution_path=input_path).solve_pdptw(cleared, mode="eval")
    return served_ids(cfg, cleared, runs), len(cleared[PayloadKeys.REQUESTS])


served = {}
served["RHO"], n_req = run_one("rho")
print("RHO served", len(served["RHO"]), "of", n_req, flush=True)
for seed in (301, 302, 303, 304):
    res = json.load(open(SW / f"bestD_s{seed}" / "result.json"))
    cfg0 = Config(OUTPUT_DIR=OUT, BATCH_INTERVAL=200, STEP_SIZE=100, SEED=seed, KEEP_ACTIVE=False)
    model = build_scoring_model("mlp", feature_dim=select_feature_builder_class(cfg0).FEATURE_SIZE, hidden_dim=64)
    model.load_state_dict(torch.load(SW / f"bestD_s{seed}" / "best_actor_checkpoint.pt", map_location="cpu")["model_state_dict"])
    served[f"SRL_s{seed}"], _ = run_one(f"srl_s{seed}", model)
    print(f"SRL s{seed} served {len(served[f'SRL_s{seed}'])} of {n_req} (best epoch {res['best_epoch']}, val per-instance {res['val_curve'][-1]['per_instance'][INSTANCE]:.3f} at last eval)", flush=True)

d = json.load(open(input_path))
tm = d["travel_time_matrix"]
reqs = {}
for r in d["requests"]:
    p, q = r["pickup_pt"]["node_id"], r["dropoff_pt"]["node_id"]
    direct = tm[p][q]
    reqs[r["booking_id"]] = dict(
        pickup_start=r["pickup_time_window_start"], pickup_end=r["pickup_time_window_end"],
        dropoff_start=r["dropoff_time_window_start"], dropoff_end=r["dropoff_time_window_end"],
        load=r["am"], direct=direct,
        # slack = how much later than the earliest possible arrival the dropoff may happen (tightness of the request)
        slack=r["dropoff_time_window_end"] - (r["pickup_time_window_start"] + r["pickup_service_time"] + direct),
        depot_to_pickup=tm[0][p],
    )
srl_names = [k for k in served if k.startswith("SRL")]
table = []
for b, f in sorted(reqs.items()):
    n_srl = sum(b in served[k] for k in srl_names)
    table.append(dict(booking_id=b, rho=b in served["RHO"], srl_seeds=n_srl, **f))
json.dump(dict(served=served, n_requests=n_req, table=table), open(OUT / "served_requests.json", "w"), indent=1)

groups = {
    "SRL (>=3 of 4 seeds) but NOT RHO": [t for t in table if t["srl_seeds"] >= 3 and not t["rho"]],
    "RHO but NOT SRL (<=1 of 4 seeds)": [t for t in table if t["rho"] and t["srl_seeds"] <= 1],
    "both": [t for t in table if t["rho"] and t["srl_seeds"] >= 3],
    "neither": [t for t in table if not t["rho"] and t["srl_seeds"] <= 1],
}
feats = ["pickup_start", "pickup_end", "dropoff_start", "dropoff_end", "load", "direct", "slack", "depot_to_pickup"]
print(f"\nrequests: {n_req}; RHO serves {len(served['RHO'])}; SRL serves " + ", ".join(f"{k}={len(served[k])}" for k in srl_names))
print(f"{'group':36s} n  " + " ".join(f"{f[:13]:>13s}" for f in feats))
for g, rows in groups.items():
    print(f"{g:36s} {len(rows):2d} " + " ".join(f"{(st.mean(r[f] for r in rows) if rows else float('nan')):13.1f}" for f in feats))
print("\nSRL-but-not-RHO requests:", [(t["booking_id"], t["srl_seeds"]) for t in groups["SRL (>=3 of 4 seeds) but NOT RHO"]])
print("RHO-but-not-SRL requests:", [(t["booking_id"], t["srl_seeds"]) for t in groups["RHO but NOT SRL (<=1 of 4 seeds)"]])
