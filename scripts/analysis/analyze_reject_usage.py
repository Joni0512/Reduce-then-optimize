"""
2026-10-14: how often does the SRL actor leave a vehicle without a trip (explicit reject action) compared to RHO? Re-runs the 5 config-D checkpoints
(bestD_s301-305, best-val checkpoint, mode "eval" = decision from the score ILP) and RHO (mode "offline" = decision from the cost ILP, keep_active=False,
cardinality 2, batch_interval=200/step_size=100) on the 9 validation instances, then parses the solver logs: per window the number of vehicles in operation
("N vehicle(s) in operations") and the vehicles that got a trip (distinct values of the executed result dict: "Score result" for SRL, "Default result" for RHO).
idle = vehicles - vehicles with a trip. idle_with_rejects = idle vehicles in windows where requests were rejected at the same time ("rejected: k" > 0).
A vehicle can also stay idle because no feasible trip exists - RHO's idle count is the reference for that. Output: outputs/reject_usage/ + printout.
Usage: SRL_MAX_THREAD_CNT=2 ./venv/bin/python3 scripts/analysis/analyze_reject_usage.py
"""
import json
import re
import statistics as st
from pathlib import Path

import torch

from rtv_solver.pipeline import srl_buffered_actor_lr_trial  # noqa: F401  (patches FeatureBuilder to 85 features)
from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline import select_feature_builder_class
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline.srl_train_val_test_split import VAL_INSTANCES
from rtv_solver.pipeline.srl_training_loop import REPO_ROOT, MANIFEST_DIR
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

OUT = REPO_ROOT / "outputs" / "reject_usage"
SW = REPO_ROOT / "outputs" / "srl_training_sweep"


def run(name, inst, model=None):
    d = OUT / name / inst
    log = d / "main.log"
    if log.exists() and "=== Iteration" in log.read_text(errors="ignore")[-200000:]:
        return log
    d.mkdir(parents=True, exist_ok=True)
    cleared = PayloadParser.clear_vehicle_manifests(PayloadParser.load_input_data(MANIFEST_DIR / f"{inst}.json"))
    cfg = Config(MAX_THREAD_CNT=2, OUTPUT_DIR=d, MODE="coaml", BATCH_INTERVAL=200, STEP_SIZE=100, SEED=42, MAX_CARDINALITY=2, KEEP_ACTIVE=False)
    setup_loggers(cfg.OUTPUT_DIR); set_seed(cfg.SEED, cfg.DEBUG)
    if model is None:
        COAMLPipeline(cfg, cleared, imitation_solution_path=MANIFEST_DIR / f"{inst}.json").solve_pdptw(cleared, mode="offline")
    else:
        COAMLPipeline(cfg, cleared, model=model, imitation_solution_path=MANIFEST_DIR / f"{inst}.json").solve_pdptw(cleared, mode="eval")
    return log


def parse(log, key):
    wins, cur = [], None
    for line in open(log, errors="ignore"):
        if "=== Iteration" in line:
            cur = {"veh": None, "res": None, "rej": 0}; wins.append(cur); continue
        if cur is None:
            continue
        m = re.search(r"(\d+) vehicle\(s\) in operations", line)
        if m: cur["veh"] = int(m.group(1))
        m = re.search(rf"{key} result: \{{(.*?)\}}, cost: [0-9.]+, rejected: (\d+)", line)
        if m:
            cur["res"] = {int(a): int(b) for a, b in (p.split(":") for p in m.group(1).split(",") if p.strip())}
            cur["rej"] = int(m.group(2))
    return [w for w in wins if w["veh"] is not None and w["res"] is not None]


def stats(wins):
    idle = sum(w["veh"] - len(set(w["res"].values())) for w in wins)
    idle_rej = sum((w["veh"] - len(set(w["res"].values()))) for w in wins if w["rej"] > 0)
    veh_windows = sum(w["veh"] for w in wins)
    return dict(windows=len(wins), veh_windows=veh_windows, idle=idle, idle_with_rejects=idle_rej)


results = {"RHO": {}}
for inst in VAL_INSTANCES:
    results["RHO"][inst] = stats(parse(run("rho", inst), "Default"))
    print("RHO", inst, results["RHO"][inst], flush=True)
for seed in (301, 302, 303, 304, 305):
    cfg0 = Config(OUTPUT_DIR=OUT, BATCH_INTERVAL=200, STEP_SIZE=100, SEED=seed, KEEP_ACTIVE=False)
    model = build_scoring_model("mlp", feature_dim=select_feature_builder_class(cfg0).FEATURE_SIZE, hidden_dim=64)
    model.load_state_dict(torch.load(SW / f"bestD_s{seed}" / "best_actor_checkpoint.pt", map_location="cpu")["model_state_dict"])
    results[f"SRL_s{seed}"] = {}
    for inst in VAL_INSTANCES:
        results[f"SRL_s{seed}"][inst] = stats(parse(run(f"srl_s{seed}", inst, model), "Score"))
    print(f"SRL s{seed} done", flush=True)
json.dump(results, open(OUT / "reject_usage.json", "w"), indent=1)

srl = [k for k in results if k.startswith("SRL")]
print(f"\n{'instance':8s} {'veh-windows':>11s} | {'RHO idle':>8s} {'RHO idle+rej':>12s} | {'SRL idle (mean)':>15s} {'SRL idle+rej (mean)':>20s}")
tot = {"vw": 0, "ri": 0, "rr": 0, "si": 0.0, "sr": 0.0}
for inst in VAL_INSTANCES:
    r = results["RHO"][inst]
    si = st.mean(results[k][inst]["idle"] for k in srl); sr = st.mean(results[k][inst]["idle_with_rejects"] for k in srl)
    vw = st.mean(results[k][inst]["veh_windows"] for k in srl)
    print(f"{inst:8s} {vw:11.0f} | {r['idle']:8d} {r['idle_with_rejects']:12d} | {si:15.1f} {sr:20.1f}")
    tot["vw"] += vw; tot["ri"] += r["idle"]; tot["rr"] += r["idle_with_rejects"]; tot["si"] += si; tot["sr"] += sr
print(f"{'sum':8s} {tot['vw']:11.0f} | {tot['ri']:8d} {tot['rr']:12d} | {tot['si']:15.1f} {tot['sr']:20.1f}")
print(f"idle share of vehicle-windows: RHO {tot['ri']/tot['vw']:.3f} (with rejects {tot['rr']/tot['vw']:.3f}), SRL {tot['si']/tot['vw']:.3f} (with rejects {tot['sr']/tot['vw']:.3f})")
