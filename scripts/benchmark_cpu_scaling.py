"""
2026-10-10: does requesting more CPUs speed up the buffered SRL trial? (see chat) Short fixed-work benchmark instead of a 24 h trial:
for each instance, (a) one full rollout in SRL collect mode (trip generation incl. the multiprocessing pool, scoring, step ILP, critic graph
features, window snapshots) and (b) the actor-loss computation for the first N stored windows (each ~40 MAP-ILP solves = the update-phase cost).
Prints wall-clock seconds plus the number of CPUs the process may actually use. Same seed/model/payload for every call, so run it under
different `sbatch --cpus-per-task=K` allocations and compare. --threads sets Config.MAX_THREAD_CNT (multiprocessing pool size; default 16 =
the value every real run uses).
Usage: ./venv/bin/python3 scripts/benchmark_cpu_scaling.py [--threads 16] [--windows 5] lr107 lrc205
"""
import argparse
import copy
import os
import tempfile
import time
from pathlib import Path

import torch

from rtv_solver.pipeline import srl_buffered_actor_lr_trial  # noqa: F401  (patches pickup_slack -> 85 features)
from rtv_solver.pipeline import feat_builder as fb
from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.actor_transition_buffer import ActorTransitionBuffer
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.replay_buffer import ReplayBuffer
from rtv_solver.pipeline.srl_training_loop import MANIFEST_DIR
from rtv_solver.structure.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("instances", nargs="+")
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--windows", type=int, default=5)
a = ap.parse_args()
try:
    cpus = len(os.sched_getaffinity(0))
except AttributeError:
    cpus = os.cpu_count()
print(f"allowed CPUs={cpus} (os.cpu_count={os.cpu_count()}), MAX_THREAD_CNT={a.threads}", flush=True)
total_roll = total_loss = 0.0
for inst in a.instances:
    cfg = Config(OUTPUT_DIR=Path(tempfile.mkdtemp()), MODE="coaml", BATCH_INTERVAL=200, STEP_SIZE=100, SEED=42, MAX_CARDINALITY=2, KEEP_ACTIVE=False,
                 MAX_THREAD_CNT=a.threads)
    path = MANIFEST_DIR / f"{inst}.json"
    payload = PayloadParser.clear_vehicle_manifests(PayloadParser.load_input_data(path))
    torch.manual_seed(0)
    model = build_scoring_model("mlp", feature_dim=fb.FeatureBuilder.FEATURE_SIZE, hidden_dim=64)
    critic = CriticGNN(aggregator="gat")
    buf = ActorTransitionBuffer()
    p = COAMLPipeline(cfg, payload, imitation_solution_path=path, model=model, critic=critic, critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
                      target_critic=copy.deepcopy(critic), critic_target_mode="td_bootstrap", replay_buffer=ReplayBuffer(capacity=40), actor_transition_buffer=buf)
    t0 = time.perf_counter()
    p.solve_pdptw(payload, mode="srl", optimizer=None, train_critic=False)
    t_roll = time.perf_counter() - t0
    n = min(a.windows, len(buf))
    t0 = time.perf_counter()
    for tr in buf._items[:n]:
        torch.manual_seed(1)
        p.compute_srl_actor_loss_for_transition(tr).backward()
    t_loss = time.perf_counter() - t0
    total_roll += t_roll; total_loss += t_loss
    print(f"{inst}: rollout {t_roll:.1f}s for {len(buf)} windows | actor loss for {n} windows {t_loss:.1f}s ({t_loss / max(n, 1):.2f}s each)", flush=True)
print(f"TOTAL rollout {total_roll:.1f}s | loss {total_loss:.1f}s | CPUs {cpus} threads {a.threads}", flush=True)
