"""
2026-11-23: diagnostic run for the NYC SRL collapse (user: "ja mach die" / "Ja, Skript + Lauf", see chat).
Finding so far: in outputs/srl_nyc/nyc_bi900_ss450_mc3_ep3_pre5_val1{,_ff}_s42 the actor assigns ~50-70 % of requests
before the FIRST buffered actor update phase (13 Adam steps) and 0 % after it, and never recovers (val 0/0/0 %).
The FY gradient per window is E[y_k] - y*, with y* = softmax(Q / SRL_TAU) mixture of the m perturbed actor solutions
(srl_target_action.py), Q = mean of the twin target critics. Hypothesis to test: the target critics systematically
prefer the candidates with MORE reject mass / fewer accepted trips -> y* has less accept mass than E[y_k] -> every
step pushes trip scores down until the ILP rejects everything. Critic + update buffer of the earlier runs were not
saved, so this re-runs the same setup and logs the target-action construction instead.

Setup = run_srl_nyc_bi900_ss450.py --feature_fixes --pretrain_epochs 5 --val_every 1 --seed 42 (the _ff run), but only
1 epoch (the collapse happened in epoch 1's update phase). Imported from that module so the NYC patches / split /
hyperparameters cannot drift. NO pipeline code is changed: only wrappers around coaml_pipeline's module-level
sample_candidate_assignments / score_candidates / softmax_target_action and around srl_training_loop._run_actor_update_phase.

Per call of _compute_srl_actor_loss (one buffered window, update phase) it logs, to <output_dir>/target_action_diag.jsonl:
  * phase / call index (batch = call // ACTOR_BATCH_SIZE), num_trips, num_reject
  * theta stats (trip part / reject part: mean, std, share > 0)
  * y0 = unperturbed actor solution (one EXTRA oracle solve with the candidate oracle, diagnostic only, does not enter
    the loss): accepted trips, requests picked up, reject mass
  * per candidate k: Q_k, accepted trips, requests picked up, reject mass
  * softmax weights, argmax candidate, y*: accept / reject mass, requests picked up (weighted)
Caveat: runs are not bit-reproducible (Gurobi + multiprocessing); the collapse reproduced in 3 of 4 earlier runs.
Usage: ./venv/bin/python3 -u -m rtv_solver.pipeline.diagnose_srl_nyc_target_action
"""
import json
import os
import statistics as st
import time

import torch

from rtv_solver.pipeline import run_srl_nyc_bi900_ss450 as nyc  # import first: applies the NYC module patches
from rtv_solver.pipeline import srl_training_loop as loop
from rtv_solver import coaml_pipeline as cp

trial = nyc.trial
SEED, EPOCHS, VAL_EVERY = 42, 1, 1
# 2026-11-23: env DIAG_PRETRAIN only for a quick wrapper check (0 = no critic pretraining); default 5 = the _ff run
PRETRAIN = int(os.environ.get("DIAG_PRETRAIN", "5"))
# 2026-11-23 (user: "Critic + Graphen sichern"): DIAG_SAVE_N > 0 additionally saves, in the FIRST actor update phase, the critics
# (live + target, whole modules via torch.save) at its first window and, for the first DIAG_SAVE_N windows, every candidate's
# critic input (request_features, vehicle_features, edge_index) + Q -> offline probes (selected_flag flip, init-seed sign test)
# in scripts/analysis/probe_srl_nyc_critic.py. Own output dir ("_save" tag) so the earlier diag run is not overwritten.
SAVE_N = int(os.environ.get("DIAG_SAVE_N", "0"))
# 2026-11-23 (user: "Ja, umsetzen" after the critic probe showed a random init sign): DIAG_BUFFER_PRETRAIN=1 replaces the 5
# per-episode pretrain epochs by the SF buffer pretrain (same kwargs as run_srl_sf_uniformroad_bi900_ss450.py --buffer_pretrain:
# all train instances, 3 collect passes, sigma 0.2, 500 steps, batch 20, then 1 per-episode epoch) -> does a better trained
# critic get a consistent sign of corr(Q, requests)? Run dir tag "bufpre" instead of "pre<N>".
BUFFER_PRETRAIN = os.environ.get("DIAG_BUFFER_PRETRAIN", "0") == "1"
BUFFER_KW = (dict(pretrain_mode="buffer", pretrain_instances=["all"], pretrain_collect_epochs=3, pretrain_exploration_sigma=0.2,
                  pretrain_steps=500, pretrain_batch=20, per_episode_after_buffer_epochs=1) if BUFFER_PRETRAIN else {})
_FB = trial._feat_builder_module.FeatureBuilder
_FB.FIX_CAPACITY_FEATURES = True  # = --feature_fixes of the _ff run
_FB.FIX_TIME_NORMALIZATION = True
_FB.FIX_FEATURE_SCALING = True

OUT = loop.REPO_ROOT / "outputs" / "srl_nyc" / f"diag_target_action_ep{EPOCHS}_{'bufpre' if BUFFER_PRETRAIN else f'pre{PRETRAIN}'}_val{VAL_EVERY}_ff{'_save' if SAVE_N else ''}_s{SEED}"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "target_action_diag.jsonl"
LOG.write_text("")

_ctx: dict = {"phase": 0, "call": 0}


def _split_stats(y: torch.Tensor, n_trips: int, n_rej: int, trip_costs) -> dict:
    y = y.detach().float()
    acc = y[:n_trips]
    req = sum(float(acc[i]) * len(trip_costs[i].get_ordered_request_ids()) for i in torch.nonzero(acc > 1e-6).flatten().tolist())
    return {"accept": float(acc.sum()), "reject": float(y[n_trips:n_trips + n_rej].sum()) if n_rej else 0.0, "requests": req}


def _theta_stats(t: torch.Tensor) -> dict:
    t = t.detach().float()
    if t.numel() == 0:
        return {}
    return {"mean": float(t.mean()), "std": float(t.std()) if t.numel() > 1 else 0.0, "pos_share": float((t > 0).float().mean())}


_orig_loss = cp.COAMLPipeline._compute_srl_actor_loss
_orig_sample = cp.sample_candidate_assignments
_orig_score = cp.score_candidates
_orig_softmax = cp.softmax_target_action
_orig_update = loop._run_actor_update_phase


def _loss_wrapper(self, theta, trip_costs, *args, reject_vehicle_ids=None, **kw):
    n_trips, n_rej = len(trip_costs), len(reject_vehicle_ids or [])
    _ctx.update(rec={"phase": _ctx["phase"], "call": _ctx["call"], "batch": _ctx["call"] // trial.ACTOR_BATCH_SIZE,
                     "num_trips": n_trips, "num_reject": n_rej,
                     "theta_trip": _theta_stats(theta[:n_trips]), "theta_reject": _theta_stats(theta[n_trips:n_trips + n_rej])},
                n_trips=n_trips, n_rej=n_rej, trip_costs=trip_costs)
    _ctx["call"] += 1
    if SAVE_N and _ctx["phase"] == 1 and _ctx["call"] == 1:  # 2026-11-23: critics as used for y* in this update phase
        torch.save({"critic": self.critic, "critic2": getattr(self, "critic2", None), "target_critic": self.target_critic,
                    "target_critic2": self.target_critic2}, OUT / "critics_update_phase1.pt")
    _ctx["save"] = SAVE_N and _ctx["phase"] == 1 and _ctx["call"] <= SAVE_N
    out = _orig_loss(self, theta, trip_costs, *args, reject_vehicle_ids=reject_vehicle_ids, **kw)
    rec = _ctx.pop("rec")
    rec["fy_loss"] = float(self.last_loss.item())
    with open(LOG, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return out


def _sample_wrapper(theta, oracle, num_samples, sigma):
    cands = _orig_sample(theta, oracle, num_samples=num_samples, sigma=sigma)
    y0 = oracle(theta.detach())  # diagnostic only: unperturbed actor decision, not used by the loss
    _ctx["rec"]["y0"] = _split_stats(y0, _ctx["n_trips"], _ctx["n_rej"], _ctx["trip_costs"])
    return cands


def _score_wrapper(candidates, requests, vehicles, trip_costs, active_requests, current_time, actor_feature_builder,
                   match_graph_builder, match_feature_builder, critic, *, critic2=None, use_route_clique=False):
    q = _orig_score(candidates, requests, vehicles, trip_costs, active_requests, current_time, actor_feature_builder,
                    match_graph_builder, match_feature_builder, critic, critic2=critic2, use_route_clique=use_route_clique)
    if _ctx.get("save"):  # 2026-11-23: rebuild the same critic inputs (deterministic builders) and store them
        items = []
        for y, qv in zip(candidates, q):
            g = match_graph_builder.build_from_candidate(requests, vehicles, trip_costs, y, use_route_clique=use_route_clique)
            rf, vf = match_feature_builder.build(requests, vehicles, active_requests, g, current_time, actor_feature_builder)
            items.append({"request_features": rf, "vehicle_features": vf, "edge_index": g.edge_index.cpu(), "q": float(qv.reshape(())),
                          "requests": _split_stats(y, _ctx["n_trips"], _ctx["n_rej"], trip_costs)["requests"]})
        torch.save({"call": _ctx["rec"]["call"], "current_time": current_time, "candidates": items},
                   OUT / f"graphs_call{_ctx['rec']['call']:03d}.pt")
    return q


def _softmax_wrapper(candidates, q_values, tau):
    w, target = _orig_softmax(candidates, q_values, tau)
    n, r, tc = _ctx["n_trips"], _ctx["n_rej"], _ctx["trip_costs"]
    rec = _ctx["rec"]
    rec["cands"] = [dict(q=float(q.reshape(())), **_split_stats(y, n, r, tc)) for q, y in zip(q_values, candidates)]
    rec["weights"] = [float(x) for x in w]
    rec["argmax"] = int(torch.argmax(w))
    rec["y_star"] = _split_stats(target, n, r, tc)
    mean_y = torch.stack([c.float() for c in candidates]).mean(0)
    rec["mean_cand"] = _split_stats(mean_y, n, r, tc)
    return w, target


def _update_wrapper(*a, **kw):
    _ctx["phase"] += 1
    _ctx["call"] = 0
    print(f"[diag] actor update phase {_ctx['phase']} starts", flush=True)
    return _orig_update(*a, **kw)


cp.COAMLPipeline._compute_srl_actor_loss = _loss_wrapper
cp.sample_candidate_assignments = _sample_wrapper
cp.softmax_target_action = _softmax_wrapper
cp.score_candidates = _score_wrapper  # 2026-11-23: pass-through unless DIAG_SAVE_N
loop._run_actor_update_phase = _update_wrapper


def main() -> None:
    print(f"=== diag {OUT.name}: train={nyc.TRAIN} val={nyc.VAL} num_samples={trial.NUM_SAMPLES} tau={cp.Config().SRL_TAU} "
          f"sigma={trial.SIGMA} feature_size={_FB.FEATURE_SIZE} ===", flush=True)
    t0 = time.time()
    result = loop.run_srl_training_loop(
        reward_mode=trial.REWARD_MODE, actor_lr=nyc.ACTOR_LR, critic_lr=trial.CRITIC_LR,
        output_dir=OUT, actor_checkpoint=trial.ACTOR_CHECKPOINT,
        gamma=trial.GAMMA, tau=trial.TAU, sigma=trial.SIGMA, num_samples=trial.NUM_SAMPLES,
        actor_clip_norm=trial.ACTOR_CLIP, critic_clip_norm=trial.CRITIC_CLIP,
        epochs=EPOCHS, val_every_n_epochs=VAL_EVERY, critic_pretrain_epochs=PRETRAIN,
        batch_interval=nyc.BATCH_INTERVAL, step_size=nyc.STEP_SIZE, seed=SEED, max_cardinality=nyc.MAX_CARDINALITY,
        use_twin_critic=True, use_actor_warmstart=False,
        actor_update_mode="buffered", actor_batch_size=trial.ACTOR_BATCH_SIZE,
        **BUFFER_KW,  # 2026-11-23: empty unless DIAG_BUFFER_PRETRAIN=1
    )
    wall = time.time() - t0
    (OUT / "result.json").write_text(json.dumps({
        "run_id": OUT.name, "best_val_service_rate": result.best_val_service_rate, "val_curve": result.val_curve,
        "overfit_curve": result.overfit_curve, "num_samples": trial.NUM_SAMPLES, "wall_s": round(wall, 1)}, indent=2, default=str))
    # short summary per batch
    recs = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    for b in sorted({r["batch"] for r in recs}):
        rb = [r for r in recs if r["batch"] == b]
        print(f"[diag] batch {b}: n={len(rb)} y0 req {st.mean(r['y0']['requests'] for r in rb):.1f}  "
              f"mean_cand req {st.mean(r['mean_cand']['requests'] for r in rb):.1f}  y* req {st.mean(r['y_star']['requests'] for r in rb):.1f}  "
              f"theta_trip mean {st.mean(r['theta_trip'].get('mean', 0) for r in rb):.3f}", flush=True)
    print(f"=== diag DONE: val {result.val_curve}, wall {wall:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
