"""
2026-10-08: ONE table of ALL finished runs of the twin-critic / no-warmstart / pickup_slack / batch_interval=200/
step_size=100 SRL protocol - new buffered actor update (collect, then update, batch 16) and the old per-window
actor update - for direct comparison (see chat). Sources per row group:
- new method: scripts/analysis/make_buffered_actor_results_overview.py (result.json files + cluster logs pasted into chat)
- old, cm4 20-epoch sweep (8 runs): result.json values pasted into the chat on 2026-10-08 (cluster print-out)
- old, 30-epoch baselines: local runs from srl_train_val_curves.csv (seed 42 local = twin_critic_no_warmstart_pickupslack_1,
  seeds 47/48), val curve of the cluster seed-42 run (job 5635899, from its Slurm log pasted into the chat on 2026-10-08) and test values from the
  project notes (project_srl_minibatch_update_plan.md / project_srl_replicate_noise_finding.md)
- old, collapsed 30-epoch runs: srl_train_val_curves.csv (twin_critic_no_warmstart_pickupslack_criticlr001_1,
  ..._criticlr0015_pickupslack_1); their test rates (marked *) were RECOMPUTED on 2026-10-08 from the saved best_actor_checkpoint.pt
  (85 features, 9 test instances, batch_interval=200/step_size=100): 0.0124 and 0.0086; the original value was never stored (first run
  printed it only to the terminal, second run crashed in the test phase with the old feature-size bug)
2026-10-14 (table continued): further rows - config D repeats bestD_s301-305 (result.json), config E 20-epoch repeats bestE20_s201-205 (result.json), config E 30-epoch
runs bestE30_s201-204 (val curves/best values pasted from the cluster on 2026-10-12; test values not available; seed 205 still to add), the joint 10-epoch-buffer run
jointbuf_E_s201 (cluster result.json values pasted on 2026-10-14) and the local ablation runs abl_* (result.json, differing settings in the "Note" column).
2026-10-17: GNN actor config D seed 301 test value 0.5635 added (cluster result.json values pasted 2026-10-17: val 5/10/15/20 = 0.511/0.535/0.509/0.558);
abl_sigma0p1_s301_r2 (all val and test 0.0, collapsed) and abl_sigma0p3_s301_r2 (best val 0.6964 at epoch 15, test 0.7124) are cluster reruns already picked up via the abl_* glob.
n=1 per row; "-" = epoch not part of the run, "n/a" = value not available. PNG + PDF + tex in figures_export/SRL/.
"""
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures_export" / "SRL"
BASE_CLR = 0.0010833586558285635
new_runs = runpy.run_path(str(REPO / "scripts/analysis/make_buffered_actor_results_overview.py"))["RUNS"]


def old(where, clr, seed, epochs, best, best_ep, test, val):
    return {"method": "old (per window)", "where": where, "actor_lr": 0.0031923396291543868, "critic_lr": clr, "seed": seed, "epochs": epochs,
            "best_val": best, "best_epoch": best_ep, "test": test, "val": val}


ROWS = [dict(method="new (buffered)", **{k: v for k, v in r.items() if k not in ("label", "source")}) for r in new_runs]
ROWS += [
    # cm4 20-epoch sweep (cluster print-out of result.json files, 2026-10-08)
    old("cm4", 0.0006, 42, 20, 0.015, 15, 0.0188, {5: 0.0, 10: 0.0, 15: 0.015, 20: 0.0}),
    old("cm4", 0.0006, 43, 20, 0.6822, 10, 0.7397, {5: 0.603, 10: 0.682, 15: 0.635, 20: 0.603}),
    old("cm4", 0.0008, 42, 20, 0.6956, 15, 0.7402, {5: 0.484, 10: 0.562, 15: 0.696, 20: 0.614}),
    old("cm4", 0.0008, 43, 20, 0.6739, 15, 0.7183, {5: 0.631, 10: 0.631, 15: 0.674, 20: 0.629}),
    old("cm4", BASE_CLR, 42, 20, 0.5642, 20, 0.5775, {5: 0.506, 10: 0.520, 15: 0.424, 20: 0.564}),
    old("cm4", BASE_CLR, 43, 20, 0.6263, 10, 0.6646, {5: 0.592, 10: 0.626, 15: 0.601, 20: 0.603}),
    old("cm4", 0.0015, 42, 20, 0.365, 20, 0.2831, {5: 0.0, 10: 0.0, 15: 0.0, 20: 0.365}),
    old("cm4", 0.0015, 43, 20, 0.683, 20, 0.7071, {5: 0.570, 10: 0.644, 15: 0.678, 20: 0.683}),
    # 30-epoch baselines
    old("local", BASE_CLR, 42, 30, 0.7215, 25, 0.7702, {5: 0.608, 10: 0.606, 15: 0.592, 20: 0.704, 25: 0.722, 30: 0.594}),
    old("serial 5635899", BASE_CLR, 42, 30, 0.7329, 30, 0.7656, {5: 0.5917, 10: 0.5902, 15: 0.7154, 20: 0.6488, 25: 0.6442, 30: 0.7329}),
    old("local", BASE_CLR, 47, 30, 0.6809, 15, 0.7250, {5: 0.501, 10: 0.650, 15: 0.681, 20: 0.659, 25: 0.626, 30: 0.678}),
    old("local", BASE_CLR, 48, 30, 0.6892, 30, 0.6862, {5: 0.600, 10: 0.596, 15: 0.663, 20: 0.674, 25: 0.173, 30: 0.689}),
    # collapsed 30-epoch runs
    old("local", 0.001, 42, 30, 0.0086, 25, 0.0124, {5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0, 25: 0.009, 30: 0.004}),
    old("local", 0.0015, 42, 30, 0.0022, 30, 0.0086, {5: 0.0, 10: 0.0, 15: 0.002, 20: 0.0, 25: 0.0, 30: 0.002}),
]

# ---------------- 2026-10-14: continued rows ----------------
import glob as _g
import json as _j
SWEEP_DIR = REPO / "outputs" / "srl_training_sweep"


def _from_result(d, method, where, note):
    j = _j.load(open(d / "result.json"))
    curve = {pt["epoch"]: pt["service_rate"] for pt in j["val_curve"]}
    return dict(method=method, where=where, actor_lr=j["actor_lr"], critic_lr=j["critic_lr"], seed=j["seed"], epochs=j["epochs"], best_val=j["best_val_service_rate"],
                best_epoch=j["best_epoch"], test=j.get("test_service_rate"), val=curve, note=note)


for r in ROWS:
    r.setdefault("note", "")
for d in sorted(SWEEP_DIR.glob("bestD_s30*")):
    if (d / "result.json").exists() and "dead" not in d.name:
        ROWS.append(_from_result(d, "new (buffered)", "local", "config D"))
for d in sorted(SWEEP_DIR.glob("bestE20_s20*")):
    if (d / "result.json").exists() and "dead" not in d.name:
        ROWS.append(_from_result(d, "new (buffered)", "local", "config E"))
# config E, 30 epochs (cluster; values pasted on 2026-10-12, test not available, seed 205 pending)
for seed, vals, best, bep in ((201, [0.475, 0.519, 0.536, 0.549, 0.711, 0.642], 0.7111, 25), (202, [0.653, 0.696, 0.609, 0.633, 0.708, 0.643], 0.7083, 25),
                              (203, [0.605, 0.596, 0.602, 0.680, 0.685, 0.635], 0.6846, 25), (204, [0.603, 0.639, 0.696, 0.592, 0.669, 0.644], 0.6956, 15)):
    ROWS.append(dict(method="new (buffered)", where="serial", actor_lr=0.009315365931220558, critic_lr=0.0016399458556079478, seed=seed, epochs=30, best_val=best, best_epoch=bep,
                     test=None, val=dict(zip((5, 10, 15, 20, 25, 30), vals)), note="config E"))
# joint 10-epoch buffer (cluster result.json values pasted on 2026-10-14)
ROWS.append(dict(method="new (joint 10-ep buffer)", where="serial", actor_lr=0.009315365931220558, critic_lr=0.0016399458556079478, seed=201, epochs=20, best_val=0.6504, best_epoch=15,
                 test=0.6621, val={5: 0.538, 10: 0.579, 15: 0.650, 20: 0.594}, note="config E, joint"))
ROWS.append(dict(method="new (buffered, GNN actor)", where="serial", actor_lr=0.009315365931220558, critic_lr=0.0016399458556079478, seed=201, epochs=20, best_val=0.5277, best_epoch=5,
                 test=0.5338, val={5: 0.5277, 10: 0.5162, 15: 0.5105, 20: 0.4253}, note="config E, GNN actor (gcn, 2 layers)"))
ROWS.append(dict(method="new (buffered, GNN actor)", where="serial", actor_lr=0.0031923396291543868, critic_lr=0.002, seed=301, epochs=20, best_val=0.5583, best_epoch=20,
                 test=0.5635, val={5: 0.5107, 10: 0.5346, 15: 0.5094, 20: 0.5583}, note="config D, GNN actor (gcn, 2 layers); val values from the job log (2026-10-15), test from result.json (cluster values pasted 2026-10-17)"))
ROWS.append(dict(method="new (buffered, GNN actor)", where="serial", actor_lr=0.0031923396291543868, critic_lr=0.002, seed=302, epochs=20, best_val=0.5645, best_epoch=20,
                 test=0.6021, val={5: 0.5406, 10: 0.5432, 15: 0.5424, 20: 0.5645}, note="config D, GNN actor (gcn, 2 layers); job 5664373, values pasted 2026-10-18"))
ROWS.append(dict(method="new (buffered, GNN actor)", where="serial", actor_lr=0.0031923396291543868, critic_lr=0.002, seed=303, epochs=20, best_val=0.7457, best_epoch=15,
                 test=0.7463, val={5: 0.6853, 10: 0.7158, 15: 0.7457, 20: 0.737}, note="config D, GNN actor (gcn, 2 layers); job 5664374, values pasted 2026-10-17"))
ROWS.append(dict(method="new (joint 10-ep buffer)", where="serial", actor_lr=0.0031923396291543868, critic_lr=0.002, seed=301, epochs=20, best_val=0.642, best_epoch=15,
                 test=0.6626, val={5: 0.6112, 10: 0.605, 15: 0.642, 20: 0.618}, note="config D, joint"))
ROWS.append(dict(method="new (joint 10-ep buffer)", where="serial", actor_lr=0.0031923396291543868, critic_lr=0.002, seed=302, epochs=20, best_val=0.6942, best_epoch=5,
                 test=0.7183, val={5: 0.6942, 10: 0.6824, 15: 0.6224, 20: 0.6313}, note="config D, joint (values pasted 2026-10-17)"))
for seed_, nm, vals_, best_, bep_, test_ in ((302, "sigma=0.1", {5: 0.676, 10: 0.659, 15: 0.6999, 20: 0.6956}, 0.6999, 15, 0.7283),
                                              (302, "polyak_tau=0.001", {5: 0.2886, 10: 0.5056, 15: 0.5452, 20: 0.5496}, 0.5496, 20, 0.5328),
                                              (302, "gamma=0.97", {5: 0.4764, 10: 0.7219, 15: 0.6919, 20: 0.7153}, 0.7219, 10, 0.7568)):
    ROWS.append(dict(method="new (buffered)", where="serial", actor_lr=0.0031923396291543868, critic_lr=0.002, seed=seed_, epochs=20, best_val=best_, best_epoch=bep_,
                     test=test_, val=vals_, note="config D, " + nm))
# 2026-10-16: repeat of config D seed 301 with gradient-norm logging, and the actor-clip ablation (clip100_*)
for d in sorted(SWEEP_DIR.glob("gradlog_D_s*")):
    if (d / "result.json").exists():
        ROWS.append(_from_result(d, "new (buffered)", "local", "config D, repeat of seed 301 with grad-norm logging (clip 1.0 active in 99.8% of steps)"))
for d in sorted(SWEEP_DIR.glob("clip*_D_s*")):
    if (d / "result.json").exists():
        j = _j.load(open(d / "result.json"))
        ROWS.append(_from_result(d, "new (buffered)", "local", f"config D, actor_clip={j.get('actor_clip')}, critic_clip={j.get('critic_clip')}"))
# 2026-10-17: reward_mode=local_served (+1 in the window of the real pickup), config D seed 301, with critic-loss logging
for d in sorted(SWEEP_DIR.glob("served_D_s*")):
    if (d / "result.json").exists():
        ROWS.append(_from_result(d, "new (buffered)", "local", "config D, reward local_served (+1 at pickup), critic-loss logging"))
# 2026-10-18: episodic scheme (buffer pretrain 10 instances x 150 steps x batch 20; per episode 1 actor + 1 critic step, batch 20, shared buffer of the last 10 episodes), local_served
for d in sorted(list(SWEEP_DIR.glob("episodic_D_s*")) + list(SWEEP_DIR.glob("episodic_D_ep30_s*")) + list(SWEEP_DIR.glob("epigrid_a*_s*"))):  # 2026-10-20: + 30-epoch runs + learning-rate grid (lrs are in the actor_lr / critic_lr columns)
    if (d / "result.json").exists():
        ROWS.append(_from_result(d, "new (episodic)", "local" if d.name.endswith("301") else "serial", "config D, episodic (buffer pretrain 10 inst x150, 1 actor+1 critic step/episode, buffer 10 ep), local_served"))
# 2026-10-18: episodic scheme + two-step TD target (SRL_TD_STEPS=2), otherwise identical to episodic_D_s*
for d in sorted(SWEEP_DIR.glob("episodic_td2_D_*s30*")):  # also episodic_td2_D_clr0p004_s301 (critic_lr 0.004, see the critic_lr column)
    if (d / "result.json").exists():
        ROWS.append(_from_result(d, "new (episodic, 2-step TD)", "local" if d.name.endswith("301") else "cm4", "config D, episodic, TWO-STEP TD target, local_served"))
# 2026-10-19: finished cluster runs of the episodic scheme (values pasted by the user from the cluster result.json files, jobs 226701-226703 / 226709-226710)
_ALR_D = 0.0031923396291543868
for _seed, _bv, _be, _te, _val, _job in ((302, 0.687268150930748, 10, 0.711679246337034, {5: 0.6245, 10: 0.6873, 15: 0.6765, 20: 0.6373}, 5671417),
                                          (303, 0.7241508169144129, 15, 0.7528010592149204, {5: 0.6791, 10: 0.7132, 15: 0.7242, 20: 0.6095}, 226701),
                                          (304, 0.713130013818138, 10, 0.7057353529668242, {5: 0.6522, 10: 0.7131, 15: 0.7046, 20: 0.6851}, 226702),
                                          (305, 0.6743655197928228, 20, 0.6736445980495802, {5: 0.6033, 10: 0.5841, 15: 0.62, 20: 0.6744}, 226703)):
    ROWS.append(dict(method="new (episodic)", where=f"cm4 {_job}", actor_lr=_ALR_D, critic_lr=0.002, seed=_seed, epochs=20, best_val=_bv, best_epoch=_be, test=_te, val=_val,
                     note="config D, episodic (buffer pretrain 10 inst x150, 1 actor+1 critic step/episode, buffer 10 ep), local_served"))
for _seed, _bv, _be, _te, _val, _job in ((302, 0.7132573810043289, 10, 0.7441189540375631, {5: 0.6521, 10: 0.7133, 15: 0.6697, 20: 0.6786}, 226709),
                                          (303, 0.7069730056189545, 15, 0.728587426513188, {5: 0.683, 10: 0.6137, 15: 0.707, 20: 0.6612}, 226710)):
    ROWS.append(dict(method="new (episodic, 2-step TD)", where=f"cm4 {_job}", actor_lr=_ALR_D, critic_lr=0.002, seed=_seed, epochs=20, best_val=_bv, best_epoch=_be, test=_te, val=_val,
                     note="config D, episodic, TWO-STEP TD target, local_served"))
# 2026-10-19: OLD buffered main phase (local_positive) with the NEW buffer pretrain (10 instances x 150 steps x batch 20) - separates the pretrain from the episodic main phase; and the episodic scheme with the OLD pretrain
for d in sorted(SWEEP_DIR.glob("bufpre_D_s*")):
    if (d / "result.json").exists():
        ROWS.append(_from_result(d, "new (buffered, buffer pretrain)", "local" if d.name.endswith("301") else "cluster", "config D, buffered main phase, NEW buffer pretrain, local_positive"))
for d in sorted(SWEEP_DIR.glob("epi_oldpre_D_s*")):
    if (d / "result.json").exists():
        ROWS.append(_from_result(d, "new (episodic, old pretrain)", "local" if d.name.endswith("301") else "cluster", "config D, episodic main phase, OLD per-episode pretrain, local_served"))
# 2026-10-20: bufpre_D seeds 302-305 (cluster result.json values pasted by the user; jobs 227067-227069, 5673139): OLD buffered main phase + NEW buffer pretrain, local_positive
for _seed, _bv, _be, _te, _val, _job in ((302, 0.6960747317795042, 15, 0.6667019691461671, {5: 0.6348, 10: 0.6462, 15: 0.6961, 20: 0.6789}, 227067),
                                          (303, 0.7241539789597504, 10, 0.7611866207537683, {5: 0.7006, 10: 0.7242, 15: 0.642, 20: 0.7092}, 227068),
                                          (304, 0.7223525933514835, 5, 0.7290797604856059, {5: 0.7224, 10: 0.6831, 15: 0.6179, 20: 0.6377}, 227069),
                                          (305, 0.6758233491751804, 10, 0.7030114792785899, {5: 0.6675, 10: 0.6758, 15: 0.6399, 20: 0.5664}, 5673139)):
    ROWS.append(dict(method="new (buffered, buffer pretrain)", where=f"cluster {_job}", actor_lr=0.0031923396291543868, critic_lr=0.002, seed=_seed, epochs=20, best_val=_bv, best_epoch=_be, test=_te, val=_val,
                     note="config D, buffered main phase, NEW buffer pretrain, local_positive"))
# 2026-10-20: finished runs of the episodic learning-rate grid (cluster result.json values pasted by the user; unfinished cells are added when they have a result.json)
for _alr, _clr, _seed, _bv, _be, _te, _val in ((0.0016, 0.002, 301, 0.7043653300701025, 5, 0.7053516633588159, {5: 0.7044, 10: 0.7003, 15: 0.6554, 20: 0.6549}),
                                                (0.0016, 0.002, 302, 0.7175679444491876, 20, 0.7568599554713749, {5: 0.6094, 10: 0.6827, 15: 0.6915, 20: 0.7176}),
                                                (0.0016, 0.004, 301, 0.7047006965985879, 10, 0.7377410067038874, {5: 0.6741, 10: 0.7047, 15: 0.6573, 20: 0.605}),
                                                (0.0016, 0.004, 302, 0.6847878425680868, 20, 0.7230520694005711, {5: 0.6503, 10: 0.6053, 15: 0.6051, 20: 0.6848}),
                                                (0.0032, 0.004, 302, 0.7152231929701408, 15, 0.7202310928134088, {5: 0.6848, 10: 0.7001, 15: 0.7152, 20: 0.6482}),
                                                (0.0008, 0.002, 301, 0.6243172986014273, 20, 0.6604950498180242, {5: 0.5317, 10: 0.5636, 15: 0.6156, 20: 0.6243}),
                                                (0.0032, 0.004, 301, 0.7306801559520761, 20, 0.7508564751134735, {5: 0.6286, 10: 0.67, 15: 0.7111, 20: 0.7307}),
                                                (0.0008, 0.002, 302, 0.7112384150563951, 20, 0.7326930577997294, {5: 0.6739, 10: 0.7024, 15: 0.6462, 20: 0.7112}),
                                                (0.0008, 0.004, 301, 0.7044442863421777, 10, 0.703174412448059, {5: 0.6157, 10: 0.7044, 15: 0.7001, 20: 0.6568})):
    ROWS.append(dict(method="new (episodic)", where="cluster grid", actor_lr=_alr, critic_lr=_clr, seed=_seed, epochs=20, best_val=_bv, best_epoch=_be, test=_te, val=_val,
                     note="learning-rate grid cell (see actor_lr / critic_lr columns)"))
# 2026-10-20/21: cosine learning-rate annealing (actor + critics, final factor 0.1): new episodic buffer (episodic_cos_D_s*) and the OLD buffered method (buffered_cos_D_s*)
for d in sorted(SWEEP_DIR.glob("episodic_cos_D_s*")):
    if (d / "result.json").exists():
        ROWS.append(_from_result(d, "new (episodic, cosine lr)", "local", "config D, lr cosine annealing x0.1 (actor + critics)"))
for d in sorted(SWEEP_DIR.glob("buffered_cos_D_s*")):
    if (d / "result.json").exists():
        ROWS.append(_from_result(d, "new (buffered, cosine lr)", "local", "config D, lr cosine annealing x0.1 (actor + critics)"))
# 2026-10-21: 30-epoch episodic 2-step run, seed 302 (cluster job 227265; result.json values pasted by the user)
ROWS.append(dict(method="new (episodic, 2-step TD)", where="cluster 227265", actor_lr=0.0031923396291543868, critic_lr=0.002, seed=302, epochs=30, best_val=0.6806870492109115, best_epoch=10,
                 test=0.707893325941174, val={5: 0.605, 10: 0.6807, 15: 0.6527, 20: 0.6524, 25: 0.6543, 30: 0.6436}, note="30 epochs"))
for d in sorted(SWEEP_DIR.glob("abl_*_s*")):
    if (d / "result.json").exists():
        j = _j.load(open(d / "result.json"))
        diffs = [f"{k}={j[k]}" for k, dflt in (("sigma", 0.2), ("num_samples", 20), ("gamma", 0.99), ("polyak_tau", 0.005)) if j.get(k, dflt) != dflt]
        # 2026-10-17: the *_r2 ablation reruns (sigma 0.1 / 0.3, seed 301) ran on the cluster (jobs 5664738/5664739, serial), only their result.json was copied here
        ROWS.append(_from_result(d, "new (buffered)", "serial" if d.name.endswith("_r2") else "local", "config D, " + ", ".join(diffs)))

# 2026-10-17: finished wandb sweep trials (v1-v4, cluster coordinator + local agent) that are not yet among the rows above - read from wandb (summary + history of
# val_service_rate_curve per epoch + test), cached in outputs/wandb_sweep_rows.json so the script also works offline. Matching = same (actor_lr, critic_lr, seed).
_WB_CACHE = REPO / "outputs" / "wandb_sweep_rows.json"
_have = {(round(r["actor_lr"], 6), round(r["critic_lr"], 6), int(r["seed"])) for r in ROWS}
try:
    import wandb as _wandb
    _api = _wandb.Api()
    _wb_rows = []
    for _f in ("sweep_srl_buffered_id.txt", "sweep_srl_buffered_v2_id.txt", "sweep_srl_buffered_v3_id.txt", "sweep_srl_buffered_v4_id.txt"):
        for _r in _api.sweep((REPO / _f).read_text().strip()).runs:
            _c, _sm = _r.config, _r.summary
            if not _c.get("actor_lr") or _r.state != "finished" or not _sm.get("best_val_service_rate") or _sm.get("test_service_rate") is None:
                continue
            _key = (round(_c["actor_lr"], 6), round(_c["critic_lr"], 6), int(_c.get("seed", -1)))
            if _key in _have:
                continue
            _vals = {int(h["epoch"]): h["val_service_rate_curve"] for h in _r.history(pandas=False, samples=500) if h.get("val_service_rate_curve") is not None and h.get("epoch") is not None}
            _wb_rows.append(dict(method="new (buffered)", where=f"{_sm.get('where', 'cluster')} (sweep {_r.name.replace('-sweep', '')})", actor_lr=_c["actor_lr"], critic_lr=_c["critic_lr"], seed=int(_c["seed"]), epochs=20,
                                 best_val=_sm["best_val_service_rate"], best_epoch=_sm.get("best_epoch"), test=_sm["test_service_rate"], val={str(k): v for k, v in _vals.items()}, note="wandb sweep trial"))
    _WB_CACHE.write_text(_j.dumps(_wb_rows, indent=1))
except Exception as _e:  # offline: use the cache
    print("wandb not reachable, using cache:", type(_e).__name__)
    _wb_rows = _j.loads(_WB_CACHE.read_text()) if _WB_CACHE.exists() else []
for _row in _wb_rows:
    _row["val"] = {int(k): v for k, v in _row["val"].items()}
    if (round(_row["actor_lr"], 6), round(_row["critic_lr"], 6), int(_row["seed"])) not in _have:
        ROWS.append(_row)


# 2026-10-20 (user request: always label clearly what each run is): standardized label "scheme | pretrain | TD | reward" is put in front of every row's note
def _label(r):
    m, n = r["method"], r.get("note", "")
    if m.startswith("old"):
        return "old per-window actor update | old pretrain | 1-step TD | reward local_positive"
    scheme = "episodic (new buffer)" if "episodic" in m else ("joint 10-ep buffer" if "joint" in m else "buffered") + (" + GNN actor" if "GNN" in m else "")
    new_pre = ("episodic" in m and "old pretrain" not in m) or "buffer pretrain" in m
    td = "2-step TD" if "2-step" in m else "1-step TD"
    lr = " | lr cosine x0.1" if "cosine" in m else ""
    reward = "local_served" if ("episodic" in m or "local_served" in n) else "local_positive"
    return f"{scheme} | {'new buffer pretrain' if new_pre else 'old pretrain'} | {td} | reward {reward}{lr}"


for _r in ROWS:
    _r["note"] = _label(_r) + (" | " + _r["note"] if _r.get("note") else "")


def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


cols = ["Method", "Where", "actor_lr", "critic_lr", "Seed", "Ep.", "Best val (epoch)", "Test", "@5", "@10", "@15", "@20", "@25", "@30", "Note"]
table = []
for r in ROWS:
    cells = []
    recomputed = r["method"].startswith("old") and r["epochs"] == 30 and r["best_val"] < 0.01
    for e in (5, 10, 15, 20, 25, 30):
        if e > r["epochs"]:
            cells.append("-")
        else:
            cells.append(f(r["val"].get(e)))
    best = f"{r['best_val']:.4f}" + (f" ({r['best_epoch']})" if r["best_epoch"] else " (n/a)")
    table.append([r["method"], r["where"], f"{r['actor_lr']:.5g}", f"{r['critic_lr']:.5g}", str(r["seed"]), str(r["epochs"]), best, f(r["test"], 4) + ("*" if recomputed else ""), *cells, r.get("note", "")])

fig, ax = plt.subplots(figsize=(20, 1.6 + 0.3 * len(table)))
ax.axis("off")
tbl = ax.table(cellText=table, colLabels=cols, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.auto_set_column_width(col=list(range(len(cols))))
tbl.scale(1, 1.45)
for (i, j), cell in tbl.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if i == 0:
        cell.set_facecolor("#e8e8e8")
        cell.set_text_props(weight="bold")
    elif table[i - 1][0].startswith("new"):
        cell.set_facecolor("#eaf3fb")
ax.set_title("All finished runs, one row per run (n=1 per row); twin critic, no warmstart, pickup_slack, batch_interval=200, step_size=100; "
             "Note column starts with: scheme | pretrain | TD steps | reward; Ep. = number of epochs (blue = new methods; * = test rate recomputed from the saved checkpoint)", fontsize=9)
out = OUT / "all_finished_runs_table"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)

tex = ["\\begin{tabular}{llrrrrrrrrrrrr}", "\\hline", " & ".join(c.replace("_", "\\_") for c in cols[1:]) .join(["Method & ", " \\\\"]), "\\hline"]
tex = ["\\begin{tabular}{llrrrrlrrrrrrrl}", "\\hline", " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\hline"]
for row in table:
    tex.append(" & ".join(c.replace("_", "\\_") for c in row) + " \\\\")
tex += ["\\hline", "\\end{tabular}"]
(OUT / "all_finished_runs_table.tex").write_text("\n".join(tex) + "\n")
print(len(table), "rows written")
