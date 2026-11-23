"""
2026-11-23: offline probes of the NYC SRL critic (user: "Critic + Graphen sichern", see chat). Question: WHY do the target
critics rank perturbed candidates by fewer accepted requests (corr(Q, requests) -0.65 -> -1.0 in the first actor update
phase, which collapses the actor to 0 %)? Hypothesis (not verified before this script): the critic is barely trained
(15 per-episode gradient steps in pretraining, Q mean 2.5 vs. target 7.6) and the sign is an initialization artifact that
pretraining only scales up.
Input (written by rtv_solver/pipeline/diagnose_srl_nyc_target_action.py with DIAG_SAVE_N=50):
  outputs/srl_nyc/diag_target_action_ep1_pre5_val1_ff_save_s42/critics_update_phase1.pt + graphs_call*.pt
Probes, each = mean over saved windows of corr(Q, requests picked up) across that window's candidates:
  P0 sanity: Q recomputed = mean(target_critic, target_critic2) must equal the logged Q
  P1 which input carries the ranking: (a) original, (b) selected_flag (request feature 14, index 13) set to 0 for all
     requests (only edges differ), (c) all edges removed (only selected_flag differs)
  P2 init-seed sign: N_SEEDS freshly initialized CriticGNNs (same architecture) on the same graphs -> share of seeds with
     negative corr; plus the pretrained live critics and target critics individually
Output: printed table + <run dir>/critic_probe_results.json
Usage: ./venv/bin/python3 scripts/analysis/probe_srl_nyc_critic.py
"""
import json
import os
import statistics as st
from pathlib import Path

import torch

from rtv_solver.pipeline.critic_gnn import CriticGNN, GATLayer
from rtv_solver.pipeline.candidate_scoring_gnn import GCNMeanLayer

ROOT = Path(__file__).resolve().parents[2]
# 2026-11-23: env PROBE_RUN (run dir name under outputs/srl_nyc) so the same probes run on the buffer-pretrain diag run;
# default = the original per-episode pre5 run, chart name gets a "_bufpre" suffix for the buffer run
RUN = ROOT / "outputs/srl_nyc" / os.environ.get("PROBE_RUN", "diag_target_action_ep1_pre5_val1_ff_save_s42")
BUFPRE = "_bufpre_" in RUN.name
N_SEEDS = 50
SEL_COL = 13  # selected_flag, match_graph_features.py _build_request_features (feature 14)


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    sa = sum((x - ma) ** 2 for x in a) ** .5; sb = sum((y - mb) ** 2 for y in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa > 1e-12 and sb > 1e-12 else None


windows = [torch.load(p, weights_only=False) for p in sorted(RUN.glob("graphs_call*.pt"))]
crit = torch.load(RUN / "critics_update_phase1.pt", weights_only=False)
print(f"{len(windows)} windows, candidates per window {len(windows[0]['candidates'])}")


@torch.no_grad()
def q_of(model, c, variant="orig"):
    rf, ei = c["request_features"], c["edge_index"]
    if variant == "no_flag":
        rf = rf.clone(); rf[:, SEL_COL] = 0.0
    elif variant == "no_edges":
        ei = torch.zeros((2, 0), dtype=ei.dtype)
    return float(model(rf, c["vehicle_features"], ei))


def mean_corr(score_fn, variant="orig"):
    cs = []
    for w in windows:
        qs = [score_fn(c, variant) for c in w["candidates"]]
        r = corr(qs, [c["requests"] for c in w["candidates"]])
        if r is not None:
            cs.append(r)
    return (st.mean(cs) if cs else None), len(cs)


tc, tc2 = crit["target_critic"].eval(), crit["target_critic2"].eval()
pair = lambda c, v: (q_of(tc, c, v) + q_of(tc2, c, v)) / 2  # = score_candidates (mean of twin target critics)

res = {}
# P0
err = max(abs(pair(c, "orig") - c["q"]) for w in windows for c in w["candidates"])
res["P0_max_abs_q_error"] = err
print(f"P0 max |Q recomputed - Q logged| = {err:.2e}")
# P1
for v in ("orig", "no_flag", "no_edges"):
    m, n = mean_corr(pair, v)
    res[f"P1_target_pair_{v}"] = m
    print(f"P1 target critics, {v:9s}: mean corr(Q, requests) {m:+.3f} (n={n} windows)")
# P2: trained critics individually
for name in ("critic", "critic2", "target_critic", "target_critic2"):
    mdl = crit.get(name)
    if mdl is None:
        continue
    mdl.eval()
    m, n = mean_corr(lambda c, v, mdl=mdl: q_of(mdl, c, v))
    res[f"P2_{name}"] = m
    print(f"P2 {name:15s}: mean corr {m:+.3f}")
# P2: fresh inits, same architecture
layer0 = tc.message_passing_layers[0]
agg = "gat" if isinstance(layer0, GATLayer) else ("gcn" if isinstance(layer0, GCNMeanLayer) else None)
assert agg is not None, type(layer0)
seed_corrs = []
for s in range(N_SEEDS):
    torch.manual_seed(s)
    fresh = CriticGNN(aggregator=agg, num_message_passing_layers=len(tc.message_passing_layers)).eval()
    m, _ = mean_corr(lambda c, v, f=fresh: q_of(f, c, v))
    seed_corrs.append(m)
neg = sum(1 for m in seed_corrs if m is not None and m < 0)
res["P2_fresh_init"] = {"aggregator": agg, "n_seeds": N_SEEDS, "share_negative": neg / N_SEEDS,
                        "mean": st.mean(m for m in seed_corrs if m is not None), "per_seed": seed_corrs}
print(f"P2 fresh inits ({agg}, {N_SEEDS} seeds): negative corr in {neg}/{N_SEEDS}, mean {res['P2_fresh_init']['mean']:+.3f}, "
      f"min {min(seed_corrs):+.3f}, max {max(seed_corrs):+.3f}")
(RUN / "critic_probe_results.json").write_text(json.dumps(res, indent=1))

# 2026-11-23: chart (user rule: results as chart + table with exact hyperparameters and source)
import matplotlib.pyplot as plt  # noqa: E402
OUT = ROOT / f"figures_export/SRL/srl_nyc_critic_probe{'_bufpre' if BUFPRE else ''}"
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
ax[0].hist([m for m in seed_corrs if m is not None], bins=20, range=(-1, 1), color="#9ec5f8", edgecolor="white", linewidth=1.5)
ax[0].axvline(0, color="#444", lw=1)
ax[0].set_xlabel("mean corr(Q, requests picked up) over the 20 candidates of a window")
ax[0].set_ylabel("freshly initialized critics (seeds 0-49)")
ax[0].set_title(f"untrained {agg.upper()} critics: negative in {neg}/{N_SEEDS}", fontsize=10)
names = ["critic", "critic2", "target_critic", "target_critic2"]
vals = [res.get(f"P2_{n}") for n in names]
ax[1].barh(names, vals, color=["#2a78d6" if v >= 0 else "#d9822b" for v in vals], height=0.6)
for i, v in enumerate(vals):
    ax[1].text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax[1].axvline(0, color="#444", lw=1); ax[1].set_xlim(-1, 1); ax[1].invert_yaxis()
ax[1].set_title("trained critics at the start of update phase 1\n(targets = y* scoring, mean of both)", fontsize=10)
for a in ax:
    a.grid(axis="x", color="#e5e5e5"); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.suptitle("NYC SRL critic probe, 50 windows of the first actor update phase (batch_interval=900/step_size=450, max_cardinality=3, 50 vehicles)\n"
             f"critic pretraining {'buffer (3 collect passes, 500 steps, batch 20) + 1 per-episode epoch' if BUFPRE else '5 epochs (15 per-episode steps)'}, num_samples 20, sigma 0.2, SRL_TAU 0.02, Polyak tau 0.005, seed 42 - "
             f"source: probe_srl_nyc_critic.py on {RUN.name}", fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT.with_suffix(f".{ext}"), dpi=200)
with open(OUT.parent / (OUT.name + "_table.csv"), "w") as fh:
    fh.write("critic,mean_corr_q_requests\n")
    for n, v in zip(names, vals):
        fh.write(f"{n},{v:.3f}\n")
    for s, m in enumerate(seed_corrs):
        fh.write(f"fresh_init_seed{s},{m:.3f}\n")
