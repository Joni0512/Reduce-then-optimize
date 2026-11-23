"""
2026-11-23: chart + table of the NYC SRL target-action diagnosis (rtv_solver/pipeline/diagnose_srl_nyc_target_action.py).
Both runs: batch_interval=900/step_size=450, max_cardinality=3, 50 vehicles, train 2016-01-05/06/12, val 2016-01-13,
buffered actor update (batch 16 -> 13 Adam steps over 195 windows), num_samples 20, sigma 0.2, SRL_TAU 0.02,
actor_lr 0.0031923396291543868, critic_lr = srl_buffered_actor_lr_trial.CRITIC_LR, twin critic, feature fixes on, seed 42, 1 epoch.
Only difference: critic pretraining 0 vs 5 per-episode epochs vs buffer pretrain (added 2026-11-23).
Input: outputs/srl_nyc/diag_target_action_ep1_pre{0,5}_val1_ff_s42/target_action_diag.jsonl (+ result.json)
Output: figures_export/SRL/srl_nyc_target_action_diag.{png,pdf} + _table.csv
"""
import csv, json, statistics as st
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_export/SRL/srl_nyc_target_action_diag"
# 2026-11-23: third run added (buffer critic pretrain: 3 collect passes, 500 steps, batch 20, + 1 per-episode epoch, DIAG_BUFFER_PRETRAIN=1);
# keys = run dir tag between "ep1_" and "_val1"
RUNS = {"pre0": "critic pretraining 0 epochs", "pre5": "critic pretraining 5 per-episode epochs",
        "bufpre": "buffer critic pretraining (500 steps)"}


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b); sa = sum((x - ma) ** 2 for x in a) ** .5; sb = sum((y - mb) ** 2 for y in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa > 0 and sb > 0 else None


rows = []
for pre in RUNS:
    d = ROOT / f"outputs/srl_nyc/diag_target_action_ep1_{pre}_val1_ff{'_save' if pre == 'bufpre' else ''}_s42"  # 2026-11-23: bufpre run only exists as _save
    R = [json.loads(l) for l in open(d / "target_action_diag.jsonl") if l.strip()]
    val = json.load(open(d / "result.json"))["best_val_service_rate"]
    for b in sorted({r["batch"] for r in R}):
        rb = [r for r in R if r["batch"] == b]
        cr = [c for c in (corr([c["q"] for c in r["cands"]], [c["requests"] for c in r["cands"]]) for r in rb) if c is not None]
        rows.append({"critic_pretrain": pre, "val_service_rate_after_epoch1": round(val, 4), "update_step": b + 1, "windows": len(rb),
                     "theta_trip_mean": round(st.mean(r["theta_trip"]["mean"] for r in rb), 3),
                     "req_actor_unperturbed": round(st.mean(r["y0"]["requests"] for r in rb), 1),
                     "req_candidates_mean": round(st.mean(r["mean_cand"]["requests"] for r in rb), 1),
                     "req_target_ystar": round(st.mean(r["y_star"]["requests"] for r in rb), 1),
                     "q_span_median": round(st.median(max(c["q"] for c in r["cands"]) - min(c["q"] for c in r["cands"]) for r in rb), 4),
                     "max_softmax_weight_median": round(st.median(max(r["weights"]) for r in rb), 3),
                     "corr_q_requests_mean": round(st.mean(cr), 2) if cr else None})
with open(OUT.parent / (OUT.name + "_table.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

fig, ax = plt.subplots(2, len(RUNS), figsize=(6 * len(RUNS), 7), sharex=True)
for j, pre in enumerate(RUNS):
    rr = [r for r in rows if r["critic_pretrain"] == pre]; x = [r["update_step"] for r in rr]
    a = ax[0, j]
    a.plot(x, [r["req_actor_unperturbed"] for r in rr], "-o", ms=4, lw=2, color="#123f7a", label="actor (unperturbed θ)")
    a.plot(x, [r["req_candidates_mean"] for r in rr], "-o", ms=4, lw=2, color="#9ec5f8", label="mean of 20 perturbed candidates")
    a.plot(x, [r["req_target_ystar"] for r in rr], "-o", ms=4, lw=2, color="#d9822b", label="target action y*")
    a.set_title(f"{RUNS[pre]} (val after epoch 1: {100 * rr[0]['val_service_rate_after_epoch1']:.1f} %)", fontsize=10)
    a.set_ylabel("requests picked up per window")
    b = ax[1, j]
    b.plot(x, [r["corr_q_requests_mean"] if r["corr_q_requests_mean"] is not None else float("nan") for r in rr], "-o", ms=4, lw=2, color="#7a1f3d", label="corr(Q, requests) over candidates")
    b.plot(x, [r["max_softmax_weight_median"] for r in rr], "-o", ms=4, lw=2, color="#2a78d6", label="max softmax weight (1/20 = uniform)")
    b.axhline(0, color="#444", lw=1); b.axhline(0.05, color="#2a78d6", lw=1, ls=":")
    b.set_ylim(-1.05, 1.05); b.set_xlabel("actor update step in epoch 1 (16 windows each)")
    for s in (a, b):
        s.grid(axis="y", color="#e5e5e5"); s.set_axisbelow(True)
        for sp in ("top", "right"): s.spines[sp].set_visible(False)
ax[0, 0].legend(frameon=False, fontsize=8); ax[1, 0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("NYC SRL, first actor update phase: target action y* vs. actor (batch_interval=900/step_size=450, max_cardinality=3, 50 vehicles)\n"
             "num_samples 20, sigma 0.2, SRL_TAU 0.02, actor_lr 0.00319, buffered batch 16, twin critic, feature fixes, seed 42 - source: diagnose_srl_nyc_target_action.py",
             fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT.with_suffix(f".{ext}"), dpi=200)
