"""
2026-11-18: SF pair-pruner RHO test results (user: "gibt es schon Ergebnisse beim Pair Pruner?").
Input: outputs/pairpruner_sf_results.csv - summary of the cluster runs, written by the inline python block given in chat
(reads outputs/new_tests/sf_lodes_nodepot/*/run_offline_*/*/final/results.json on the cluster). Runs =
submit_sf_lodes_rho_uniformroad_pairpruner_allwindows_3seeds.sbatch: RHO on SF test instance seed 42, 150 vehicles,
uniform road-node starts (start seeds 42-46), max_cardinality=2, time-rule filter off, no depot return, rtv_timeout 1800,
pair pruner L1 / pos_weight 5 / lr 0.003, training seeds 1-3, geographic=True, thresholds 0.5-0.9.
Baseline (BASE) = same setup without pruner, same start seeds.
Variant A (user 2026-11-17): mean over training seeds (and start seeds), min-max over the 3 training-seed means, no selection on test.
NOTE (2026-11-18): config column now keeps the "_rtv21600" suffix (export fixed), so bi3600_ss1800_rtv21600 is its own config.
Output: figures_export/SF/sf_pair_pruner_rho_results_bi{BI}_ss{SS}.{png,pdf} + sf_pair_pruner_rho_results_table.csv
"""
import collections, csv, statistics as st
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_export" / "SF"
rows = list(csv.DictReader(open(ROOT / "outputs" / "pairpruner_sf_results.csv")))
for r in rows:
    r["service_rate"] = float(r["service_rate"]); r["total_time_s"] = float(r["total_time_s"])
configs = sorted({r["config"] for r in rows if r["label"] != "BASE"}, key=lambda c: int(c[2:].split("_")[0]))
table = []
for c in configs:
    base = [r for r in rows if r["config"] == c and r["label"] == "BASE"]
    b_sr, b_t = st.mean(r["service_rate"] for r in base), st.mean(r["total_time_s"] for r in base)
    table.append({"config": c, "label": "none (RHO baseline)", "threshold": "-", "n_runs": len(base), "train_seeds": "-",
                  "sr_mean": b_sr, "sr_ts_min": "", "sr_ts_max": "", "delta_sr_pp": 0.0, "time_mean_s": b_t, "time_rel": 1.0})
    for lbl in ("overlap", "vehicle"):
        for thr in ("0.5", "0.6", "0.7", "0.8", "0.9"):
            sel = [r for r in rows if r["config"] == c and r["label"] == lbl and r["threshold"] == thr]
            if not sel: continue
            per_ts = collections.defaultdict(list)
            for r in sel: per_ts[r["train_seed"]].append(r["service_rate"])
            ts_means = [st.mean(v) for v in per_ts.values()]
            sr = st.mean(r["service_rate"] for r in sel); t = st.mean(r["total_time_s"] for r in sel)
            table.append({"config": c, "label": lbl, "threshold": thr, "n_runs": len(sel), "train_seeds": "/".join(sorted(per_ts)),
                          "sr_mean": sr, "sr_ts_min": min(ts_means), "sr_ts_max": max(ts_means), "delta_sr_pp": sr - b_sr,
                          "time_mean_s": t, "time_rel": t / b_t})
with open(OUT / "sf_pair_pruner_rho_results_table.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
for t in table:
    rng = f"{t['sr_ts_min']:.2f}-{t['sr_ts_max']:.2f}" if t["sr_ts_min"] != "" else "-"
    print(f"{t['config']:13s} {t['label']:20s} {t['threshold']:4s} n={t['n_runs']:3d} ts={t['train_seeds']:6s} SR {t['sr_mean']:6.2f} "
          f"[{rng:11s}] dSR {t['delta_sr_pp']:+6.2f}pp  time {t['time_mean_s']:7.0f}s ({t['time_rel']*100:5.1f}%)")
COL = {"overlap": "#2a6fdb", "vehicle": "#e07b39"}
for c in configs:
    tc = [t for t in table if t["config"] == c and t["label"] != "none (RHO baseline)"]
    base = next(t for t in table if t["config"] == c and t["label"] == "none (RHO baseline)")
    thrs = ["0.5", "0.6", "0.7", "0.8", "0.9"]; labels = [l for l in ("overlap", "vehicle") if any(t["label"] == l for t in tc)]
    bi, ss = c[2:].split("_ss")
    rtv = 21600 if ss.endswith("_rtv21600") else 1800  # 2026-11-19: title said "rtv_timeout 1800 s" also for the rtv21600 config
    ss = ss.replace("_rtv21600", "")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    wbar = 0.8 / len(labels)
    for j, l in enumerate(labels):
        for k, (key, a) in enumerate((("delta_sr_pp", ax[0]), ("time_rel", ax[1]))):
            xs, ys, lo, hi, ns = [], [], [], [], []
            for i, thr in enumerate(thrs):
                t = next((t for t in tc if t["label"] == l and t["threshold"] == thr), None)
                if not t: continue
                xs.append(i + (j - (len(labels) - 1) / 2) * wbar)
                ys.append(t[key] * (100 if key == "time_rel" else 1))
                if key == "delta_sr_pp":
                    lo.append(t[key] - (t["sr_ts_min"] - base["sr_mean"])); hi.append((t["sr_ts_max"] - base["sr_mean"]) - t[key])
                ns.append(t["n_runs"])
            bars = a.bar(xs, ys, wbar * 0.9, color=COL[l], label=f"label {l}", yerr=[lo, hi] if key == "delta_sr_pp" else None, capsize=3)
            for x, y, n in zip(xs, ys, ns):
                if n < 15: a.annotate(f"n={n}", (x, y), ha="center", va="bottom" if y >= 0 else "top", fontsize=7)
    ax[0].axhline(0, color="gray", lw=1); ax[1].axhline(100, color="gray", lw=1, ls="--")
    ax[0].set_ylabel("service rate − RHO baseline [pp]"); ax[1].set_ylabel("runtime [% of RHO baseline]")
    ax[0].set_title(f"Service rate (baseline {base['sr_mean']:.2f} %)"); ax[1].set_title(f"Runtime (baseline {base['time_mean_s']:.0f} s)")
    for a in ax:
        a.set_xticks(range(len(thrs))); a.set_xticklabels(thrs); a.set_xlabel("pruner threshold (keep pair iff score ≥ threshold)"); a.grid(axis="y", alpha=.3); a.legend(fontsize=8)
    fig.suptitle(f"SF pair pruner in RHO, test seed 42, batch_interval={bi}/step_size={ss}, max_cardinality=2, 150 vehicles, rtv_timeout {rtv} s\n"
                 f"pruner L1 / pos_weight 5 / lr 0.003, mean over training seeds 1-3 and start seeds 42-46 (error bar = min-max of training-seed means; n<15 = incomplete, full = 3 training seeds x 5 start seeds)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "pdf"): fig.savefig(OUT / f"sf_pair_pruner_rho_results_bi{bi}_ss{ss}{'_rtv21600' if rtv == 21600 else ''}.{ext}", dpi=150)
