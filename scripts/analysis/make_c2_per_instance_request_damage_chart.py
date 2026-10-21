"""
2026-10-10: Where does the Request Pruner (t=0.5, expert-prefix-label) hurt? c2 mixedclass, per
validation instance: delta = mean rate(Request-only) - mean rate(Baseline), over the available seeds.
Standard error of the delta = sqrt(s_r^2/n_r + s_b^2/n_b) (unpaired); an instance is flagged as
"hurt" when delta < -2*SE. Seeds differ in the epoch whose checkpoint was best (per-seed best val
epoch), see the summarize scripts.

Inputs (written on the cluster by the summarize scripts, rsynced to outputs/):
  outputs/per_instance_mixedclass_prefixlabel.csv   (Request-only t=0.5)
  outputs/per_instance_mixedclass_baseline.csv      (Baseline)
  columns: config, seed, instance, serviced, total, rate

Outputs (figures_export/Request Pruner SIL/Cardinality 2/):
  c2_mixedclass_per_instance_request_damage.png/.pdf, ..._table.csv

Run: ./venv/bin/python3 scripts/analysis/make_c2_per_instance_request_damage_chart.py [req_csv base_csv out_dir]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REQ_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/per_instance_mixedclass_prefixlabel.csv")
BASE_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/per_instance_mixedclass_baseline.csv")
OUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("figures_export/Request Pruner SIL/Cardinality 2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# 2026-10-10: optional 4th arg = threshold tag (e.g. "0.8"): changes title + output names so the t=0.5 chart is not overwritten
THR = sys.argv[4] if len(sys.argv) > 4 else "0.5"
SUFFIX = "" if THR == "0.5" else f"_t{THR}"

CFG_LABEL = {
    "bi200_ss100": "batch_interval=200/step_size=100",
    "bi400_ss200": "batch_interval=400/step_size=200",
    "bi400_ss80": "batch_interval=400/step_size=80",
}
TYPE_COLOR = {"lc": "#276575", "lr": "#A6493A", "lrc": "#E8A33D"}


def inst_type(name: str) -> str:
    return "lrc" if name.startswith("lrc") else ("lr" if name.startswith("lr") else "lc")


def agg(df):
    g = df.groupby(["config", "instance"])["rate"].agg(["mean", "std", "count"]).reset_index()
    return g


def main():
    req_raw = pd.read_csv(REQ_CSV)
    base_raw = pd.read_csv(BASE_CSV)
    # 2026-10-10: compare on the SAME seeds - restrict the baseline to the seeds present in the request-only CSV (per config)
    keep = req_raw[["config", "seed"]].drop_duplicates()
    base_raw = base_raw.merge(keep, on=["config", "seed"])
    req = agg(req_raw)
    base = agg(base_raw)
    m = req.merge(base, on=["config", "instance"], suffixes=("_req", "_base"))
    m["delta"] = m["mean_req"] - m["mean_base"]
    se2 = (m["std_req"].fillna(0) ** 2 / m["count_req"]) + (m["std_base"].fillna(0) ** 2 / m["count_base"])
    m["se"] = np.sqrt(se2)
    m["hurt"] = m["delta"] < -2 * m["se"]
    m["type"] = m["instance"].map(inst_type)
    m.to_csv(OUT_DIR / f"c2_mixedclass_per_instance_request_damage{SUFFIX}_table.csv", index=False)

    cfgs = [c for c in CFG_LABEL if c in set(m["config"])]
    fig, axes = plt.subplots(1, len(cfgs), figsize=(5.6 * len(cfgs), 6.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, cfg in zip(axes, cfgs):
        d = m[m["config"] == cfg].sort_values("delta")
        y = np.arange(len(d))
        ax.barh(y, d["delta"] * 100, xerr=d["se"] * 100, color=[TYPE_COLOR[t] for t in d["type"]],
                edgecolor=["black" if h else "none" for h in d["hurt"]], linewidth=1.6,
                capsize=3, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{i}{'  *' if h else ''}" for i, h in zip(d["instance"], d["hurt"])], fontsize=9)
        ax.axvline(0, color="#222222", linewidth=1)
        ax.set_title(f"{CFG_LABEL[cfg]}\nseeds: Request n={int(d['count_req'].max())}, Baseline n={int(d['count_base'].max())}",
                     fontsize=10.5)
        ax.set_xlabel("Request-only minus Baseline (percentage points)", fontsize=9.5)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        ax.grid(axis="x", alpha=0.3, zorder=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_COLOR.values()]
    fig.legend(handles, [f"{k} instances" for k in TYPE_COLOR], loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"c2 mixedclass: per-instance effect of the Request Pruner (t={THR}) vs. Baseline", fontsize=12.5)
    fig.text(0.5, 0.0, "mean over seeds; error bars = standard error of the difference; black outline and * = delta < -2 SE; "
             "epochs=5, lr=0.0001, pickup_slack off", ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(OUT_DIR / f"c2_mixedclass_per_instance_request_damage{SUFFIX}.png", dpi=200)
    fig.savefig(OUT_DIR / f"c2_mixedclass_per_instance_request_damage{SUFFIX}.pdf")
    print(f"saved to {OUT_DIR}")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(m.sort_values(["config", "delta"])[["config", "instance", "mean_base", "mean_req", "delta", "se", "hurt"]].to_string(index=False))


if __name__ == "__main__":
    main()
