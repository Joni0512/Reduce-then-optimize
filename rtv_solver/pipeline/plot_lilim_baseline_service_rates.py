"""
2026-10-07: bar charts of the RHO baseline (no pruner) service rate per instance and per
geography x class group on the 12 Li&Lim test instances.

Source: new_tests/cardinality2_compare/results/results_summary.csv (variant == "baseline").
Setting of those runs (see Config in new_tests/cardinality2_compare/rh_baseline_c2.log):
mode=offline (RHO), max_cardinality=2, batch_interval=400, step_size=100, seed=42, no pruner.

Output: figures_export/LiLim Dataset/ (PNG+PDF, CSV, LaTeX table).
Run: ./venv/bin/python3 -m rtv_solver.pipeline.plot_lilim_baseline_service_rates
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "new_tests/cardinality2_compare/results/results_summary.csv"
OUT = REPO / "figures_export/LiLim Dataset"
GEO_COLOR = {"lc": "#1C7293", "lr": "#F2A541", "lrc": "#132A4C"}
GROUPS = ["LC1", "LR1", "LRC1", "LC2", "LR2", "LRC2"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(SRC)
    b = d[d.variant == "baseline"].copy()
    b["geo"] = b.instance.str.extract(r"^(lrc|lc|lr)")[0]
    b["cls"] = b.instance.str.extract(r"^(?:lrc|lc|lr)(\d)")[0]
    b["group"] = b.geo.str.upper() + b.cls
    b["rate_pct"] = b.service_rate * 100
    geo_rank = {"lc": 0, "lr": 1, "lrc": 2}
    b = b.sort_values(by=["cls", "geo", "instance"], key=lambda s: s.map(geo_rank) if s.name == "geo" else s)
    b[["instance", "group", "serviced", "total_requests", "service_rate", "runtime_s"]].to_csv(
        OUT / "lilim_rho_baseline_service_rate_per_instance.csv", index=False)

    grp = b.groupby("group").agg(mean_pct=("rate_pct", "mean"), n=("rate_pct", "count"),
                                 min_pct=("rate_pct", "min"), max_pct=("rate_pct", "max")).reindex(GROUPS)
    grp.to_csv(OUT / "lilim_rho_baseline_service_rate_per_group.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.1, 1]})

    # per instance, grouped by class
    x, ticks, labels, sep = 0, [], [], None
    for cls in ["1", "2"]:
        sub = b[b.cls == cls]
        for _, r in sub.iterrows():
            ax1.bar(x, r.rate_pct, color=GEO_COLOR[r.geo], alpha=0.6 if cls == "1" else 0.95)
            ax1.text(x, r.rate_pct + 1, f"{r.rate_pct:.0f}", ha="center", va="bottom", fontsize=8)
            ticks.append(x)
            labels.append(r.instance)
            x += 1
        if cls == "1":
            sep = x - 0.5
            x += 0.6
    ax1.axvline(sep + 0.3, color="grey", lw=0.8, ls="--")
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("service rate (%)")
    ax1.set_title("Per instance (class 1 left, class 2 right)", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # per group: mean bar + individual instances as dots
    for i, g in enumerate(GROUPS):
        geo = g[:-1].lower()
        ax2.bar(i, grp.loc[g, "mean_pct"], color=GEO_COLOR[geo], alpha=0.6 if g.endswith("1") else 0.95)
        pts = b[b.group == g].rate_pct.values
        ax2.scatter([i] * len(pts), pts, color="black", s=14, zorder=3)
        ax2.text(i, grp.loc[g, "max_pct"] + 2, f"{grp.loc[g, 'mean_pct']:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(range(len(GROUPS)))
    ax2.set_xticklabels(GROUPS)
    ax2.set_ylim(0, 105)
    ax2.set_title("Per group (mean of 2 instances, dots = instances)", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Li&Lim: RHO baseline (no pruner) service rate - max_cardinality=2, batch_interval=400, "
                 "step_size=100, seed=42, 12 test instances", fontsize=10.5)
    fig.text(0.01, -0.02, "Source: new_tests/cardinality2_compare/results/results_summary.csv (variant=baseline); "
             "colours: LC teal, LR amber, LRC navy; light = class 1, dark = class 2", fontsize=8, color="#5B6B7C")
    fig.tight_layout()
    stem = "lilim_rho_baseline_service_rate_bi400_ss100"
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{RHO baseline (no pruner) service rate on the 12 Li\&Lim test instances "
             r"(max\_cardinality=2, batch\_interval=400, step\_size=100, seed 42).}",
             r"\label{tab:lilim_rho_baseline_service_rate}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"Group & \#Inst. & Mean service rate (\%) & Min--Max (\%)\\", r"\midrule"]
    for g in GROUPS:
        r = grp.loc[g]
        lines.append(f"{g} & {int(r.n)} & {r.mean_pct:.1f} & {r.min_pct:.1f}--{r.max_pct:.1f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "lilim_rho_baseline_service_rate_table.tex").write_text("\n".join(lines) + "\n")
    print(grp.round(1).to_string())


if __name__ == "__main__":
    main()
