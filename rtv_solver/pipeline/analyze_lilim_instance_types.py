"""
2026-10-07: descriptive statistics of the 56 Li&Lim instances, grouped by geography
(lc / lr / lrc) x class (1 / 2) -> 6 groups (LC1, LC2, LR1, LR2, LRC1, LRC2).

Source data: solutions/li_lim/manifests/*.json (requests, depot, reference driver_runs) and
solutions/li_lim/txt_files/*.txt are the published solutions (route listings only, no instance header).

Outputs go to figures_export/LiLim Dataset/ (CSV per instance, CSV per group, LaTeX table,
PNG+PDF figures). No solver runs, no service rates - purely instance descriptives.

Run: ./venv/bin/python3 -m rtv_solver.pipeline.analyze_lilim_instance_types
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "solutions/li_lim/manifests"
OUT = REPO / "figures_export/LiLim Dataset"

GROUP_ORDER = ["LC1", "LC2", "LR1", "LR2", "LRC1", "LRC2"]
GEO_COLOR = {"LC": "#1C7293", "LR": "#F2A541", "LRC": "#132A4C"}


def group_of(name: str) -> str:
    m = re.match(r"(lrc|lc|lr)(\d)", name)
    return f"{m.group(1).upper()}{m.group(2)}"


def analyze_instance(name: str) -> tuple[dict, pd.DataFrame]:
    p = json.load(open(MANIFESTS / f"{name}.json"))
    reqs = p["requests"]

    runs = p["driver_runs"]
    horizon = runs[0]["state"]["end_time"]
    veh_capacity = runs[0]["state"]["am_capacity"]
    route_sizes = []
    for dr in runs:
        ids = {int(s["booking_id"]) for s in dr["manifest"] if int(s["booking_id"]) >= 0}
        if ids:
            route_sizes.append(len(ids))

    rows = []
    for r in reqs:
        px, py = r["pickup_pt"]["lon"], r["pickup_pt"]["lat"]
        dx, dy = r["dropoff_pt"]["lon"], r["dropoff_pt"]["lat"]
        direct = float(np.hypot(px - dx, py - dy))
        pw = r["pickup_time_window_end"] - r["pickup_time_window_start"]
        dw = r["dropoff_time_window_end"] - r["dropoff_time_window_start"]
        rows.append({
            "instance": name, "group": group_of(name), "horizon": horizon,
            "pickup_start": r["pickup_time_window_start"],
            "pickup_window": pw, "dropoff_window": dw,
            "pickup_window_rel": pw / horizon, "dropoff_window_rel": dw / horizon,
            "pickup_start_rel": r["pickup_time_window_start"] / horizon,
            "ready_zero": int(r["pickup_time_window_start"] == 0),
            "demand": r["am"], "wc": r["wc"], "direct_dist": direct,
            # derived: detour budget = latest dropoff - (earliest pickup + pickup service + direct travel)
            "ride_slack": r["dropoff_time_window_end"]
            - (r["pickup_time_window_start"] + r["pickup_service_time"] + direct),
        })
    df = pd.DataFrame(rows)

    # NN distance over all pickup + dropoff nodes (no depot) - reproduces the values in docs/LiLim_Dataset.md
    pts = np.array([[r[k]["lon"], r[k]["lat"]] for r in reqs for k in ("pickup_pt", "dropoff_pt")])
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)

    inst = {
        "instance": name, "group": group_of(name), "geo": group_of(name)[:-1], "class": int(group_of(name)[-1]),
        "n_requests": len(reqs), "capacity": veh_capacity, "horizon": horizon,
        "ref_routes": len(route_sizes), "req_per_route_mean": float(np.mean(route_sizes)),
        "req_per_route_max": int(np.max(route_sizes)),
        "avg_nn_dist": float(d.min(axis=1).mean()),
        "total_demand_over_capacity": float(df["demand"].sum() / veh_capacity),
        "wc_share": float((df["wc"] > 0).mean()),
    }
    return inst, df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in MANIFESTS.glob("*.json"))
    insts, dfs = [], []
    for n in names:
        i, d = analyze_instance(n)
        insts.append(i)
        dfs.append(d)
    inst_df = pd.DataFrame(insts)
    req_df = pd.concat(dfs, ignore_index=True)
    inst_df.to_csv(OUT / "lilim_instance_descriptives.csv", index=False)

    # ---- group table: instance-level (mean +- std over instances) + request-level (pooled) ----
    rows = []
    for g in GROUP_ORDER:
        I = inst_df[inst_df.group == g]
        R = req_df[req_df.group == g]
        rows.append({
            "group": g, "n_instances": len(I), "n_requests_per_inst": I.n_requests.mean(),
            "horizon": I.horizon.mean(), "capacity": I.capacity.mean(),
            "ref_routes_mean": I.ref_routes.mean(), "ref_routes_std": I.ref_routes.std(),
            "req_per_route_mean": I.req_per_route_mean.mean(), "req_per_route_std": I.req_per_route_mean.std(),
            "avg_nn_dist": I.avg_nn_dist.mean(),
            "pickup_win_mean": R.pickup_window.mean(), "pickup_win_median": R.pickup_window.median(),
            "pickup_win_rel_mean": R.pickup_window_rel.mean(),
            "dropoff_win_mean": R.dropoff_window.mean(), "dropoff_win_rel_mean": R.dropoff_window_rel.mean(),
            "ready_zero_share": R.ready_zero.mean(),
            "demand_mean": R.demand.mean(), "demand_over_cap_mean": I.total_demand_over_capacity.mean(),
            "wc_share": R.wc.gt(0).mean(),
            "direct_dist_mean": R.direct_dist.mean(), "direct_dist_median": R.direct_dist.median(),
            "ride_slack_mean": R.ride_slack.mean(), "ride_slack_min": R.ride_slack.min(),
        })
    grp = pd.DataFrame(rows)
    grp.to_csv(OUT / "lilim_group_descriptives.csv", index=False)

    # ---- LaTeX table (English) ----
    def pm(m, s): return f"{m:.1f} $\\pm$ {s:.1f}"
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Descriptive statistics of the Li\&Lim instances by geography and class. "
        r"Instance-level values are means over instances ($\pm$ std); window and distance values are means over all requests of the group.}",
        r"\label{tab:lilim_descriptives}",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"Group & \#Inst. & \#Req./inst. & Horizon & Capacity & Ref.\ routes & Req./route & Pickup win. & Dropoff win.\\",
        r"\midrule",
    ]
    for _, r in grp.iterrows():
        lines.append(
            f"{r.group} & {int(r.n_instances)} & {r.n_requests_per_inst:.1f} & {r.horizon:.0f} & {r.capacity:.0f} & "
            f"{pm(r.ref_routes_mean, r.ref_routes_std)} & {pm(r.req_per_route_mean, r.req_per_route_std)} & "
            f"{r.pickup_win_mean:.1f} & {r.dropoff_win_mean:.1f}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "lilim_group_descriptives_table.tex").write_text("\n".join(lines) + "\n")

    # ---- figures ----
    def savefig(fig, stem):
        fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    def box(ax, col, title, ylabel):
        data = [req_df[req_df.group == g][col].values for g in GROUP_ORDER]
        bp = ax.boxplot(data, tick_labels=GROUP_ORDER, patch_artist=True, showfliers=False)
        for patch, g in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(GEO_COLOR[g[:-1]])
            patch.set_alpha(0.55 if g.endswith("1") else 0.95)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    box(axes[0, 0], "pickup_window", "Pickup window length (absolute)", "time units")
    box(axes[0, 1], "pickup_window_rel", "Pickup window length / horizon", "fraction of horizon")
    box(axes[1, 0], "dropoff_window", "Dropoff window length (absolute)", "time units")
    box(axes[1, 1], "dropoff_window_rel", "Dropoff window length / horizon", "fraction of horizon")
    fig.suptitle("Li&Lim: time window length by group (pooled requests, 56 instances, outliers hidden)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "lilim_time_window_length_by_group")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ci, geo in enumerate(["LC", "LR", "LRC"]):
        for ri, cls in enumerate(["1", "2"]):
            ax = axes[ri, ci]
            R = req_df[req_df.group == f"{geo}{cls}"]
            ax.hist(R.pickup_start_rel, bins=np.linspace(0, 1, 21), color=GEO_COLOR[geo], alpha=0.85)
            ax.set_title(f"{geo}{cls}  (n={len(R)} requests)", fontsize=10)
            ax.set_xlabel("earliest pickup / horizon")
            if ci == 0:
                ax.set_ylabel("requests")
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Li&Lim: distribution of earliest pickup time over the horizon (pooled)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "lilim_pickup_time_distribution_by_group")

    def bars(ax, col, title, ylabel):
        m = [inst_df[inst_df.group == g][col].mean() for g in GROUP_ORDER]
        s = [inst_df[inst_df.group == g][col].std() for g in GROUP_ORDER]
        ax.set_ylim(0, max(a + (b if b == b else 0) for a, b in zip(m, s)) * 1.15)
        for i, g in enumerate(GROUP_ORDER):
            ax.bar(i, m[i], yerr=s[i], capsize=3, color=GEO_COLOR[g[:-1]], alpha=0.55 if g.endswith("1") else 0.95)
            ax.text(i, m[i] + (s[i] if s[i] == s[i] else 0) + 0.01 * ax.get_ylim()[1], f"{m[i]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(6))
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    bars(axes[0], "ref_routes", "Routes in reference solution", "routes per instance")
    bars(axes[1], "req_per_route_mean", "Requests per route", "mean requests per route")
    bars(axes[2], "avg_nn_dist", "Spatial spread (avg. NN distance)", "distance units")
    fig.suptitle("Li&Lim: reference-solution structure and spatial spread (mean ± std over instances)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "lilim_reference_structure_by_group")

    print("instances:", len(inst_df), "| requests:", len(req_df))
    print("wheelchair requests in total:", int((req_df.wc > 0).sum()))
    print("min ride_slack over all requests:", float(req_df.ride_slack.min()))
    print(grp.round(2).T.to_string())


if __name__ == "__main__":
    main()
