"""
2026-11-14: "no fixed depot" on SF seed 42 (user request: vehicles start randomly instead of at a depot, then
"4 mehr seeds" -> start seeds 42-46). Compares, all with 150 vehicles, batch_interval=900/step_size=450, time-rule
filter off, max_cardinality=2, capacity 2, solver seed 42, keep_active=False:
  * depot start + depot return  (fleet ablation, job 5732035, submit_sf_lodes_rho_fleet_ablation_bi900_ss450.sbatch)
  * depot start, no return      (job 5733840_0, submit_sf_lodes_rho_nodepot_v150.sbatch)
  * random start, no return     (job 5734247, start seeds 42-46, submit_sf_lodes_rho_nodepot_v150_startseeds.sbatch)
Random start = each vehicle starts at the pickup node of a request of THIS instance drawn uniformly without
replacement (np.random.default_rng(start_seed)). The start manifests were rebuilt locally with the same logic;
consistency check below: every vehicle's first stop must be reachable from its start in time.
Reachability / distance bins = scripts/analysis/sf_rho_reachability.analyse().
Output: figures_export/SF/sf_rho_nodepot_startseeds_{service,reachability,map}.{png,pdf} + _table.csv
2026-11-14: + starts drawn from the pickups of instance seed 1 / seed 2 (oracle check, job 5734394).
"""
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sf_rho_reachability as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_export" / "SF" / "sf_rho_nodepot_startseeds"
SEEDS = (42, 43, 44, 45, 46)
RUNS = [("depot start + return", "outputs/new_tests/sf_lodes_fleet_ablation/bi900_ss450_mc2_v150_notimerule", None),
        ("depot start, no return", "outputs/new_tests/sf_lodes_nodepot/bi900_ss450_mc2_v150_notimerule_noreturn_depotstart", None)] + \
       [("random start, no return", f"outputs/new_tests/sf_lodes_nodepot/bi900_ss450_mc2_v150_notimerule_noreturn_randomstart_s{s}", s)
        for s in SEEDS] + \
       [(f"start from seed {src}, no return", f"outputs/new_tests/sf_lodes_nodepot/bi900_ss450_mc2_v150_notimerule_noreturn_fromseed{src}_s{s}", s)
        for src in (1, 2) for s in SEEDS]  # 2026-11-14: oracle check, starts from another instance (job 5734394)
BINS = [0, 600, 900, 1200, 1500, 1800, 10_000]
BIN_LABELS = ["< 10 min", "10-15 min", "15-20 min", "20-25 min", "25-30 min", "> 30 min"]
GROUP_COLOR = {"depot start + return": "#2a78d6", "depot start, no return": "#eb6834", "random start, no return": "#1baf7a",
               "start from seed 1, no return": "#eda100", "start from seed 2, no return": "#e87ba4"}  # 2026-11-14: + foreign starts
GROUP_LEGEND = {"random start, no return": "random start from seed 42 = own requests (oracle), no return",
                "start from seed 1, no return": "start from seed-1 pickups (snapped), no return",
                "start from seed 2, no return": "start from seed-2 pickups (snapped), no return"}
INK, MUTED, GRID, GRAY, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2", "#2a78d6", "#eb6834"
FOOT = ("150 vehicles, batch_interval=900/step_size=450, time-rule filter off, max_cardinality=2, capacity 2, keep_active=False, "
        "solver seed 42; LRZ cluster serial_std.\nRandom start: each vehicle starts at the pickup of a request of this instance "
        "drawn uniformly without replacement (start seeds 42-46). Seed 1/2: pickups of another instance (same generator), "
        "snapped to the nearest seed-42 node (median ~0.2 km, max 1.7-7.1 km).")


def load(label, d, seed):
    f = glob.glob(str(ROOT / d / "run_*/*/final/results.json"))[0]
    s = json.loads(Path(f).read_text())["stats"]
    r = R.analyse(d)
    sv = np.array([x["served"] for x in r["rows"]])
    unr = np.array([x["unreachable"] for x in r["rows"]])
    dt = np.array([x["depot_tt"] for x in r["rows"]])
    idx = np.digitize(dt, BINS) - 1
    row = {"setup": label, "start_seed": seed if seed is not None else "", "served": int(s["serviced"]),
           "service_rate": s["serviced"] / s["total_requests"], "rejected_reachable": int((~sv & ~unr).sum()),
           "rejected_unreachable": int((~sv & unr).sum()), "wait_s": round(s["average_wait_time"], 1),
           "detour_s": round(s["average_detour"], 1), "vehicle_travel_time_h": round(s["vmt"] / 3600, 1),
           "depot_movements": s["depot_movements"], "solver_time_s": round(s["total_time"], 1),
           **{f"rate_{b}": round(float(sv[idx == i].mean()), 4) for i, b in enumerate(BIN_LABELS)},
           "source": str(Path(f).relative_to(ROOT))}
    # consistency check of the locally rebuilt start manifest + advantage of the requests whose pickup is a start node
    if seed is not None:
        cfg = json.loads((Path(f).parent.parent / "config.json").read_text())["config_dict"]
        inp = json.loads((ROOT / cfg["INPUT_FILE"]).read_text())
        p = json.loads((Path(f).parent / "result_driver_runs.json").read_text())
        tt = np.asarray(p["travel_time_matrix"], float)
        start = {int(dr["state"]["run_id"]): int(dr["state"]["loc"]["node_id"]) for dr in inp["driver_runs"]}
        bad = 0
        for dr in p["driver_runs"]:
            st = sorted(dr["manifest"], key=lambda x: x["order"])
            if st and float(st[0]["scheduled_time"]) + 1e-6 < tt[start[int(dr["state"]["run_id"])], int(st[0]["loc"]["node_id"])]:
                bad += 1
        assert bad == 0, f"start seed {seed}: {bad} vehicles cannot reach their first stop -> local manifest != cluster manifest"
        info = inp["vehicle_start_info"]
        if "booking_ids" in info:  # own-instance starts only; foreign ids belong to another instance (2026-11-14)
            ids = set(info["booking_ids"])
            row["served_start_requests"] = round(float(np.mean([x["served"] for x in r["rows"] if x["id"] in ids])), 4)
            row["served_other_requests"] = round(float(np.mean([x["served"] for x in r["rows"] if x["id"] not in ids])), 4)
        if "snap_dist_m" in info:  # 2026-11-14
            row["snap_median_m"] = round(float(np.median(info["snap_dist_m"])), 0)
            row["snap_max_m"] = round(float(np.max(info["snap_dist_m"])), 0)
    return row, r


def style(ax):
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    loaded = [load(*x) for x in RUNS]
    rows = [x[0] for x in loaded]
    groups = list(GROUP_COLOR)

    # --- chart 1: service rate overall + by depot distance (random start: mean over seeds, whisker = min-max) ---
    fig, ax = plt.subplots(figsize=(15, 5.8))
    keys = ["service_rate"] + [f"rate_{b}" for b in BIN_LABELS]
    xb = np.concatenate([[0.0], np.arange(len(BIN_LABELS)) + 1.4])
    wb = 0.8 / len(groups)  # 2026-11-14: scales with the number of groups
    for j, g in enumerate(groups):
        gr = [r for r in rows if r["setup"] == g]
        vals = np.array([[r[k] for k in keys] for r in gr])
        mean = vals.mean(0)
        pos = xb + (j - (len(groups) - 1) / 2) * (wb + 0.01)
        lab = GROUP_LEGEND.get(g, g) + (f" (mean of {len(gr)} start seeds, whisker min-max)" if len(gr) > 1 else "")
        ax.bar(pos, mean, width=wb, color=GROUP_COLOR[g], edgecolor="white", linewidth=1, label=lab)
        if len(gr) > 1:
            ax.vlines(pos, vals.min(0), vals.max(0), color=INK, linewidth=1)
        for p_, v, top in zip(pos, mean, vals.max(0)):
            ax.text(p_, top + 0.012, f"{v:.0%}" if p_ > 0.5 else f"{v:.1%}", ha="center", va="bottom", fontsize=6, color=INK)
    r0 = loaded[0][1]
    idx = np.digitize([x["depot_tt"] for x in r0["rows"]], BINS) - 1
    counts = [len(idx)] + [int((idx == i).sum()) for i in range(len(BIN_LABELS))]
    ax.set_xticks(xb, [f"{b}\n(n={c})" for b, c in zip(["All requests"] + BIN_LABELS, counts)])
    ax.axvline(0.7, color=GRID, linewidth=1.5)
    ax.set_xlabel("Overall  |  by free-flow travel time from the (former) depot to the pickup", color=MUTED)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Service rate", color=MUTED)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)  # 2026-11-14: below the axes, lower left covered bars
    ax.set_title("RHO on SF seed 42: depot vs. random vehicle starts", fontsize=11, loc="left", color=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}_service.{ext}", dpi=200)
    plt.close(fig)

    # --- chart 2: why requests are rejected, every run ---
    fig, ax = plt.subplots(figsize=(17, 5.4))  # 2026-11-14: 17 runs
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for key, color, lab in (("served", BLUE, "served"), ("rejected_reachable", GRAY, "rejected, reachable (optimistic bound)"),
                            ("rejected_unreachable", ORANGE, "rejected, no vehicle could reach it in time")):
        vals = np.array([r[key] for r in rows], dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.6, color=color, edgecolor="white", linewidth=2, label=lab)
        for xi, b, v in zip(x, bottom, vals):
            if v >= 40:  # 2026-11-14: thin top segments (26/28) were clipped
                ax.text(xi, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=9, color="white" if color != GRAY else INK)
        bottom += vals
    short = {"depot start + return": "depot\n+ return", "depot start, no return": "depot\nno return",
             "random start, no return": "own req.\n(oracle)", "start from seed 1, no return": "from\nseed 1",
             "start from seed 2, no return": "from\nseed 2"}  # 2026-11-14: short labels for 17 bars
    ax.set_xticks(x, [short[r["setup"]] + (f"\ns{r['start_seed']}" if r["start_seed"] != "" else "") for r in rows], fontsize=7.5)
    ax.set_ylim(0, 1000)
    ax.set_ylabel("Requests (of 1000)", color=MUTED)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_title("RHO on SF seed 42: why requests are rejected, depot vs. random vehicle starts", fontsize=11, loc="left", color=INK)
    fig.text(0.01, 0.01, FOOT + "\nReachability: nearest vehicle (last stop by the time the request first reaches trip generation), "
             "ignoring its other commitments.", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}_reachability.{ext}", dpi=200)
    plt.close(fig)

    # --- chart 3: map, depot start without return vs random start (seed 42) ---
    from matplotlib.collections import LineCollection
    f = glob.glob(str(ROOT / loaded[0][1]["source"]))[0]
    p = json.loads(Path(f).read_text())
    pick = {r["booking_id"]: (r["pickup_pt"]["lon"], r["pickup_pt"]["lat"]) for r in p["requests"]}
    depot = (p["depot"]["pt"]["lon"], p["depot"]["pt"]["lat"])
    roads_f = ROOT / "dataset" / "lodes_sf" / "osm" / "major_roads_segments.npy"
    roads = np.load(roads_f) if roads_f.exists() else None
    panels = [(loaded[1], "depot start, no return"), (loaded[2], "start at own requests (oracle), start seed 42"),
              (loaded[7], "start from seed-1 pickups (snapped), start seed 42")]  # 2026-11-14: + foreign-start panel
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), sharex=True, sharey=True)
    for ax, ((row, r), title) in zip(axes, panels):
        if roads is not None:
            ax.add_collection(LineCollection(roads.reshape(-1, 2, 2), colors="#d8d7d2", linewidths=0.4, zorder=0))
        grp = {"served": [], "reach": [], "unreach": []}
        for xr in r["rows"]:
            grp["served" if xr["served"] else ("unreach" if xr["unreachable"] else "reach")].append(pick[xr["id"]])
        for k, color, lab in (("served", BLUE, "served"), ("reach", GRAY, "rejected, reachable"),
                              ("unreach", ORANGE, "rejected, unreachable")):
            pts = np.asarray(grp[k])
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=9, color=color, edgecolor="white", linewidth=0.3, label=f"{lab} ({len(pts)})", zorder=2)
        if row["start_seed"] == "":
            ax.scatter(*depot, marker="*", s=260, color=INK, edgecolor="white", linewidth=1, zorder=3, label="depot (all starts)")
        else:
            cfg = json.loads((Path(ROOT / row["source"]).parent.parent / "config.json").read_text())["config_dict"]
            inp = json.loads((ROOT / cfg["INPUT_FILE"]).read_text())
            st = np.array([(dr["state"]["loc"]["lon"], dr["state"]["loc"]["lat"]) for dr in inp["driver_runs"]])
            ax.scatter(st[:, 0], st[:, 1], marker="^", s=22, facecolor="none", edgecolor=INK, linewidth=0.7, zorder=3,
                       label=f"vehicle starts ({len(st)})")
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.set_aspect(1 / np.cos(np.radians(37.75)))
        ax.tick_params(colors=MUTED, labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=True, framealpha=0.9, fontsize=7.5, loc="upper right")
    lons = np.array([v[0] for v in pick.values()])
    lats = np.array([v[1] for v in pick.values()])
    axes[0].set_xlim(lons.min() - 0.03, lons.max() + 0.03)
    axes[0].set_ylim(lats.min() - 0.03, lats.max() + 0.03)
    fig.suptitle("RHO on SF seed 42: pickups served / rejected, depot vs. random vehicle starts", fontsize=12, x=0.01, ha="left", color=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}_map.{ext}", dpi=200)
    plt.close(fig)

    fields = list(dict.fromkeys(k for r in rows for k in r))  # 2026-11-14: union, foreign rows have other extra keys
    with open(f"{OUT}_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "source"})


if __name__ == "__main__":
    main()
