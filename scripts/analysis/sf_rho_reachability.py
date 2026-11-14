"""
2026-11-11: why does RHO serve only ~50 % on SF? Reachability check of the rejected requests (user request, see chat).
Hypothesis: with the time-rule filter a request reaches trip generation only shortly before its pickup window opens
(earliest_pickup < t + step_size instead of < t + batch_interval), so with 300 s windows and long approaches in the Bay
Area no vehicle can get there in time - independent of fleet size.

Per run (final/result_driver_runs.json + config.json) and per request:
  * t_vis = first RHO solve time t_k = t0 + k*step_size (t0 = max(0, first pickup start - batch_interval), as in
    offline_rtv_solver.solve_rtv) at which the request reaches trip generation: pickup window overlaps
    [t_k, t_k + batch_interval) and, with the time-rule filter, earliest_pickup < t_k + step_size
    (request_pruner.time_rule_filter_requests).
  * best_arrival = min over vehicles of  max(t_vis, ready_v) + tt(node_v, pickup), where (node_v, ready_v) is the vehicle's
    last stop with scheduled_time <= t_vis (ready = scheduled_time + dwell) or the depot at t=0.
    OPTIMISTIC lower bound: it ignores passengers on board / stops the vehicle is already committed to.
  * rejected request with best_arrival > latest_pickup = "unreachable when it became visible" (no vehicle could make it
    even if it were free). The remaining rejections were reachable at least under this optimistic bound.
Vehicle positions come from each run's own final schedule, so the bound is policy-dependent (where the vehicles happen
to be), not a property of the instance alone.
Output: figures_export/SF/sf_rho_reachability.{png,pdf,csv}
"""
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_export" / "SF" / "sf_rho_reachability"
DIRS = ["outputs/outputs/new_tests/sf_lodes_smoke/bi900_ss450_seed42_cap2",
        "outputs/new_tests/sf_lodes_v100/bi900_ss450_seed42_v100_cap2_timerule",
        "outputs/new_tests/sf_lodes_smoke/bi900_ss450_seed42_cap2_notimerule",
        "outputs/new_tests/sf_lodes_v100/bi900_ss450_seed42_v100_cap2_notimerule",
        "outputs/new_tests/sf_lodes_v200/bi900_ss450_seed42_v200_cap2_timerule",  # 2026-11-11: + 200 vehicles (user request)
        "outputs/new_tests/sf_lodes_v200/bi900_ss450_seed42_v200_cap2_notimerule"]
# 2026-11-11: distributed vehicle starts (scripts/sf/make_distributed_vehicle_starts.py, user request) vs depot start, 50 vehicles
DIRS_VSTART = ["outputs/outputs/new_tests/sf_lodes_smoke/bi900_ss450_seed42_cap2",
               "outputs/new_tests/sf_lodes_smoke/bi900_ss450_seed42_cap2_notimerule",
               "outputs/outputs/new_tests/sf_lodes_vstart/bi900_ss450_seed42_v50_cap2_vstart_timerule",
               "outputs/outputs/new_tests/sf_lodes_vstart/bi900_ss450_seed42_v50_cap2_vstart_notimerule",
               # 2026-11-11: + two depots (A = old depot in SF, B = East Bay, --mode two_depots), user request
               "outputs/outputs/new_tests/sf_lodes_vstart/bi900_ss450_seed42_v50_cap2_twodepots_timerule",
               "outputs/outputs/new_tests/sf_lodes_vstart/bi900_ss450_seed42_v50_cap2_twodepots_notimerule"]
START_ORDER = {"depot": 0, "random_pickup_nodes": 1, "two_depots": 2}
START_LABEL = {"depot": "1 depot (SF)", "random_pickup_nodes": "random pickup locations", "two_depots": "2 depots (SF + East Bay)"}
# 2026-11-12: fleet-size ablation (cluster, filter off, max_cardinality=2, batch_interval=900/step_size=450), user request
DIRS_FLEET = [f"outputs/new_tests/sf_lodes_fleet_ablation/bi900_ss450_mc2_v{v}_notimerule" for v in (100, 150, 200, 250, 300)]
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#b9b8b2", "#0b0b0b", "#52514e", "#e4e3df"


def analyse(d):
    f = Path(glob.glob(str(ROOT / d / "run_*/*/final/result_driver_runs.json"))[0])
    cfg = json.loads((f.parent.parent / "config.json").read_text())["config_dict"]
    p = json.loads(f.read_text())
    bi, ss, filt = cfg["BATCH_INTERVAL"], cfg["STEP_SIZE"], cfg["USE_REQUEST_TIME_RULE_FILTER"]
    tt = np.asarray(p["travel_time_matrix"], dtype=float)
    depot = int(p["depot"]["node_id"])
    reqs = p["requests"]
    # 2026-11-11: vehicle start node from the INPUT manifest (state.loc), not always the depot (distributed starts);
    # the final driver_runs' state.loc has moved on during the run.
    # 2026-11-12: the fleet-ablation manifests (v150/v250/v300) were built on the cluster only, as copies of the v50
    # depot-start manifest (submit_sf_lodes_rho_fleet_ablation_*.sbatch) -> if the input file is missing locally, all
    # vehicles start at the depot (true for those runs by construction).
    inp_path = ROOT / cfg["INPUT_FILE"]
    inp = json.loads(inp_path.read_text()) if inp_path.exists() else {
        "driver_runs": [{"state": {"run_id": dr["state"]["run_id"], "loc": {"node_id": depot}}} for dr in p["driver_runs"]]}
    start = {int(dr["state"]["run_id"]): int(dr["state"]["loc"]["node_id"]) for dr in inp["driver_runs"]}
    distributed = any(n != depot for n in start.values())
    start_mode = inp.get("vehicle_start_info", {}).get("mode", "depot")  # 2026-11-11
    t0 = max(0, min(r["pickup_time_window_start"] for r in reqs) - bi)
    # vehicle stop timelines: (ready_time, node)
    timelines, served = [], set()
    for dr in p["driver_runs"]:
        tl = [(0.0, start[int(dr["state"]["run_id"])])]
        for s in sorted(dr["manifest"], key=lambda s: s["order"]):
            if s["booking_id"] >= 0 and s["action"] == "pickup":
                served.add(int(s["booking_id"]))
            tl.append((float(s["scheduled_time"]), int(s["loc"]["node_id"]), float(s.get("dwell", 0))))
        timelines.append(tl)

    rows = []
    for r in reqs:
        e, l = r["pickup_time_window_start"], r["pickup_time_window_end"]
        k = 0
        while True:
            t = t0 + k * ss
            if l > t and e < t + bi and (not filt or e < t + ss):
                break
            k += 1
        node = int(r["pickup_pt"]["node_id"])
        best = np.inf
        for tl in timelines:
            pos, ready = tl[0][1], 0.0
            for st in tl[1:]:
                if st[0] <= t:
                    pos, ready = st[1], st[0] + st[2]
                else:
                    break
            best = min(best, max(t, ready) + tt[pos, node])
        rows.append({"id": r["booking_id"], "served": r["booking_id"] in served, "t_vis": t, "lead_s": e - t,
                     "best_arrival": best, "latest_pickup": l, "unreachable": best > l,
                     "depot_tt": tt[depot, node]})
    return {"vehicles": len(p["driver_runs"]), "filter": filt, "distributed": distributed, "start_mode": start_mode, "bi": bi, "ss": ss, "rows": rows, "source": str(f.relative_to(ROOT))}


def main():
    res = sorted([analyse(d) for d in DIRS], key=lambda r: (not r["filter"], r["vehicles"]))  # 2026-11-11: filter on first, then fleet size
    summary = []
    for r in res:
        rows = r["rows"]
        n = len(rows)
        srv = sum(x["served"] for x in rows)
        unr = sum((not x["served"]) and x["unreachable"] for x in rows)
        rej_reach = n - srv - unr
        lead = np.array([x["lead_s"] for x in rows])
        summary.append({"vehicles": r["vehicles"], "time_rule_filter": r["filter"], "batch_interval": r["bi"], "step_size": r["ss"],
                        "served": srv, "rejected_unreachable": unr, "rejected_reachable": rej_reach,
                        "served_but_flagged_unreachable": sum(x["served"] and x["unreachable"] for x in rows),
                        "lead_time_median_s": float(np.median(lead)), "lead_time_max_s": float(lead.max()),
                        "source": r["source"]})
    with open(f"{OUT}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print({k: v for k, v in s.items() if k != "source"})

    # served share by depot travel time (geography), per run
    bins = [0, 600, 900, 1200, 1500, 1800, 10_000]
    print("served share by depot->pickup travel time bin", bins)
    for r in res:
        dt = np.array([x["depot_tt"] for x in r["rows"]])
        sv = np.array([x["served"] for x in r["rows"]])
        idx = np.digitize(dt, bins) - 1
        print(f"  {r['vehicles']} veh filter={r['filter']}:",
              [f"{sv[idx == i].mean():.0%} (n={int((idx == i).sum())})" for i in range(len(bins) - 1)])

    # 2026-11-11: second figure (user request "Service-Rate-Grafik"): served share per depot->pickup travel-time bin, one
    # grouped bar per run; bin sizes n below the x labels. Colors = dataviz categorical slots 1-4 in fixed order.
    bin_labels = ["< 10 min", "10-15 min", "15-20 min", "20-25 min", "25-30 min", "> 30 min"]
    run_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    # 2026-11-11: + overall service rate as the first group "all requests" (user request), separated by a gap and a divider
    fig, ax = plt.subplots(figsize=(14, 5.4))
    nb = len(bins) - 1
    xb = np.concatenate([[0.0], np.arange(nb) + 1.4])
    wb = 0.8 / len(DIRS)  # 2026-11-11: bar width scales with the number of runs
    counts = None
    for j, (r, color) in enumerate(zip(res, run_colors)):
        dt = np.array([x["depot_tt"] for x in r["rows"]])
        sv = np.array([x["served"] for x in r["rows"]])
        idx = np.digitize(dt, bins) - 1
        rates = [sv.mean()] + [sv[idx == i].mean() for i in range(nb)]
        counts = [len(sv)] + [int((idx == i).sum()) for i in range(nb)]
        pos = xb + (j - (len(res) - 1) / 2) * (wb + 0.005)
        ax.bar(pos, rates, width=wb, color=color, edgecolor="white", linewidth=1,
               label=f"{r['vehicles']} vehicles, time-rule filter {'on' if r['filter'] else 'off'}")
        for p_, v in zip(pos, rates):
            ax.text(p_, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=6, color=INK)
    ax.set_xticks(xb, [f"{b}\n(n={c})" for b, c in zip(["All requests"] + bin_labels, counts)])
    ax.axvline(0.7, color=GRID, linewidth=1.5)
    ax.set_xlabel("Overall  |  by free-flow travel time depot -> pickup", color=MUTED)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Service rate", color=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("RHO on SF seed 42: overall service rate and by distance of the pickup from the depot", fontsize=11, loc="left", color=INK)
    fig.text(0.01, 0.01, "batch_interval=900/step_size=450, capacity 2, max_cardinality=3, keep_active=False, solver seed 42; "
             "all vehicles start at the depot (mean pickup location, in SF); all dropoffs are in SF county", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures_export" / "SF" / f"sf_rho_service_rate_by_depot_distance.{ext}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [f"{s['vehicles']} vehicles\nfilter {'on' if s['time_rule_filter'] else 'off'}" for s in summary]
    x = np.arange(len(summary))
    bottom = np.zeros(len(summary))
    for key, color, lab in (("served", BLUE, "served"), ("rejected_reachable", GRAY, "rejected, reachable (optimistic bound)"),
                            ("rejected_unreachable", ORANGE, "rejected, no vehicle could reach it in time")):
        vals = np.array([s[key] for s in summary], dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.6, color=color, edgecolor="white", linewidth=2, label=lab)
        for xi, b, v in zip(x, bottom, vals):
            if v >= 30:
                ax.text(xi, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=9, color="white" if color != GRAY else INK)
        bottom += vals
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1000)
    ax.set_ylabel("Requests (of 1000)", color=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_title("RHO on SF seed 42: why requests are rejected", fontsize=11, loc="left", color=INK)
    fig.text(0.01, 0.01, "batch_interval=900/step_size=450, capacity 2, max_cardinality=3, keep_active=False, solver seed 42.\n"
             "Reachability: nearest vehicle (last stop by the time the request first reaches trip generation), "
             "ignoring its other commitments", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # 2026-11-11: room for the two-line footnote
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200)
    plt.close(fig)
    vstart_chart()


def vstart_chart():
    """2026-11-11: depot start vs distributed start (50 vehicles), served share overall + by depot->pickup travel time,
    plus rejected-unreachable counts. Output: figures_export/SF/sf_rho_vstart_vs_depot.{png,pdf,csv}"""
    res = [analyse(d) for d in DIRS_VSTART]
    res.sort(key=lambda r: (not r["filter"], START_ORDER[r["start_mode"]]))
    bins = [0, 600, 900, 1200, 1500, 1800, 10_000]
    bin_labels = ["< 10 min", "10-15 min", "15-20 min", "20-25 min", "25-30 min", "> 30 min"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    nb = len(bins) - 1
    xb = np.concatenate([[0.0], np.arange(nb) + 1.4])
    wb = 0.8 / len(res)
    rows_csv = []
    fig, ax = plt.subplots(figsize=(14, 5.6))
    for j, (r, color) in enumerate(zip(res, colors)):
        dt = np.array([x["depot_tt"] for x in r["rows"]])
        sv = np.array([x["served"] for x in r["rows"]])
        idx = np.digitize(dt, bins) - 1
        rates = [sv.mean()] + [sv[idx == i].mean() for i in range(nb)]
        counts = [len(sv)] + [int((idx == i).sum()) for i in range(nb)]
        unr = sum((not x["served"]) and x["unreachable"] for x in r["rows"])
        lab = f"{START_LABEL[r['start_mode']]}, time-rule filter {'on' if r['filter'] else 'off'}"
        rows_csv.append({"start": r["start_mode"], "time_rule_filter": r["filter"],
                         "vehicles": r["vehicles"], "served": int(sv.sum()), "service_rate": round(float(sv.mean()), 4),
                         "rejected_unreachable": unr,
                         **{f"rate_{b}": round(float(v), 4) for b, v in zip(bin_labels, rates[1:])}, "source": r["source"]})
        pos = xb + (j - (len(res) - 1) / 2) * (wb + 0.005)
        ax.bar(pos, rates, width=wb, color=color, edgecolor="white", linewidth=1, label=lab)
        for p_, v in zip(pos, rates):
            ax.text(p_, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(xb, [f"{b}\n(n={c})" for b, c in zip(["All requests"] + bin_labels, counts)])
    ax.axvline(0.7, color=GRID, linewidth=1.5)
    ax.set_xlabel("Overall  |  by free-flow travel time depot -> pickup", color=MUTED)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Service rate", color=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("RHO on SF seed 42: where the vehicles start", fontsize=11, loc="left", color=INK)
    fig.text(0.01, 0.01, "batch_interval=900/step_size=450, 50 vehicles, capacity 2, max_cardinality=3, keep_active=False, solver seed 42. "
             "\nRandom pickup locations: each vehicle starts at the pickup of a randomly drawn request (seed 42). "
             "2 depots: 38 vehicles at the SF depot, 12 in the East Bay (mean of pickups > 20 min from the SF depot)",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = ROOT / "figures_export" / "SF" / "sf_rho_vstart_vs_depot"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    with open(f"{out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    for r in rows_csv:
        print({k: v for k, v in r.items() if k != "source"})


def fleet_charts():
    """2026-11-12: location hypothesis for the fleet ablation (user request "1 + 2 + Karte"):
    (1) served share by depot->pickup travel time per fleet size, (2) served / rejected reachable / rejected
    unreachable per fleet size. Output: figures_export/SF/sf_rho_fleet_ablation_{by_distance,reachability}.{png,pdf,csv}"""
    res = sorted([analyse(d) for d in DIRS_FLEET], key=lambda r: r["vehicles"])
    bins = [0, 600, 900, 1200, 1500, 1800, 10_000]
    bin_labels = ["< 10 min", "10-15 min", "15-20 min", "20-25 min", "25-30 min", "> 30 min"]
    # sequential (fleet size = magnitude): one hue light -> dark
    colors = ["#9ec5f0", "#6aa6e6", "#2a78d6", "#1d5aa6", "#123c70"]
    nb = len(bins) - 1
    xb = np.concatenate([[0.0], np.arange(nb) + 1.4])
    wb = 0.8 / len(res)
    out_rows = []
    fig, ax = plt.subplots(figsize=(13, 5.4))
    for j, (r, color) in enumerate(zip(res, colors)):
        dt = np.array([x["depot_tt"] for x in r["rows"]])
        sv = np.array([x["served"] for x in r["rows"]])
        unr = np.array([x["unreachable"] for x in r["rows"]])
        idx = np.digitize(dt, bins) - 1
        rates = [sv.mean()] + [sv[idx == i].mean() for i in range(nb)]
        counts = [len(sv)] + [int((idx == i).sum()) for i in range(nb)]
        n_unr = int(((~sv) & unr).sum())
        out_rows.append({"vehicles": r["vehicles"], "served": int(sv.sum()), "rejected_reachable": int(len(sv) - sv.sum() - n_unr),
                         "rejected_unreachable": n_unr,
                         **{f"rate_{b}": round(float(v), 4) for b, v in zip(bin_labels, rates[1:])},
                         **{f"unreachable_{b}": int(((~sv) & unr & (idx == i)).sum()) for i, b in enumerate(bin_labels)},
                         "source": r["source"]})
        pos = xb + (j - (len(res) - 1) / 2) * (wb + 0.005)
        ax.bar(pos, rates, width=wb, color=color, edgecolor="white", linewidth=1, label=f"{r['vehicles']} vehicles")
        for p_, v in zip(pos, rates):
            ax.text(p_, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(xb, [f"{b}\n(n={c})" for b, c in zip(["All requests"] + bin_labels, counts)])
    ax.axvline(0.7, color=GRID, linewidth=1.5)
    ax.set_xlabel("Overall  |  by free-flow travel time depot -> pickup", color=MUTED)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Service rate", color=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("RHO on SF seed 42, fleet size: service rate by distance of the pickup from the depot", fontsize=11, loc="left", color=INK)
    fig.text(0.01, 0.01, "batch_interval=900/step_size=450, time-rule filter off, max_cardinality=2, capacity 2, single depot (SF), "
             "keep_active=False, solver seed 42; LRZ cluster", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    base = ROOT / "figures_export" / "SF" / "sf_rho_fleet_ablation"
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}_by_distance.{ext}", dpi=200)
    plt.close(fig)
    with open(f"{base}_reachability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(out_rows))
    bottom = np.zeros(len(out_rows))
    for key, color, lab in (("served", BLUE, "served"), ("rejected_reachable", GRAY, "rejected, reachable (optimistic bound)"),
                            ("rejected_unreachable", ORANGE, "rejected, no vehicle could reach it in time")):
        vals = np.array([s_[key] for s_ in out_rows], dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.6, color=color, edgecolor="white", linewidth=2, label=lab)
        for xi, b, v in zip(x, bottom, vals):
            if v >= 30:
                ax.text(xi, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=9, color="white" if color != GRAY else INK)
        bottom += vals
    ax.set_xticks(x, [f"{s_['vehicles']} vehicles" for s_ in out_rows])
    ax.set_ylim(0, 1000)
    ax.set_ylabel("Requests (of 1000)", color=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title("RHO on SF seed 42, fleet size: why requests are rejected", fontsize=11, loc="left", color=INK)
    fig.text(0.01, 0.01, "batch_interval=900/step_size=450, time-rule filter off, max_cardinality=2, capacity 2, single depot, solver seed 42.\n"
             "Reachability: nearest vehicle (last stop by the time the request first reaches trip generation), ignoring its other commitments",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}_reachability.{ext}", dpi=200)
    plt.close(fig)
    for r in out_rows:
        print({k: v for k, v in r.items() if k != "source"})


if __name__ == "__main__":
    import sys
    if "--fleet" in sys.argv:  # 2026-11-12: only the fleet-ablation charts
        fleet_charts()
    else:
        main()
