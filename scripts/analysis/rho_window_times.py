"""2026-11-06: per-window time breakdown of an RHO offline run, read from main.log (standard library only, runs on the cluster login node).
Per window (iteration): solver time t, requests kept/total (only if the time-rule filter ran), RTV combos, feasible trip costs, RTV generation seconds
(the solver's own "feasible trip costs generated: Xs") and the window wall-clock (timestamp of the next iteration minus this one; last window: last log line).
rest = wall - RTV = ILP + simulation + logging. Prints a sum line with the RTV share.
Usage: python3 scripts/analysis/rho_window_times.py <main.log> [<main.log> ...]
"""
import re
import sys
from datetime import datetime

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+"


def ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


for path in sys.argv[1:]:
    wins = []
    last_ts = None
    for line in open(path, errors="replace"):
        m = re.match(TS, line)
        if m:
            last_ts = ts(m.group(1))
        m = re.search(TS + r".*=== Iteration (\d+) offline RTV Solver at time (\d+)", line)
        if m:
            wins.append({"ts": ts(m.group(1)), "it": int(m.group(2)), "t": int(m.group(3)), "kept": None, "total": None, "combos": None, "costs": None, "rtv": 0.0})
            continue
        if not wins:
            continue
        m = re.search(r"Request time-rule filter: kept (\d+)/(\d+) requests", line)
        if m:
            wins[-1]["kept"], wins[-1]["total"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"(\d+) RTV combos / (\d+) feasible trip costs generated: ([\d.]+)s", line)
        if m:
            wins[-1]["combos"], wins[-1]["costs"], wins[-1]["rtv"] = int(m.group(1)), int(m.group(2)), float(m.group(3))
    for i, w in enumerate(wins):
        end = wins[i + 1]["ts"] if i + 1 < len(wins) else last_ts
        w["wall"] = (end - w["ts"]).total_seconds()
    print("\n" + path)
    print("%4s %6s %12s %8s %10s %9s %9s %9s" % ("iter", "t[s]", "kept/total", "combos", "trip_costs", "RTV[s]", "wall[s]", "rest[s]"))
    for w in wins:
        kt = "-" if w["kept"] is None else "%d/%d" % (w["kept"], w["total"])
        print("%4d %6d %12s %8s %10s %9.1f %9.1f %9.1f" % (w["it"], w["t"], kt, w["combos"] if w["combos"] is not None else "-",
                                                         w["costs"] if w["costs"] is not None else "-", w["rtv"], w["wall"], w["wall"] - w["rtv"]))
    rtv = sum(w["rtv"] for w in wins)
    wall = sum(w["wall"] for w in wins)
    print("SUM windows %d: RTV %.0f s, wall %.0f s, RTV share %.0f %%, rest %.0f s" % (len(wins), rtv, wall, 100 * rtv / wall if wall else 0, wall - rtv))
