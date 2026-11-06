"""2026-11-06: evaluation of the NYC Train-day replication (nopruner vs. time rule), RHO offline,
batch_interval=3600/step_size=1800, cardinality 3, seed 1 (submit_nyc_rho3060_trainday_timerule.sbatch).
Reads <base>/<variant>_seed1/run_offline_mc3_bi3600_ss1800/*/final/results.json (stats: serviced, total_requests,
vmt in metres, total_time = solver wall-clock in seconds) and counts pruned/deferred requests per window from main.log.
Runs on the cluster login node: standard library only (no numpy, no nested f-string quotes).
Usage: python3 scripts/analysis/eval_nyc_rho3060_trainday.py [base_dir]
"""
import glob
import json
import os
import re
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "outputs/outputs/new_tests/nyc_rho3060_trainday"
VARIANTS = ["nopruner", "timerule"]

rows = []
for v in VARIANTS:
    pattern = os.path.join(BASE, v + "_seed1", "run_offline_mc3_bi3600_ss1800", "*", "final", "results.json")
    hits = sorted(glob.glob(pattern))
    if not hits:
        print("KEIN results.json fuer %s (%s)" % (v, pattern))
        continue
    res_path = hits[-1]
    st = json.load(open(res_path))["stats"]
    log_path = os.path.join(os.path.dirname(os.path.dirname(res_path)), "main.log")
    pruned = None
    windows = 0
    if os.path.exists(log_path):
        txt = open(log_path).read()
        found = re.findall(r"Request time-rule filter: .*?, (\d+) deferred\)", txt)
        if found:
            pruned = sum(int(x) for x in found)
            windows = len(found)
    rows.append((v, st["serviced"], st["total_requests"], st["vmt"] / 1000.0, st["total_time"],
                 st.get("average_wait_time"), st.get("average_detour"), pruned, windows, res_path))

print("%-10s %9s %8s %9s %10s %9s %9s %12s" %
      ("variant", "served", "rate_%", "VMT_km", "time_s", "wait_s", "detour_s", "deferred_sum"))
for v, s, n, vmt, t, w, d, p, nw, _ in rows:
    print("%-10s %4d/%-4d %8.2f %9.1f %10.1f %9.1f %9.1f %12s" %
          (v, s, n, 100.0 * s / n, vmt, t, w or 0.0, d or 0.0, "-" if p is None else "%d (%d Fenster)" % (p, nw)))
if len(rows) == 2:
    a, b = rows
    print("\nZeitregel vs. Baseline: %+0.2f %%-Punkte Service Rate, VMT %+0.1f %%, Laufzeit Faktor %.1fx schneller" %
          (100.0 * (b[1] / b[2] - a[1] / a[2]), 100.0 * (b[3] / a[3] - 1.0), a[4] / b[4]))
print("\nQuellen:")
for r in rows:
    print("  " + r[-1])
