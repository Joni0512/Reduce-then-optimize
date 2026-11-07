"""2026-11-06: table of all NYC RHO runs found under outputs/**/new_tests/nyc_rho*/ (standard library only, for the cluster login node).
Reads final/results.json: serviced, total_requests, vmt (m), total_time (solver wall-clock s). The label is the path below new_tests/ up to the run folder
(contains variant, seed and batch_interval/step_size, e.g. nyc_rho_baseline_bi1800_ss900/val_seed1/run_offline_mc3_bi1800_ss900).
Usage: python3 scripts/analysis/eval_nyc_rho_cluster_runs.py [substring-filter]
"""
import glob
import json
import os
import sys

flt = sys.argv[1] if len(sys.argv) > 1 else ""
rows = []
for f in sorted(glob.glob("outputs/**/new_tests/nyc_rho*/**/final/results.json", recursive=True)):
    if flt and flt not in f:
        continue
    st = json.load(open(f))["stats"]
    run_dir = os.path.dirname(os.path.dirname(f))
    label = run_dir.split("new_tests/", 1)[1].rsplit("/", 1)[0]  # drop the timestamped run folder
    rows.append((label, st["serviced"], st["total_requests"], st["vmt"] / 1000.0, st["total_time"]))
print("%-95s %12s %8s %8s %9s" % ("run", "served", "rate_%", "VMT_km", "time_s"))
for label, s, n, vmt, t in rows:
    print("%-95s %5d/%-5d %8.2f %8.1f %9.1f" % (label, s, n, 100.0 * s / n, vmt, t))
