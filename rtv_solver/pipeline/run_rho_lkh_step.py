"""
2026-11-24: RHO (main.py --mode offline) with LKH-3 as the per-step optimizer instead of RTV + Gurobi ILP
(user request "LKH im RHO-Schritt", rejection variant R1 + ILP fallback, see rtv_solver/lkh_step_solver.py).

Does not touch main.py / Config: patches OnlineRTVSolver.solve_pdptw_rtv in this process only, then runs main.py
unchanged, so the rolling-horizon loop, simulate_manifest and StatsParser are exactly those of the ILP-RHO runs.

Usage (own flags first, everything after "--" goes to main.py):
  ./venv/bin/python3 -m rtv_solver.pipeline.run_rho_lkh_step --lkh_runs 1 --lkh_time_limit 10 -- \
      --mode offline --input_file solutions/li_lim/manifests/lc101.json \
      --batch_interval 40 --step_size 10 --output_dir outputs/rho_lkh/lc101
LKH counters are written to <output_dir>/lkh_step_stats.json.
"""
import argparse
import atexit
import json
import runpy
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rtv_solver.online_rtv_solver import OnlineRTVSolver
from rtv_solver.lkh_step_solver import LKHStepSolver


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        own, rest = argv[:split], argv[split + 1:]
    else:
        own, rest = argv, []
    ap = argparse.ArgumentParser()
    ap.add_argument("--lkh_runs", type=int, default=1)
    ap.add_argument("--lkh_time_limit", type=float, default=10)
    ap.add_argument("--lkh_max_drop_iters", type=int, default=10)
    ap.add_argument("--lkh_seed", type=int, default=1)
    ap.add_argument("--lkh_special", action="store_true", help="2026-11-24: add LKH SPECIAL to .par (see lkh_step_solver.py)")
    ap.add_argument("--lkh_max_trials", type=int, default=None, help="2026-11-24: LKH MAX_TRIALS (None = LKH default #nodes)")
    ap.add_argument("--lkh_keep_files", type=str, default=None, help="dir to keep each step's .pdptw/.par/.tour (debug)")
    args = ap.parse_args(own)

    original = OnlineRTVSolver.solve_pdptw_rtv
    holder = {}
    t_start = time.time()

    def patched(self, payload, iteration=0, current_time=None):
        if "solver" not in holder:
            holder["solver"] = LKHStepSolver(
                self.config, ilp_step_fn=lambda p, it, current_time=None: original(self, p, it, current_time=current_time),
                runs=args.lkh_runs, time_limit_s=args.lkh_time_limit, max_drop_iters=args.lkh_max_drop_iters,
                seed=args.lkh_seed, keep_files_dir=args.lkh_keep_files, max_trials=args.lkh_max_trials, special=args.lkh_special)
            holder["out"] = Path(self.config.OUTPUT_DIR)
        return holder["solver"].solve_pdptw_rtv(payload, iteration, current_time=current_time)

    OnlineRTVSolver.solve_pdptw_rtv = patched

    def dump():
        if "solver" in holder:
            d = asdict(holder["solver"].stats)
            d["lkh_params"] = vars(args)
            d["wall_clock_total_s"] = time.time() - t_start
            holder["out"].mkdir(parents=True, exist_ok=True)
            with open(holder["out"] / "lkh_step_stats.json", "w") as f:
                json.dump(d, f, indent=2)
            print("LKH step stats:", json.dumps(d))
    atexit.register(dump)

    sys.argv = ["rtv_solver/main.py"] + rest
    runpy.run_module("rtv_solver.main", run_name="__main__")


if __name__ == "__main__":
    main()
