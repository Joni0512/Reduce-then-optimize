"""
Builds a proper SIL "expert" manifest by merging:
- driver_runs[i].state from the ORIGINAL solve-ready manifest (fresh start_time=0,
  depot position - what build_nyc_solve_manifest.py produced)
- driver_runs[i].manifest (the solved pickup/dropoff route) from the RHO run's
  result_driver_runs.json (final state snapshot - state is stale/post-run, but
  manifest holds the correct expert route)

Root cause this fixes: using result_driver_runs.json directly as --input_file/
--val_input_file gave 0% service rate every epoch, because its vehicle `state`
reflects the END of the RHO simulation (e.g. location_dt_seconds near the end of
the window) - after clear_vehicle_manifests() strips the route, the vehicle still
"thinks" it's already late in the day, so every request from the start of the day
is infeasible. Li&Lim's manifests never hit this because their `state` was always
the initial/starting state to begin with (state = start, manifest = expert route).

Usage:
    python3 scripts/analysis/build_sil_expert_manifest.py \
        --solve-manifest solutions/nyc/manifests/nyc_real500_20160112_0610_solve.json \
        --rho-result outputs/.../final/result_driver_runs.json \
        --out solutions/nyc/manifests/nyc_real500_20160112_0610_expert.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve-manifest", required=True, help="original empty-vehicle solve-ready manifest")
    parser.add_argument("--rho-result", required=True, help="RHO run's result_driver_runs.json (final state)")
    parser.add_argument("--out", required=True, help="output path for the merged expert manifest")
    args = parser.parse_args()

    solve_data = json.loads(Path(args.solve_manifest).read_text())
    rho_data = json.loads(Path(args.rho_result).read_text())

    solve_state_by_run_id = {dr["state"]["run_id"]: dr["state"] for dr in solve_data["driver_runs"]}
    rho_manifest_by_run_id = {dr["state"]["run_id"]: dr["manifest"] for dr in rho_data["driver_runs"]}

    if set(solve_state_by_run_id) != set(rho_manifest_by_run_id):
        raise ValueError(
            f"run_id mismatch: solve-manifest has {sorted(solve_state_by_run_id)}, "
            f"rho-result has {sorted(rho_manifest_by_run_id)}"
        )

    merged_driver_runs = [
        {
            "state": solve_state_by_run_id[run_id],  # fresh, start-of-day state
            "manifest": rho_manifest_by_run_id[run_id],  # solved expert route
        }
        for run_id in sorted(solve_state_by_run_id)
    ]

    merged = {
        "depot": solve_data["depot"],
        "requests": solve_data["requests"],
        "driver_runs": merged_driver_runs,
        "travel_time_matrix": solve_data["travel_time_matrix"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged))
    print(f"Wrote merged expert manifest ({len(merged_driver_runs)} vehicles, "
          f"{len(merged['requests'])} requests) -> {out_path}")


if __name__ == "__main__":
    main()
