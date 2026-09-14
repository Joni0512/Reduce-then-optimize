"""
2026-09-14: same-horizon variant of srl_rho_outcome_advantage_loop.py - see
chat. RHO baseline now runs at bi200/ss100, SAME as the actor (instead of
bi400/ss100) - isolates whether the actor's earlier advantage over RHO came
from a genuinely better policy, or just from RHO being handicapped by a
longer/less frequent re-optimization horizon in the original test.
"""
import rtv_solver.pipeline.srl_rho_outcome_advantage_loop as m

m.RHO_BATCH_INTERVAL = 200
m.RHO_STEP_SIZE = 100

if __name__ == "__main__":
    _output_dir = m.REPO_ROOT / "outputs" / "srl_rho_outcome_advantage_loop" / "run_samehorizon"
    _output_dir.mkdir(parents=True, exist_ok=True)
    try:
        m.run(_output_dir)
    except Exception:
        import traceback
        crash_path = _output_dir / "crash_traceback.txt"
        with open(crash_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! srl_rho_outcome_advantage_loop_samehorizon CRASHED - full traceback written to {crash_path}")
        raise
