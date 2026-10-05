"""
2026-10-05: central place for starting a Gurobi environment with retry.

Why: every ILP solve (co_scoreMaximization.py, co_tripCostMinimization.py, swap_handler.py)
starts its own gp.Env, and with the LRZ token-server license each env.start() has to reach
license1.lrz.de:41954. SRL runs do ~0.5 M solves per 30-epoch run (rough estimate: ~40 oracle
solves per step, see chat), so a single short token-server outage raised
GurobiError(10009, "Failed to connect to token server ...") out of env.start() and killed the
whole ~20 h run (three cm4 jobs died together after 6:40:31, earlier sweeps likewise). Retrying
with backoff turns such hiccups into a short wait; results are unchanged (same solves).

Only "token server" connection errors are retried - any other GurobiError (e.g. "User name
mismatch" on the local WLS license) is raised immediately, retrying it would only waste time.
"""
import logging
import time

import gurobipy as gp

from rtv_solver.util.logger import BASIC_LOGGER

console_logger = logging.getLogger(BASIC_LOGGER)

_TOKEN_SERVER_MARKER = "token server"


def start_gurobi_env(
    output_flag: int = 0,
    max_wait_s: float = 600.0,
    first_delay_s: float = 5.0,
    max_delay_s: float = 60.0,
) -> gp.Env:
    """Return a started gp.Env (usable as `with start_gurobi_env() as env:`).

    On a token-server connection error, waits first_delay_s, doubling up to max_delay_s, and
    retries until max_wait_s of total waiting is used up, then re-raises the last error.
    """
    waited = 0.0
    delay = first_delay_s
    attempt = 0
    while True:
        attempt += 1
        env = None
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", output_flag)
            env.start()
            return env
        except gp.GurobiError as exc:
            if env is not None:
                try:
                    env.dispose()
                except Exception:
                    pass
            if _TOKEN_SERVER_MARKER not in str(exc).lower() or waited >= max_wait_s:
                raise
            sleep_s = min(delay, max_delay_s, max_wait_s - waited)
            console_logger.warning(
                "Gurobi token server not reachable (attempt %d), retrying in %.0f s "
                "(waited %.0f of %.0f s): %s", attempt, sleep_s, waited, max_wait_s, exc,
            )
            time.sleep(sleep_s)
            waited += sleep_s
            delay = min(delay * 2, max_delay_s)
