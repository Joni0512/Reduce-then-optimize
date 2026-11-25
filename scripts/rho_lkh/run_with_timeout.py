"""2026-11-24: wall-clock cap for one solver run (CLAUDE.md: max 2 h per instance). Replaces `perl -e alarm`, which
did not stop a 2 h 44 min lc109 LKH run. Runs the command in its own process group and kills the whole group
(incl. multiprocessing workers and LKH subprocesses) on timeout. Exit code 124 on timeout, like GNU timeout.
Usage: python3 run_with_timeout.py <seconds> <cmd> [args...]"""
import os
import signal
import subprocess
import sys

cap = float(sys.argv[1])
proc = subprocess.Popen(sys.argv[2:], start_new_session=True)
try:
    sys.exit(proc.wait(timeout=cap))
except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    print(f"TIMEOUT after {cap:.0f} s: {' '.join(sys.argv[2:])}", flush=True)
    sys.exit(124)
