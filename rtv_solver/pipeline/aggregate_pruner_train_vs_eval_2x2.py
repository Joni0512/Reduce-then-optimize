"""
2026-09-21: Aggregates the 2x2 diagnostic of run_eval_pruner_train_vs_eval_2x2.sh
(model trained without/with Request Pruner x evaluated without/with Request Pruner).
Per cell: pooled val service rate over the 6 class-1 val instances, summed wall-clock
runtime (first vs. last log line of each run's main.log) and the runtime saving vs.
the (baseline model, no pruner) cell - runtime saving is always reported alongside
service rate (user request 2026-09-21).

Usage: python -m rtv_solver.pipeline.aggregate_pruner_train_vs_eval_2x2 [seed=42]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

CELLS = [
    ("base_noPruner", "trained w/o pruner", "eval w/o pruner"),
    ("base_withPruner", "trained w/o pruner", "eval WITH pruner"),
    ("prunedtr_noPruner", "trained WITH pruner", "eval w/o pruner"),
    ("prunedtr_withPruner", "trained WITH pruner", "eval WITH pruner"),
]
VAL = ["lc108", "lc109", "lr111", "lr112", "lrc107", "lrc108"]
FMT = "%Y-%m-%d %H:%M:%S,%f"


def run_dir(base: Path, inst: str):
    hits = sorted((base / inst).glob("*/*/final/results.json"))
    return hits[-1].parent.parent if hits else None


def main(seed: str = "42"):
    base = Path(f"outputs/new_tests/pruner_train_vs_eval_2x2_seed{seed}")
    rows = []
    for cell, tr, ev in CELLS:
        serviced = total = 0
        minutes = 0.0
        missing = []
        per_inst = {}
        for inst in VAL:
            rd = run_dir(base / cell, inst)
            if rd is None:
                missing.append(inst)
                continue
            stats = json.load(open(rd / "final" / "results.json"))["stats"]
            serviced += stats["serviced"]
            total += stats["total_requests"]
            per_inst[inst] = f'{stats["serviced"]}/{stats["total_requests"]}'
            lines = (rd / "main.log").read_text(errors="ignore").splitlines()
            minutes += (datetime.strptime(lines[-1][:23], FMT) - datetime.strptime(lines[0][:23], FMT)).total_seconds() / 60
        rows.append((cell, tr, ev, serviced, total, minutes, missing, per_inst))

    ref_minutes = rows[0][5] if rows and not rows[0][6] else None
    print(f"{'trained':22s} {'evaluated':18s} {'service rate':>22s} {'runtime':>10s} {'saving':>8s}")
    for cell, tr, ev, s, t, m, missing, per_inst in rows:
        rate = f"{100 * s / t:5.1f}% ({s}/{t})" if t else "n/a"
        saving = f"{100 * (ref_minutes - m) / ref_minutes:6.1f}%" if ref_minutes and t else "   n/a"
        note = f"   MISSING: {','.join(missing)}" if missing else ""
        print(f"{tr:22s} {ev:18s} {rate:>22s} {m:8.1f} m {saving:>8s}{note}")
    print("\nper instance (serviced/total):")
    for cell, *_rest in rows:
        print(f"  {cell:22s} {rows[[r[0] for r in rows].index(cell)][7]}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["42"]))
