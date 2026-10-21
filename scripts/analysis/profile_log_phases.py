"""
2026-10-14: where does the time go in a SIL/COAML training log? Splits every rolling-horizon window
("=== Iteration i-j COAML Solver (mode: train|eval|optimal) at time T ===") into phases using the timings the
solver prints itself, and attributes the rest to "other" (incl. FY loss / perturbed ILP samples / imitation scoring):
  rtv        "N RTV combos / M feasible trip costs generated: S s"   (trip enumeration + costs, all cardinalities)
  features   "N features for M items created in S s"
  ilp_score  "[CO_ScoreMaximization] ILP solved in S s"
  ilp_default "[CO_TripCostMinimization] ILP solved in S s"   (reference solve every window)
  other      window wall time minus the above (loss/sampling/imitation scoring/bookkeeping)
Window wall time = timestamp of the next window header (or next file load / end of log) minus this header.
Files are attributed via the last "Loading input data from .../<stem>.json" line.

Usage: python3 scripts/analysis/profile_log_phases.py <log> [--top N]
Output: totals per mode (train/eval/optimal) and per phase, per-file totals, and the N slowest windows.
"""
import re
import sys
from collections import defaultdict
from datetime import datetime

TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")
LOAD = re.compile(r"Loading input data from \S*/(\w+)\.json")
HDR = re.compile(r"=== Iteration (\S+) COAML Solver \(mode: (\w+)\) at time (\d+)")
RTV = re.compile(r"(\d+) RTV combos / (\d+) feasible trip costs generated: ([\d.]+)s")
FEAT = re.compile(r"(\d+) features for (\d+) items created in ([\d.]+) s")
ILPS = re.compile(r"\[CO_ScoreMaximization\] ILP solved in ([\d.]+) s")
ILPD = re.compile(r"\[CO_TripCostMinimization\] ILP solved in ([\d.]+) s")


def ts(line):
    m = TS.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") if m else None


def main(path, top=8):
    windows = []  # dicts
    cur = None
    stem = None
    last_t = None
    for line in open(path, errors="ignore"):
        t = ts(line)
        if t is None:
            continue
        last_t = t
        m = LOAD.search(line)
        if m:
            if cur:
                cur["end"] = t
                windows.append(cur); cur = None
            stem = m.group(1)
            continue
        m = HDR.search(line)
        if m:
            if cur:
                cur["end"] = t
                windows.append(cur)
            cur = dict(stem=stem, it=m.group(1), mode=m.group(2), start=t, end=None, rtv=0.0, feat=0.0, ilps=0.0, ilpd=0.0,
                       trips=0, req=0)
            continue
        if cur is None:
            continue
        for rx, key in ((RTV, "rtv"), (FEAT, "feat"), (ILPS, "ilps"), (ILPD, "ilpd")):
            m = rx.search(line)
            if m:
                cur[key] += float(m.group(3 if key in ("rtv", "feat") else 1))
                if key == "rtv":
                    cur["trips"] = int(m.group(2))
    if cur:
        cur["end"] = last_t
        windows.append(cur)
    for w in windows:
        w["wall"] = (w["end"] - w["start"]).total_seconds()
        w["other"] = w["wall"] - w["rtv"] - w["feat"] - w["ilps"] - w["ilpd"]

    print(f"log: {path}\nwindows: {len(windows)}")
    for mode in sorted({w["mode"] for w in windows}):
        ws = [w for w in windows if w["mode"] == mode]
        tot = sum(w["wall"] for w in ws)
        print(f"\n== mode {mode}: {len(ws)} windows, {tot/3600:.2f} h wall")
        for k, name in (("rtv", "RTV generation"), ("feat", "features"), ("ilps", "ILP score"), ("ilpd", "ILP default"), ("other", "other (loss/sampling/scoring)")):
            s = sum(w[k] for w in ws)
            print(f"   {name:32s} {s/3600:7.2f} h  {100*s/tot:5.1f} %")
    per_file = defaultdict(lambda: [0.0, 0, 0])
    for w in windows:
        a = per_file[(w["stem"], w["mode"])]
        a[0] += w["wall"]; a[1] += 1; a[2] = max(a[2], w["trips"])
    print("\n== per file (stem, mode): hours, windows, max feasible trip costs in a window")
    for (s, m), a in sorted(per_file.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"   {s:8s} {m:8s} {a[0]/3600:6.2f} h  {a[1]:4d} windows  max trips {a[2]}")
    print(f"\n== {top} slowest windows")
    for w in sorted(windows, key=lambda w: -w["wall"])[:top]:
        print(f"   {w['stem']} {w['mode']} it {w['it']}: wall {w['wall']:.1f}s rtv {w['rtv']:.1f} feat {w['feat']:.1f} "
              f"ilpS {w['ilps']:.1f} ilpD {w['ilpd']:.1f} other {w['other']:.1f}  trips {w['trips']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 8
    main(args[0], top)
