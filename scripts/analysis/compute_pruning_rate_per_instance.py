"""
2026-10-12: per-INSTANCE Request-Pruner pruning rate from SIL training logs, to test whether the pruner prunes
more in class-2 / lr / lrc instances (question "why do class 2 instances suffer more from pruning?").
Each "Request pruner: kept K/N requests (... M pruned)" line is attributed to the instance of the most recent
"Loading input data from .../<stem>.json" line above it (same attribution as the log order of the training loop:
optimal_val pre-pass, then per epoch train files, then val files). Rates are pooled over everything the log
contains (all epochs/phases); rate = pruned / seen requests (same definition as compute_pruning_rates.py).
Output: per-instance table and a class x geometry summary for the 12 test/val instances.

Usage: ./venv/bin/python3 scripts/analysis/compute_pruning_rate_per_instance.py <log> [<log> ...]
       (several logs = pooled, e.g. all seeds of one config)
"""
import re
import sys
from collections import defaultdict

LOAD_RE = re.compile(r"Loading input data from \S*/([a-z]+\d+)\.json")
REQ_RE = re.compile(r"Request pruner: kept (\d+)/(\d+) requests \((\d+) force-kept as active, (\d+) force-kept as urgent, (\d+) pruned\)")
TEST = ["lc108", "lc109", "lc207", "lc208", "lr111", "lr112", "lr210", "lr211", "lrc107", "lrc108", "lrc207", "lrc208"]


def parse(paths):
    seen = defaultdict(int)
    pruned = defaultdict(int)
    for p in paths:
        cur = None
        with open(p, errors="ignore") as f:
            for line in f:
                m = LOAD_RE.search(line)
                if m:
                    cur = m.group(1)
                    continue
                m = REQ_RE.search(line)
                if m and cur:
                    seen[cur] += int(m.group(2))
                    pruned[cur] += int(m.group(5))
    return seen, pruned


def cls_geo(stem):
    geo = re.match(r"[a-z]+", stem).group()
    return stem[len(geo)], geo


def main(paths):
    seen, pruned = parse(paths)
    print(f"logs: {len(paths)}")
    print("instance   seen  pruned  rate")
    for s in sorted(seen):
        print(f"{s:9s} {seen[s]:6d} {pruned[s]:6d} {100 * pruned[s] / seen[s]:6.1f}%")
    print("\nTEST instances, by class x geometry (pooled):")
    agg = defaultdict(lambda: [0, 0])
    for s in TEST:
        if s in seen:
            c, g = cls_geo(s)
            agg[(c, g)][0] += seen[s]
            agg[(c, g)][1] += pruned[s]
    for (c, g), (n, k) in sorted(agg.items()):
        print(f"class{c} {g:3s} seen={n:6d} pruned={k:6d} rate={100 * k / n:5.1f}%")
    for c in ("1", "2"):
        n = sum(v[0] for (cc, _), v in agg.items() if cc == c)
        k = sum(v[1] for (cc, _), v in agg.items() if cc == c)
        if n:
            print(f"class{c} ALL      seen={n:6d} pruned={k:6d} rate={100 * k / n:5.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
