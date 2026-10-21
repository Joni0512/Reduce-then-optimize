"""
2026-09-26: parse a SIL training log's per-batch "Request pruner: kept K/N requests
(... M pruned)" and "Allowed request pairs: K / N candidate pairs (edge_reduction=X)"
lines to compute the run's overall request- and pair-pruning rate (requests/pairs
pruned as a fraction of all requests/pairs seen across all batches, all epochs).

Usage: ./venv/bin/python3 scripts/analysis/compute_pruning_rates.py <log_file> [<log_file> ...]
"""
import re
import sys

REQ_RE = re.compile(r"Request pruner: kept (\d+)/(\d+) requests \((\d+) force-kept as active, (\d+) force-kept as urgent, (\d+) pruned\)")
PAIR_RE = re.compile(r"Allowed request pairs: (\d+) / (\d+) candidate pairs \(edge_reduction=([\d.]+)\)")


def analyze(path):
    req_total = req_pruned = 0
    pair_total = pair_allowed = 0
    n_req_batches = n_pair_batches = 0
    with open(path, errors="ignore") as f:
        for line in f:
            m = REQ_RE.search(line)
            if m:
                kept, total, _, _, pruned = map(int, m.groups())
                req_total += total
                req_pruned += pruned
                n_req_batches += 1
                continue
            m = PAIR_RE.search(line)
            if m:
                allowed, total, _ = m.groups()
                pair_total += int(total)
                pair_allowed += int(allowed)
                n_pair_batches += 1

    print(f"=== {path} ===")
    if n_req_batches:
        print(f"  Request pruner: {n_req_batches} batches, {req_total} requests seen, "
              f"{req_pruned} pruned -> {100*req_pruned/req_total:.2f}% pruning rate")
    else:
        print("  Request pruner: no matching lines (pruner disabled or different log)")
    if n_pair_batches:
        pruned_pairs = pair_total - pair_allowed
        print(f"  Pair pruner: {n_pair_batches} batches, {pair_total} candidate pairs, "
              f"{pair_allowed} allowed, {pruned_pairs} pruned -> {100*pruned_pairs/pair_total:.2f}% pruning rate")
    else:
        print("  Pair pruner: no matching lines (pruner disabled or different log)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
