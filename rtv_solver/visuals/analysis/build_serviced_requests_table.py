"""
Build a CSV of serviced/total counts, vmt, and pmt from experiment results.json files.

Path convention (under --root): .../mc<max_cardinality>_bi<batch_interval>_.../.../<file_name>/results.json
<file_name> is the parent directory of results.json (e.g. lc108, lr111).

Each result is one CSV row. Default path filter 'optimal_val' yields 36 rows for the usual 6 (mc,bi) × 6 scenario layout.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


MC_BI_RE = re.compile(r"mc(\d+)_bi(\d+)")


def parse_mc_bi_from_path(path: Path) -> tuple[int | None, int | None]:
    for part in path.parts:
        m = MC_BI_RE.search(part)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def _stat_float(stats: dict, key: str) -> float | None:
    v = stats.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def load_row_metrics(results_path: Path) -> tuple[int | None, int | None, float | None, float | None]:
    """Return (serviced_count, total_requests, vmt, pmt). serviced_count None means the row is invalid."""
    try:
        with results_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None, None, None
    stats = data.get("stats") if isinstance(data, dict) else None
    if not isinstance(stats, dict):
        return None, None, None, None
    serviced: int | None = None
    n = stats.get("serviced")
    if isinstance(n, int) and not isinstance(n, bool):
        serviced = n
    else:
        raw = stats.get("serviced_requests")
        if isinstance(raw, list):
            serviced = len(raw)
    total: int | None = None
    t = stats.get("total_requests")
    if isinstance(t, int) and not isinstance(t, bool):
        total = t
    vmt = _stat_float(stats, "vmt")
    pmt = _stat_float(stats, "pmt")
    if serviced is None:
        return None, None, None, None
    return serviced, total, vmt, pmt


def find_results_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob("results.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/experiment_structure"),
        help="Directory to search recursively for results.json (default: outputs/experiment_structure).",
    )
    parser.add_argument(
        "--path-must-contain",
        default="optimal_val",
        metavar="SUBSTRING",
        help="Only include files whose path contains this substring. "
        "Use empty string to disable filtering (default: optimal_val).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs/optimalRH_serviced_counts.csv"),
        help="CSV output path (default: outputs/optimalRH_serviced_counts.csv). Use - for stdout.",
    )
    parser.add_argument(
        "--include-results-path",
        action="store_true",
        help="Add a column with the path to results.json.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    files = find_results_files(root)
    needle = args.path_must_contain
    if needle:
        files = [p for p in files if needle in str(p)]

    rows: list[dict[str, str | int | float | None]] = []
    for results_path in files:
        mc, bi = parse_mc_bi_from_path(results_path)
        file_name = results_path.parent.name
        serviced, total_req, vmt, pmt = load_row_metrics(results_path)
        if serviced is None:
            row: dict[str, str | int | float | None] = {
                "file_name": file_name,
                "max_cardinality": mc,
                "batch_interval": bi,
                "serviced_count": "",
                "total_requests": "",
                "vmt": "",
                "pmt": "",
                "error": "missing_or_invalid_json",
            }
        else:
            row = {
                "file_name": file_name,
                "max_cardinality": mc,
                "batch_interval": bi,
                "serviced_count": serviced,
                "total_requests": total_req if total_req is not None else "",
                "vmt": vmt if vmt is not None else "",
                "pmt": pmt if pmt is not None else "",
                "error": "",
            }
        if args.include_results_path:
            row["results_path"] = str(results_path.relative_to(root)) if results_path.is_relative_to(root) else str(results_path)
        rows.append(row)

    fieldnames = [
        "file_name",
        "max_cardinality",
        "batch_interval",
        "serviced_count",
        "total_requests",
        "vmt",
        "pmt",
    ]
    if rows and any(r.get("error") for r in rows):
        fieldnames.append("error")
    if args.include_results_path:
        fieldnames.append("results_path")

    use_stdout = str(args.output) == "-"
    out_f = sys.stdout if use_stdout else open(args.output.expanduser(), "w", encoding="utf-8", newline="")
    try:
        w = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    finally:
        if not use_stdout:
            out_f.close()

    if not use_stdout:
        print(f"Wrote {len(rows)} rows to {args.output.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
