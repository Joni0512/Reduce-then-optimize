"""
Aggregate results.json from experiment outputs into one table (CSV or Excel).

Layouts:
  offline — outputs/experiments/<run>/<instance>/results.json
  coaml  — outputs/experiments_coaml/<run>/val/epoch_*/<instance>/results.json
           uses the last epoch folder under val (by numeric index: epoch_4 > epoch_1).

Missing results.json: all numeric metric columns are set to -10.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

MISSING_NUMERIC = -10


def _is_real_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def flatten_numeric_leaves(obj: Any, prefix: str = "") -> dict[str, float | int]:
    """Recursively collect int/float leaves from dicts; skip lists and bools."""
    out: dict[str, float | int] = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        name = f"{prefix}__{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_numeric_leaves(v, name))
        elif isinstance(v, list):
            continue
        elif _is_real_number(v):
            out[name] = v
    return out


def load_results_row(results_path: Path) -> tuple[bool, dict[str, Any]]:
    """Parse results.json; return (ok, payload). On failure ok is False."""
    try:
        with results_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, {}
    stats = data.get("stats") if isinstance(data, dict) else None
    metrics: dict[str, Any] = {}
    if isinstance(stats, dict):
        metrics.update(flatten_numeric_leaves(stats))
    if isinstance(data, dict) and isinstance(data.get("violations"), list):
        metrics["violations_count"] = len(data["violations"])
    return True, metrics


def _epoch_dir_sort_key(p: Path) -> int:
    name = p.name
    if not name.startswith("epoch_"):
        return -1
    try:
        return int(name[len("epoch_") :])
    except ValueError:
        return -1


def coaml_last_val_epoch_dir(run_dir: Path) -> Path | None:
    """Highest epoch_* under run_dir/val, or None if missing/empty."""
    val_dir = run_dir / "val"
    if not val_dir.is_dir():
        return None
    epoch_dirs = [p for p in val_dir.iterdir() if p.is_dir() and p.name.startswith("epoch_")]
    if not epoch_dirs:
        return None
    return max(epoch_dirs, key=_epoch_dir_sort_key)


def iter_slots_offline(experiments_root: Path) -> list[tuple[str, str, str, str, Path]]:
    """
    Returns (experiment_folder, split, epoch_folder, instance_folder, results_path).
    split and epoch_folder are empty for offline.
    """
    slots: list[tuple[str, str, str, str, Path]] = []
    if not experiments_root.is_dir():
        return slots
    for run_dir in sorted(experiments_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for inst_dir in sorted(run_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            slots.append(
                (
                    run_dir.name,
                    "",
                    "",
                    inst_dir.name,
                    inst_dir / "results.json",
                )
            )
    return slots


def iter_slots_coaml(experiments_root: Path) -> list[tuple[str, str, str, str, Path]]:
    """
    Per run folder: use last val/epoch_* (by numeric index), then each instance subdir.
    """
    slots: list[tuple[str, str, str, str, Path]] = []
    if not experiments_root.is_dir():
        return slots
    for run_dir in sorted(experiments_root.iterdir()):
        if not run_dir.is_dir():
            continue
        epoch_dir = coaml_last_val_epoch_dir(run_dir)
        if epoch_dir is None:
            continue
        for inst_dir in sorted(epoch_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            slots.append(
                (
                    run_dir.name,
                    "val",
                    epoch_dir.name,
                    inst_dir.name,
                    inst_dir / "results.json",
                )
            )
    return slots


def iter_result_slots(
    experiments_root: Path, layout: str
) -> list[tuple[str, str, str, str, Path]]:
    if layout == "coaml":
        return iter_slots_coaml(experiments_root)
    if layout == "offline":
        return iter_slots_offline(experiments_root)
    raise ValueError(f"Unknown layout: {layout}")


def build_rows(
    experiments_root: Path, layout: str
) -> tuple[list[str], list[dict[str, Any]]]:
    slots = iter_result_slots(experiments_root, layout)
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    all_metric_keys: set[str] = set()

    for run_name, split, epoch_name, inst_name, results_path in slots:
        ok, metrics = load_results_row(results_path)
        if ok:
            all_metric_keys.update(metrics.keys())
        row_meta: dict[str, Any] = {
            "experiment_folder": run_name,
            "split": split,
            "epoch_folder": epoch_name,
            "instance_folder": inst_name,
            "results_path": str(results_path.resolve()),
            "results_found": ok,
        }
        pending.append((row_meta, metrics))

    meta_keys = [
        "experiment_folder",
        "split",
        "epoch_folder",
        "instance_folder",
        "results_path",
        "results_found",
    ]
    sorted_metrics = sorted(all_metric_keys)
    header = meta_keys + sorted_metrics

    out_rows: list[dict[str, Any]] = []
    for row_meta, metrics in pending:
        row = dict(row_meta)
        if not row_meta["results_found"]:
            for k in sorted_metrics:
                row[k] = MISSING_NUMERIC
        else:
            for k in sorted_metrics:
                row[k] = metrics[k] if k in metrics else math.nan
        out_rows.append(row)

    return header, out_rows


def _cell_value(v: Any) -> Any:
    if isinstance(v, float) and math.isnan(v):
        return ""
    return v


def _write_csv(path: Path, header: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _cell_value(row.get(k, "")) for k in header})


def _write_xlsx(path: Path, header: list[str], rows: Iterable[dict[str, Any]]) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as e:
        raise ImportError(
            "Writing .xlsx requires openpyxl. Install with: pip install openpyxl"
        ) from e
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.append(header)
    for row in rows:
        ws.append([_cell_value(row.get(k, "")) for k in header])
    wb.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results.json files into CSV or Excel."
    )
    parser.add_argument(
        "--layout",
        choices=("offline", "coaml"),
        default="coaml",
        help="offline: run/instance; coaml: run/val/last epoch/instance (default: offline)",
    )
    parser.add_argument(
        "experiments_root",
        type=Path,
        nargs="?",
        default=None,
        help="Root directory of runs (default: outputs/experiments or outputs/experiments_coaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file (.csv or .xlsx). Default depends on --layout",
    )
    args = parser.parse_args(argv)

    if args.experiments_root is None:
        root = (
            Path("outputs/experiments_coaml")
            if args.layout == "coaml"
            else Path("outputs/experiments")
        )
    else:
        root = args.experiments_root
    root = root.expanduser().resolve()

    if args.output is None:
        out_path = (
            Path("outputs/experiments_coaml_aggregated.csv")
            if args.layout == "coaml"
            else Path("outputs/experiments_aggregated.csv")
        )
    else:
        out_path = args.output

    if not root.is_dir():
        print(f"Experiments root is not a directory: {root}", file=sys.stderr)
        return 1

    header, rows = build_rows(root, args.layout)
    if not rows:
        print(f"No matching folders found under {root} ( layout={args.layout})", file=sys.stderr)
        return 1

    out_path = out_path.expanduser()
    suffix = out_path.suffix.lower()
    try:
        if suffix == ".xlsx":
            _write_xlsx(out_path, header, rows)
        else:
            if suffix != ".csv":
                out_path = out_path.with_suffix(".csv")
            _write_csv(out_path, header, rows)
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} rows to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
