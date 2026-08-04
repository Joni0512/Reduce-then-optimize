"""Builds a run index (CSV + Markdown) for all NYC single-payload COAML/SIL
runs under outputs/new_tests/ - the runs whose INPUT_FILE points at a
nyc_real* manifest and whose MODE is "coaml" (i.e. the SIL train/eval runs
discussed in the granularity/LR comparisons, not the Li&Lim batch ablations).

Re-run any time after a new SIL run finishes to refresh the index; it always
re-scans from scratch, never edits run output in place.
"""
import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOT = REPO_ROOT / "outputs"
OUT_CSV = REPO_ROOT / "outputs" / "run_index.csv"
OUT_MD = REPO_ROOT / "outputs" / "run_index.md"


def find_config_files():
    result = subprocess.run(
        ["find", str(SEARCH_ROOT), "-name", "config.json", "-not", "-path", "*/outputs/outputs/*"],
        capture_output=True, text=True,
    )
    paths = [Path(p) for p in result.stdout.splitlines()]
    keep = []
    for p in paths:
        try:
            text = p.read_text()
        except OSError:
            continue
        if '"MODE": "coaml"' in text and "nyc_real" in text:
            keep.append(p)
    return sorted(keep)


def best_val_from_log(run_dir: Path):
    log = run_dir / "main.log"
    if not log.exists():
        return None, None
    m = None
    for line in log.read_text(errors="ignore").splitlines():
        match = re.search(r"Best val service rate = ([\d.]+) at epoch (\d+)", line)
        if match:
            m = match
    if not m:
        return None, None
    return float(m.group(1)) * 100, int(m.group(2))


def eval_service_rate(run_dir: Path):
    results_path = run_dir / "final" / "results.json"
    if not results_path.exists():
        return None, None, None
    try:
        stats = json.loads(results_path.read_text())["stats"]
    except (KeyError, json.JSONDecodeError):
        return None, None, None
    serviced = stats.get("serviced")
    total = stats.get("total_requests")
    if serviced is None or total in (None, 0):
        return None, None, None
    return serviced, total, serviced / total * 100


def manifest_label(path_str: str):
    """Extract a short day label from a manifest filename, e.g.
    nyc_real1000_20160112_0614_train_v50_expert_rho28.json -> '0112 train'."""
    if not path_str:
        return ""
    name = Path(path_str).stem
    date_m = re.search(r"_(\d{8})_", name)
    role_m = re.search(r"_(train|val|test)_", name)
    date = date_m.group(1)[4:] if date_m else "?"
    role = role_m.group(1) if role_m else "?"
    return f"{date} {role}"


def granularity_label(step_size, batch_interval):
    return f"{step_size // 60}/{batch_interval // 60} min (ss{step_size}/bi{batch_interval})"


LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def runtime_seconds(run_dir: Path):
    """Wall-clock runtime = last main.log timestamp minus first, in seconds.
    None if main.log is missing or has fewer than two timestamped lines
    (e.g. the run crashed before logging anything, or is still in progress)."""
    log = run_dir / "main.log"
    if not log.exists():
        return None
    first = last = None
    with open(log, errors="ignore") as f:
        for line in f:
            m = LOG_TS_RE.match(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            if first is None:
                first = ts
            last = ts
    if first is None or last is None or first == last:
        return None
    return round((last - first).total_seconds())


def format_runtime(seconds):
    if seconds is None:
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def build_row(config_path: Path):
    run_dir = config_path.parent
    cfg = json.loads(config_path.read_text())
    c = cfg["config_dict"]

    solve_mode = c.get("COAML_SOLVE_MODE", "")
    has_val = bool(c.get("VAL_INPUT_FILE"))
    checkpoint = c.get("COAML_MODEL_WEIGHTS", "")

    # For eval-mode runs, this config's own LEARNING_RATE is just main.py's
    # argparse default (--learning_rate was never passed to an eval call) -
    # it does NOT describe how the loaded checkpoint was trained. Resolve the
    # real value by reading the training run's own config.json instead.
    if solve_mode == "eval" and checkpoint:
        ckpt_train_cfg = REPO_ROOT / Path(checkpoint).parent / "config.json"
        resolved_lr = None
        if ckpt_train_cfg.exists():
            try:
                resolved_lr = json.loads(ckpt_train_cfg.read_text())["config_dict"].get("LEARNING_RATE")
            except (KeyError, json.JSONDecodeError):
                pass
        learning_rate_display = resolved_lr
    else:
        learning_rate_display = c.get("LEARNING_RATE")

    row = {
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "timestamp": cfg.get("timestamp", ""),
        "git_commit": cfg.get("git_commit", "")[:10],
        "solve_mode": solve_mode,
        "kind": "train+val" if (solve_mode == "train" and has_val) else solve_mode,
        "granularity": granularity_label(c["STEP_SIZE"], c["BATCH_INTERVAL"]),
        "max_cardinality": c.get("MAX_CARDINALITY"),
        "learning_rate": learning_rate_display,
        "epochs_configured": c.get("EPOCHS"),
        "seed": c.get("SEED"),
        "input_manifest": manifest_label(c.get("INPUT_FILE", "")),
        "val_manifest": manifest_label(c.get("VAL_INPUT_FILE", "")),
        "loaded_checkpoint": checkpoint,
        "runtime_seconds": runtime_seconds(run_dir),
        "best_val_service_rate_pct": None,
        "best_epoch": None,
        "eval_serviced": None,
        "eval_total": None,
        "eval_service_rate_pct": None,
    }

    if solve_mode == "train" and has_val:
        best_val, best_epoch = best_val_from_log(run_dir)
        row["best_val_service_rate_pct"] = round(best_val, 2) if best_val is not None else None
        row["best_epoch"] = best_epoch
    elif solve_mode == "eval":
        serviced, total, rate = eval_service_rate(run_dir)
        row["eval_serviced"] = serviced
        row["eval_total"] = total
        row["eval_service_rate_pct"] = round(rate, 2) if rate is not None else None

    return row


def main():
    config_files = find_config_files()
    rows = [build_row(p) for p in config_files]
    rows.sort(key=lambda r: r["timestamp"])

    fieldnames = list(rows[0].keys()) if rows else []
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {OUT_CSV} ({len(rows)} runs)")

    md_lines = [
        "# NYC SIL Run Index",
        "",
        "Auto-generated by `scripts/analysis/build_run_index.py` - re-run after new SIL runs, do not edit by hand.",
        "",
        "| Run dir | Timestamp | Mode | Granularity | LR | Seed | Manifest(s) | Best val | Eval result | Runtime |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["kind"] == "train+val":
            manifest = f"{r['input_manifest']} / {r['val_manifest']}"
            best = f"{r['best_val_service_rate_pct']}% (ep.{r['best_epoch']})" if r["best_val_service_rate_pct"] is not None else "?"
            eval_res = ""
        else:
            manifest = r["input_manifest"]
            best = ""
            eval_res = (
                f"{r['eval_serviced']}/{r['eval_total']} = {r['eval_service_rate_pct']}%"
                if r["eval_service_rate_pct"] is not None else "?"
            )
        md_lines.append(
            f"| `{r['run_dir']}` | {r['timestamp']} | {r['kind']} | {r['granularity']} | "
            f"{r['learning_rate']} | {r['seed']} | {manifest} | {best} | {eval_res} | {format_runtime(r['runtime_seconds'])} |"
        )
    OUT_MD.write_text("\n".join(md_lines) + "\n")
    print(f"saved: {OUT_MD}")


if __name__ == "__main__":
    main()
