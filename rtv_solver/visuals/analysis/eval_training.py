"""
Evaluation infrastructure for training analysis.

Handles loading and comparing data from three sources:
1. Solutions (absolute benchmark) - optimal scores per instance
2. Offline benchmark - per-instance results from offline solver runs
3. CoAML benchmark - training outputs and assignment data from COAML runs

Usage:
    python rtv_solver/visuals/analysis/eval_training.py \\
        --solutions solutions/li_lim \\
        --offline outputs/experiments/run_offline_mc3_bi400_ss100 \\
        --coaml outputs/experiments/batch_lilim_coaml_seed42/mc3_bi400_ss100_20260317_231957 \\
        --results results
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Figure signatures and output

# Central place for all analysis output base names (values + plot share same stem).
# Each entry: base name and show_plots (whether to display interactively; can be turned off via --no-show).
FIGURE_SIGNATURES = {
    "coaml_avg_loss_per_epoch": {
        "base": "coaml_avg_loss_per_epoch",
        "show": True,
    },
    "coaml_avg_loss_per_file_per_epoch": {
        "base": "coaml_avg_loss_per_file_per_epoch",
        "show": True,
    },
    "serviced_per_file_comparison": {
        "base": "serviced_per_file_comparison",
        "show": True,
    },
    "vmt_per_file_comparison": {
        "base": "vmt_per_file_comparison",
        "show": True,
    },
    "distance_per_request_comparison": {
        "base": "distance_per_request_comparison",
        "show": True,
    },
}

# LiLim file split (must match training_loop.py)
TRAINING_FILES = [
    "lc101", "lc102", "lc103", "lc104", "lc105", "lc106", "lc107",
    "lc201", "lc202", "lc203", "lc204", "lc205", "lc206",
    "lr101", "lr102", "lr103", "lr104", "lr105", "lr106", "lr107", "lr108", "lr109", "lr110",
    "lr201", "lr202", "lr203", "lr204", "lr205", "lr206", "lr207", "lr208", "lr209",
    "lrc101", "lrc102", "lrc103", "lrc104", "lrc105", "lrc106",
    "lrc201", "lrc202", "lrc203", "lrc204", "lrc205", "lrc206",
]
VALIDATION_FILES = [
    "lc108", "lc109", "lc207", "lc208",
    "lr111", "lr112", "lr210", "lr211",
    "lrc107", "lrc108", "lrc207", "lrc208",
]
VAL_STEMS = set(VALIDATION_FILES)
ALL_LILIM_FILES = sorted(set(TRAINING_FILES) | VAL_STEMS)


def get_figure_paths(results_dir: Path, signature: str) -> tuple[Path, Path]:
    """Return (values_json_path, plot_pdf_path) for a given figure signature."""
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    return (
        results_dir / f"{signature}.json",
        results_dir / f"{signature}.pdf",
    )


def apply_plot_defaults(ax: plt.Axes, ncol: int = 4) -> None:
    """Apply default styling: legend below, x/y labels and axes enabled."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=False)
    ax.set_xlabel(ax.get_xlabel() or "x")
    ax.set_ylabel(ax.get_ylabel() or "y")


def save_figure_and_values(
    results_dir: Path,
    signature: str,
    values: Dict[str, Any],
    fig: plt.Figure,
    show: bool = False,
) -> None:
    """Save both the numeric values (JSON) and the plot (PDF) with the same base name.
    If show=True, display the figure interactively before closing."""
    values_path, plot_path = get_figure_paths(results_dir, signature)
    with open(values_path, "w") as f:
        json.dump(values, f, indent=2)
    fig.savefig(plot_path, bbox_inches="tight", format="pdf")
    if show:
        plt.show()
    plt.close(fig)


# Data structures

@dataclass
class SolutionsData:
    """Absolute benchmark: optimal scores per instance (e.g. liLim_solution_scores.json)."""

    scores: Dict[str, float]  # instance_id -> optimal VMT/score
    manifests_dir: Path  # path to manifests/*.json
    txt_dir: Optional[Path] = None  # path to txt_files/*.txt

    def instance_ids(self) -> List[str]:
        return list(self.scores.keys())


@dataclass
class OfflineRunData:
    """Single offline run for one instance."""

    instance_id: str
    run_dir: Path
    results: Dict[str, Any]  # from results.json
    config: Dict[str, Any]  # from config.json
    assignment_data_path: Path  # assignment_data.jsonl


@dataclass
class OfflineData:
    """Offline benchmark: per-instance results from offline solver runs."""

    runs: Dict[str, List[OfflineRunData]]  # instance_id -> list of runs (may have multiple timestamps)
    base_dir: Path

    def instance_ids(self) -> List[str]:
        return list(self.runs.keys())

    def latest_run_per_instance(self) -> Dict[str, OfflineRunData]:
        """Return the most recent run for each instance."""
        out = {}
        for inst_id, run_list in self.runs.items():
            if run_list:
                # Sort by run_dir name (contains timestamp) and take latest
                sorted_runs = sorted(run_list, key=lambda r: r.run_dir.name, reverse=True)
                out[inst_id] = sorted_runs[0]
        return out


@dataclass
class CoAMLData:
    """CoAML benchmark: training outputs and assignment data."""

    base_dir: Path
    config: Dict[str, Any]
    training_loss_per_file: Dict[str, List[float]]  # instance_id -> loss per iteration
    assignment_data_path: Path  # assignment_data.jsonl (from last training run)
    model_weights_path: Optional[Path] = None

    def instance_ids(self) -> List[str]:
        return list(self.training_loss_per_file.keys())



# Loaders

def load_solutions(solutions_dir: Path) -> SolutionsData:
    """
    Load absolute benchmark data from solutions folder.

    Expected structure:
        solutions_dir/
            liLim_solution_scores.json   (or similar *_solution_scores.json)
            manifests/*.json
            txt_files/*.txt (optional)
    """
    solutions_dir = Path(solutions_dir).resolve()
    if not solutions_dir.is_dir():
        raise FileNotFoundError(f"Solutions dir not found: {solutions_dir}")

    # Find solution scores file (support multiple naming conventions)
    scores_file = None
    for name in ["liLim_solution_scores.json", "solution_scores.json", "*_solution_scores.json"]:
        matches = list(solutions_dir.glob(name))
        if matches:
            scores_file = matches[0]
            break
    if scores_file is None:
        raise FileNotFoundError(f"No *_solution_scores.json found in {solutions_dir}")

    with open(scores_file) as f:
        scores = json.load(f)

    manifests_dir = solutions_dir / "manifests"
    txt_dir = solutions_dir / "txt_files"
    if not manifests_dir.is_dir():
        manifests_dir = solutions_dir  # fallback

    return SolutionsData(
        scores=scores,
        manifests_dir=manifests_dir,
        txt_dir=txt_dir if txt_dir.is_dir() else None,
    )


def load_offline(offline_dir: Path) -> OfflineData:
    """
    Load offline benchmark data.

    Expected structure:
        offline_dir/
            {instance_id}_{timestamp}/   (e.g. lc108_20260316_111427)
                results.json
                config.json
                assignment_data.jsonl
                result_driver_runs.json
    """
    offline_dir = Path(offline_dir).resolve()
    if not offline_dir.is_dir():
        raise FileNotFoundError(f"Offline dir not found: {offline_dir}")

    runs: Dict[str, List[OfflineRunData]] = {}

    for run_path in offline_dir.iterdir():
        if not run_path.is_dir():
            continue
        results_file = run_path / "results.json"
        config_file = run_path / "config.json"
        assignment_file = run_path / "assignment_data.jsonl"

        if not results_file.exists() or not config_file.exists():
            continue

        # Parse instance_id from folder name (e.g. lc108_20260316_111427 -> lc108)
        name = run_path.name
        parts = name.split("_")
        if (
            len(parts) >= 3
            and parts[-1].isdigit()
            and len(parts[-1]) == 6  # HHMMSS
            and parts[-2].isdigit()
            and len(parts[-2]) == 8  # YYYYMMDD
        ):
            instance_id = "_".join(parts[:-2])
        elif len(parts) >= 2 and parts[-1].isdigit():
            instance_id = "_".join(parts[:-1])
        else:
            instance_id = name

        with open(results_file) as f:
            results = json.load(f)
        with open(config_file) as f:
            config = json.load(f)

        run_data = OfflineRunData(
            instance_id=instance_id,
            run_dir=run_path,
            results=results,
            config=config,
            assignment_data_path=assignment_file if assignment_file.exists() else Path(),
        )

        if instance_id not in runs:
            runs[instance_id] = []
        runs[instance_id].append(run_data)

    return OfflineData(runs=runs, base_dir=offline_dir)


def load_coaml(coaml_dir: Path) -> CoAMLData:
    """
    Load CoAML benchmark data.

    Expected structure:
        coaml_dir/
            config.json
            training_loss_per_file.json
            assignment_data.jsonl
            coaml_model_weights.pt (optional)
    """
    coaml_dir = Path(coaml_dir).resolve()
    if not coaml_dir.is_dir():
        raise FileNotFoundError(f"CoAML dir not found: {coaml_dir}")

    config_file = coaml_dir / "config.json"
    loss_file = coaml_dir / "training_loss_per_file.json"
    assignment_file = coaml_dir / "assignment_data.jsonl"
    weights_file = coaml_dir / "coaml_model_weights.pt"

    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    training_loss_per_file: Dict[str, List[float]] = {}
    if loss_file.exists():
        with open(loss_file) as f:
            training_loss_per_file = json.load(f)

    return CoAMLData(
        base_dir=coaml_dir,
        config=config,
        training_loss_per_file=training_loss_per_file,
        assignment_data_path=assignment_file if assignment_file.exists() else Path(),
        model_weights_path=weights_file if weights_file.exists() else None,
    )


# Analysis helpers (outline)


def load_assignment_data(path: Path) -> List[Dict[str, Any]]:
    """Load assignment_data.jsonl into a list of records."""
    if not path or not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def align_instances(
    solutions: SolutionsData,
    offline: Optional[OfflineData],
    coaml: Optional[CoAMLData],
) -> List[str]:
    """
    Return instance IDs present in all provided data sources. Use this to ensure fair comparison across benchmarks.
    """
    ids = set(solutions.instance_ids())
    if offline:
        ids &= set(offline.instance_ids())
    if coaml:
        ids &= set(coaml.instance_ids())
    return sorted(ids)


# Analyses


def _get_epochs_from_config(config: Dict[str, Any]) -> int:
    """Extract EPOCHS from config (supports nested config_dict)."""
    cfg = config.get("config_dict", config)
    return int(cfg.get("EPOCHS", 1))


def _split_into_epoch_chunks(
    losses: List[float], epochs: int
) -> List[List[float]]:
    """Split a flat loss list into EPOCHS chunks (as equal as possible)."""
    n = len(losses)
    if n == 0 or epochs <= 0:
        return [[] for _ in range(epochs)]
    chunk_size = n // epochs
    remainder = n % epochs
    chunks = []
    idx = 0
    for i in range(epochs):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(losses[idx : idx + size])
        idx += size
    return chunks


def analyze_coaml_avg_loss_per_epoch(
    coaml: CoAMLData, results_dir: Path
) -> None:
    """
    Average loss for all files and iterations across epochs (COAML training).
    X-axis: epoch (1..EPOCHS), Y-axis: mean loss across all files and iterations in that epoch.
    """
    cfg = FIGURE_SIGNATURES["coaml_avg_loss_per_epoch"]
    sig = cfg["base"]
    show = cfg["show"]
    epochs = _get_epochs_from_config(coaml.config)
    training_loss = coaml.training_loss_per_file

    if not training_loss:
        return

    # For each epoch, collect all losses from that epoch across all files
    epoch_losses: List[List[float]] = [[] for _ in range(epochs)]
    for instance_id, losses in training_loss.items():
        valid = [l for l in losses if l is not None]
        if not valid:
            continue
        chunks = _split_into_epoch_chunks(valid, epochs)
        for e, chunk in enumerate(chunks):
            epoch_losses[e].extend(chunk)

    # Per-epoch mean
    avg_per_epoch = [
        sum(epoch_losses[e]) / len(epoch_losses[e]) if epoch_losses[e] else 0.0
        for e in range(epochs)
    ]

    values = {
        "epochs": epochs,
        "avg_loss_per_epoch": avg_per_epoch,
        "epoch_labels": list(range(1, epochs + 1)),
        "num_files": len(training_loss),
        "total_iterations": sum(len([l for l in L if l is not None]) for L in training_loss.values()),
    }

    fig, ax = plt.subplots()
    ax.plot(
        values["epoch_labels"],
        avg_per_epoch,
        marker="o",
        linestyle="-",
        label="Avg loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average loss")
    ax.set_title("COAML training: average loss per epoch (all files, all iterations)")
    ax.grid(True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    apply_plot_defaults(ax)
    fig.tight_layout()
    save_figure_and_values(results_dir, sig, values, fig, show=show)


def analyze_coaml_avg_loss_per_file_per_epoch(
    coaml: CoAMLData, results_dir: Path
) -> None:
    """
    Average loss per file across epochs (COAML training).
    Each file is a separate line with distinct color; x-axis: epoch, y-axis: mean loss.
    """
    cfg = FIGURE_SIGNATURES["coaml_avg_loss_per_file_per_epoch"]
    sig = cfg["base"]
    show = cfg["show"]
    epochs = _get_epochs_from_config(coaml.config)
    training_loss = coaml.training_loss_per_file

    if not training_loss:
        return

    # Per file: average loss per epoch
    file_ids = sorted(training_loss.keys())
    avg_per_file_per_epoch: Dict[str, List[float]] = {}

    for instance_id in file_ids:
        losses = training_loss[instance_id]
        valid = [l for l in losses if l is not None]
        if not valid:
            continue
        chunks = _split_into_epoch_chunks(valid, epochs)
        avg_per_file_per_epoch[instance_id] = [
            sum(chunks[e]) / len(chunks[e]) if chunks[e] else 0.0
            for e in range(epochs)
        ]

    if not avg_per_file_per_epoch:
        return

    values = {
        "epochs": epochs,
        "epoch_labels": list(range(1, epochs + 1)),
        "avg_loss_per_file_per_epoch": avg_per_file_per_epoch,
        "file_ids": list(avg_per_file_per_epoch.keys()),
    }

    fig, ax = plt.subplots()
    epoch_labels = values["epoch_labels"]
    n_files = len(avg_per_file_per_epoch)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_files, 1)))

    for i, (instance_id, avg_per_epoch) in enumerate(avg_per_file_per_epoch.items()):
        color = colors[i % len(colors)]
        ax.plot(
            epoch_labels,
            avg_per_epoch,
            marker="o",
            linestyle="-",
            label=instance_id,
            color=color,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average loss")
    ax.set_title("COAML training: average loss per file across epochs")
    ax.grid(True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    apply_plot_defaults(ax, ncol=min(6, n_files))
    fig.tight_layout()
    save_figure_and_values(results_dir, sig, values, fig, show=show)


def _load_optimal_serviced(manifests_dir: Path, file_id: str) -> Optional[int]:
    """Optimal serves all requests; return len(requests) from manifest."""
    path = manifests_dir / f"{file_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return len(data.get("requests", []))


def _load_stat_from_results(path: Path, stat: str) -> Optional[float]:
    """Load a numeric stat from results.json. Returns int or float."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    val = data.get("stats", {}).get(stat)
    return float(val) if val is not None else None


def _load_offline_stat(offline: OfflineData, file_id: str, stat: str) -> Optional[float]:
    """Load stat from offline run results.json."""
    runs = offline.runs.get(file_id, [])
    if not runs:
        return None
    latest = sorted(runs, key=lambda r: r.run_dir.name, reverse=True)[0]
    return _load_stat_from_results(latest.run_dir / "results.json", stat)


def _load_coaml_stat_per_epoch(
    coaml_dir: Path, file_id: str, epochs: int, is_validation: bool, stat: str
) -> Dict[int, float]:
    """Load stat per epoch from val/ or train_val/."""
    subdir = "val" if is_validation else "train_val"
    out: Dict[int, float] = {}
    for e in range(epochs):
        results_path = coaml_dir / subdir / f"epoch_{e}" / file_id / "results.json"
        val = _load_stat_from_results(results_path, stat)
        if val is not None:
            out[e + 1] = val
    return out


def _get_coaml_run_file_ids(coaml_dir: Path, epochs: int) -> List[str]:
    """Return file IDs that were actually run in COAML (have val or train_val results)."""
    seen: set[str] = set()
    for subdir in ("val", "train_val"):
        base = coaml_dir / subdir
        if not base.exists():
            continue
        for e in range(epochs):
            epoch_dir = base / f"epoch_{e}"
            if epoch_dir.exists():
                for p in epoch_dir.iterdir():
                    if p.is_dir() and (p / "results.json").exists():
                        seen.add(p.name)
    return sorted(seen)


def analyze_serviced_per_file_comparison(
    solutions: SolutionsData,
    offline: Optional[OfflineData],
    coaml: CoAMLData,
    results_dir: Path,
) -> None:
    """
    Bar chart: per LiLim file, serviced requests for optimal, offline, and each COAML validation epoch.
    One subplot per file; bars: optimal, offline, COAML epoch 1..N.
    Only includes files that were actually run in the COAML runtime.
    """
    cfg = FIGURE_SIGNATURES["serviced_per_file_comparison"]
    sig = cfg["base"]
    show = cfg["show"]
    epochs = _get_epochs_from_config(coaml.config)
    coaml_dir = coaml.base_dir

    # Only files that were run in COAML
    file_ids = _get_coaml_run_file_ids(coaml_dir, epochs)
    file_ids = [f for f in file_ids if (solutions.manifests_dir / f"{f}.json").exists()]
    if not file_ids:
        return

    bar_labels = ["Offline"] + [f"COAML ep.{e}" for e in range(1, epochs + 1)]
    n_bars = len(bar_labels)

    data_per_file: Dict[str, Dict[str, Optional[int]]] = {}
    for file_id in file_ids:
        is_val = file_id in VAL_STEMS
        row: Dict[str, Optional[int]] = {}
        row["Optimal"] = _load_optimal_serviced(solutions.manifests_dir, file_id)
        row["Offline"] = _load_offline_stat(offline, file_id, "serviced") if offline else None
        coaml_serviced = _load_coaml_stat_per_epoch(coaml_dir, file_id, epochs, is_val, "serviced")
        for e in range(1, epochs + 1):
            row[f"COAML ep.{e}"] = int(v) if (v := coaml_serviced.get(e)) is not None else None
        data_per_file[file_id] = row

    values = {
        "file_ids": file_ids,
        "run_labels": ["Optimal"] + bar_labels,
        "data_per_file": {
            fid: {k: v for k, v in data.items()}
            for fid, data in data_per_file.items()
        },
    }

    n_files = len(file_ids)
    n_cols = 4
    n_rows = (n_files + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Colors: Offline=blue, COAML epochs=lighter orange (readable with dark text)
    bar_colors = ["#3498db"] + list(plt.cm.Oranges(np.linspace(0.25, 0.55, epochs)))
    optimal_line_color = "#1a5f1a"
    # Slim bars with very small gap between them
    bar_spacing = 0.5
    width = 0.42
    x = np.arange(n_bars) * bar_spacing

    for idx, file_id in enumerate(file_ids):
        row_idx, col_idx = idx // n_cols, idx % n_cols
        ax = axes[row_idx, col_idx]
        row = data_per_file[file_id]
        optimal_val = row.get("Optimal")
        vals = [row.get(lbl) for lbl in bar_labels]
        heights = [v if v is not None else 0 for v in vals]

        # Optimal: dashed horizontal line (not a bar)
        if optimal_val is not None and optimal_val > 0:
            ax.axhline(
                y=optimal_val,
                color=optimal_line_color,
                linestyle="--",
                linewidth=1.5,
                zorder=1,
            )

        # Bars: Offline + COAML epochs
        bars = ax.bar(x, heights, width, zorder=2)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            bar.set_color(bar_colors[i])
            if v is None:
                bar.set_hatch("//")
                bar.set_alpha(0.5)
        # Value labels inside bars (very small)
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    str(int(h)),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="#333333",
                )
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylabel("Serviced requests" if col_idx == 0 else "")
        # Validation file titles: bold red
        if file_id in VAL_STEMS:
            ax.set_title(file_id, fontweight="bold", color="#c0392b")
        else:
            ax.set_title(file_id)
        ax.set_ylim(bottom=0)

    for idx in range(n_files, n_rows * n_cols):
        row_idx, col_idx = idx // n_cols, idx % n_cols
        axes[row_idx, col_idx].set_visible(False)

    # Legend: Optimal as line, then bar patches
    legend_elements = [
        Line2D(
            [0], [0],
            color=optimal_line_color,
            linestyle="--",
            linewidth=2,
            label="Optimal",
        ),
    ] + [Patch(facecolor=bar_colors[i], label=bar_labels[i]) for i in range(n_bars)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements), frameon=False)
    fig.suptitle("Serviced requests per file: Optimal vs Offline vs COAML validation epochs")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_figure_and_values(results_dir, sig, values, fig, show=show)


def analyze_vmt_per_file_comparison(
    solutions: SolutionsData,
    offline: Optional[OfflineData],
    coaml: CoAMLData,
    results_dir: Path,
) -> None:
    """
    Bar chart: per file, VMT difference to optimal (VMT - optimal_VMT).
    Lower is better; optimal baseline at 0. Uses solutions.scores for optimal VMT.
    Subplot title includes optimal total distance.
    """
    cfg = FIGURE_SIGNATURES["vmt_per_file_comparison"]
    sig = cfg["base"]
    show = cfg["show"]
    epochs = _get_epochs_from_config(coaml.config)
    coaml_dir = coaml.base_dir

    file_ids = _get_coaml_run_file_ids(coaml_dir, epochs)
    file_ids = [f for f in file_ids if f in solutions.scores]
    if not file_ids:
        return

    bar_labels = ["Offline"] + [f"COAML ep.{e}" for e in range(1, epochs + 1)]
    n_bars = len(bar_labels)
    bar_colors = ["#3498db"] + list(plt.cm.Oranges(np.linspace(0.25, 0.55, epochs)))
    optimal_line_color = "#1a5f1a"
    bar_spacing = 0.5
    width = 0.42
    x = np.arange(n_bars) * bar_spacing

    data_per_file: Dict[str, Dict[str, Any]] = {}
    for file_id in file_ids:
        is_val = file_id in VAL_STEMS
        optimal_vmt = solutions.scores.get(file_id)
        if optimal_vmt is None:
            continue
        row: Dict[str, Any] = {"optimal_vmt": optimal_vmt}
        offline_vmt = _load_offline_stat(offline, file_id, "vmt") if offline else None
        coaml_vmt = _load_coaml_stat_per_epoch(coaml_dir, file_id, epochs, is_val, "vmt")
        row["Offline"] = (offline_vmt - optimal_vmt) if offline_vmt is not None else None
        for e in range(1, epochs + 1):
            v = coaml_vmt.get(e)
            row[f"COAML ep.{e}"] = (v - optimal_vmt) if v is not None else None
        data_per_file[file_id] = row

    values = {
        "file_ids": file_ids,
        "run_labels": ["Optimal"] + bar_labels,
        "data_per_file": {fid: {k: v for k, v in d.items()} for fid, d in data_per_file.items()},
    }

    n_files = len(file_ids)
    n_cols = 4
    n_rows = (n_files + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, file_id in enumerate(file_ids):
        row_idx, col_idx = idx // n_cols, idx % n_cols
        ax = axes[row_idx, col_idx]
        row = data_per_file[file_id]
        optimal_vmt = row["optimal_vmt"]
        vals = [row.get(lbl) for lbl in bar_labels]
        heights = [v if v is not None else 0 for v in vals]

        bars = ax.bar(x, heights, width, zorder=1)
        ax.axhline(y=0, color=optimal_line_color, linestyle="--", linewidth=1.5, zorder=2)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            bar.set_color(bar_colors[i])
            if v is None:
                bar.set_hatch("//")
                bar.set_alpha(0.5)
        for bar, h in zip(bars, heights):
            if h != 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{h:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="#333333",
                )
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylabel("VMT − optimal" if col_idx == 0 else "")
        if file_id in VAL_STEMS:
            ax.set_title(f"{file_id} (opt: {optimal_vmt:.0f})", fontweight="bold", color="#c0392b")
        else:
            ax.set_title(f"{file_id} (opt: {optimal_vmt:.0f})")

    for idx in range(n_files, n_rows * n_cols):
        row_idx, col_idx = idx // n_cols, idx % n_cols
        axes[row_idx, col_idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=optimal_line_color, linestyle="--", linewidth=2, label="Optimal"),
    ] + [Patch(facecolor=bar_colors[i], label=bar_labels[i]) for i in range(n_bars)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements), frameon=False)
    fig.suptitle("VMT difference to optimal per file (lower is better)")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_figure_and_values(results_dir, sig, values, fig, show=show)


def analyze_distance_per_request_comparison(
    solutions: SolutionsData,
    offline: Optional[OfflineData],
    coaml: CoAMLData,
    results_dir: Path,
) -> None:
    """
    Bar chart: per file, distance per served request (VMT / serviced).
    Coverage-efficiency tradeoff: lower is better.
    Optimal = VMT / total_requests; Offline/COAML = VMT / serviced.
    """
    cfg = FIGURE_SIGNATURES["distance_per_request_comparison"]
    sig = cfg["base"]
    show = cfg["show"]
    epochs = _get_epochs_from_config(coaml.config)
    coaml_dir = coaml.base_dir

    file_ids = _get_coaml_run_file_ids(coaml_dir, epochs)
    file_ids = [
        f for f in file_ids
        if f in solutions.scores and (solutions.manifests_dir / f"{f}.json").exists()
    ]
    if not file_ids:
        return

    bar_labels = ["Offline"] + [f"COAML ep.{e}" for e in range(1, epochs + 1)]
    n_bars = len(bar_labels)
    bar_colors = ["#3498db"] + list(plt.cm.Oranges(np.linspace(0.25, 0.55, epochs)))
    optimal_line_color = "#1a5f1a"
    bar_spacing = 0.5
    width = 0.42
    x = np.arange(n_bars) * bar_spacing

    data_per_file: Dict[str, Dict[str, Any]] = {}
    for file_id in file_ids:
        is_val = file_id in VAL_STEMS
        optimal_vmt = solutions.scores.get(file_id)
        total_requests = _load_optimal_serviced(solutions.manifests_dir, file_id)
        if optimal_vmt is None or total_requests is None or total_requests == 0:
            continue
        optimal_dpr = optimal_vmt / total_requests
        row: Dict[str, Any] = {"optimal_dpr": optimal_dpr}

        offline_vmt = _load_offline_stat(offline, file_id, "vmt") if offline else None
        offline_serviced = _load_offline_stat(offline, file_id, "serviced") if offline else None
        row["Offline"] = (offline_vmt / offline_serviced) if (offline_vmt and offline_serviced) else None

        coaml_vmt = _load_coaml_stat_per_epoch(coaml_dir, file_id, epochs, is_val, "vmt")
        coaml_serviced = _load_coaml_stat_per_epoch(coaml_dir, file_id, epochs, is_val, "serviced")
        for e in range(1, epochs + 1):
            vmt, svc = coaml_vmt.get(e), coaml_serviced.get(e)
            row[f"COAML ep.{e}"] = (vmt / svc) if (vmt and svc and svc > 0) else None
        data_per_file[file_id] = row

    values = {
        "file_ids": file_ids,
        "run_labels": ["Optimal"] + bar_labels,
        "data_per_file": {fid: {k: v for k, v in d.items()} for fid, d in data_per_file.items()},
    }

    n_files = len(file_ids)
    n_cols = 4
    n_rows = (n_files + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, file_id in enumerate(file_ids):
        row_idx, col_idx = idx // n_cols, idx % n_cols
        ax = axes[row_idx, col_idx]
        row = data_per_file[file_id]
        optimal_dpr = row["optimal_dpr"]
        vals = [row.get(lbl) for lbl in bar_labels]
        heights = [v if v is not None else 0 for v in vals]

        bars = ax.bar(x, heights, width, zorder=1)
        ax.axhline(y=optimal_dpr, color=optimal_line_color, linestyle="--", linewidth=1.5, zorder=2)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            bar.set_color(bar_colors[i])
            if v is None:
                bar.set_hatch("//")
                bar.set_alpha(0.5)
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{h:.1f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="#333333",
                )
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylabel("Distance per request" if col_idx == 0 else "")
        if file_id in VAL_STEMS:
            ax.set_title(f"{file_id} (opt: {optimal_dpr:.1f})", fontweight="bold", color="#c0392b")
        else:
            ax.set_title(f"{file_id} (opt: {optimal_dpr:.1f})")
        ax.set_ylim(bottom=0)

    for idx in range(n_files, n_rows * n_cols):
        row_idx, col_idx = idx // n_cols, idx % n_cols
        axes[row_idx, col_idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=optimal_line_color, linestyle="--", linewidth=2, label="Optimal"),
    ] + [Patch(facecolor=bar_colors[i], label=bar_labels[i]) for i in range(n_bars)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements), frameon=False)
    fig.suptitle("Distance per served request: coverage-efficiency tradeoff (lower is better)")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_figure_and_values(results_dir, sig, values, fig, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate training and benchmark data across solutions, offline, and CoAML."
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default="solutions/li_lim",
        help="Path to solutions folder (absolute benchmark, e.g. solutions/li_lim)",
    )
    parser.add_argument(
        "--offline",
        type=Path,
        default="outputs/experiments/run_offline_mc3_bi400_ss100",
        help="Path to offline benchmark folder (e.g. outputs/experiments/run_offline_mc3_bi400_ss100)",
    )
    parser.add_argument(
        "--coaml",
        type=Path,
        default="outputs/experiments/batch_lilim_coaml_seed42/mc3_bi400_ss100_20260317_231957",
        help="Path to CoAML run folder (e.g. outputs/experiments/.../mc3_bi400_ss100_20260317_231957)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/mc3_bi400_ss100"),
        help="Output folder for all plots and values (default: results)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show plots interactively (default: show)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list loaded data summary, no analysis",
    )
    args = parser.parse_args()

    results_dir = Path(args.results).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    solutions = load_solutions(args.solutions)
    offline = load_offline(args.offline) if args.offline and Path(args.offline).exists() else None
    coaml = load_coaml(args.coaml) if args.coaml and Path(args.coaml).exists() else None

    print("Loaded data:")
    print(f"  Solutions: {len(solutions.instance_ids())} instances")
    if offline:
        print(f"  Offline:   {len(offline.instance_ids())} instances, {sum(len(r) for r in offline.runs.values())} runs")
    else:
        print("  Offline:   (not provided)")
    if coaml:
        print(f"  CoAML:     {len(coaml.instance_ids())} instances in training_loss_per_file")
    else:
        print("  CoAML:     (not provided)")

    aligned = align_instances(solutions, offline, coaml)
    print(f"\nAligned instances (in all provided sources): {len(aligned)}")
    if aligned:
        print(f"  Examples: {aligned[:5]}...")

    if not args.list:
        if coaml and coaml.training_loss_per_file:
            print("\nRunning COAML analyses...")
            analyze_coaml_avg_loss_per_epoch(coaml, results_dir)
            base = FIGURE_SIGNATURES["coaml_avg_loss_per_epoch"]["base"]
            print(f"  Saved: {results_dir / f'{base}.json'}, {results_dir / f'{base}.pdf'}")
            analyze_coaml_avg_loss_per_file_per_epoch(coaml, results_dir)
            base = FIGURE_SIGNATURES["coaml_avg_loss_per_file_per_epoch"]["base"]
            print(f"  Saved: {results_dir / f'{base}.json'}, {results_dir / f'{base}.pdf'}")
        if solutions and coaml:
            print("\nRunning serviced-per-file comparison...")
            analyze_serviced_per_file_comparison(solutions, offline, coaml, results_dir)
            base = FIGURE_SIGNATURES["serviced_per_file_comparison"]["base"]
            print(f"  Saved: {results_dir / f'{base}.json'}, {results_dir / f'{base}.pdf'}")
            print("\nRunning VMT-per-file comparison...")
            analyze_vmt_per_file_comparison(solutions, offline, coaml, results_dir)
            base = FIGURE_SIGNATURES["vmt_per_file_comparison"]["base"]
            print(f"  Saved: {results_dir / f'{base}.json'}, {results_dir / f'{base}.pdf'}")
            print("\nRunning distance-per-request comparison...")
            analyze_distance_per_request_comparison(solutions, offline, coaml, results_dir)
            base = FIGURE_SIGNATURES["distance_per_request_comparison"]["base"]
            print(f"  Saved: {results_dir / f'{base}.json'}, {results_dir / f'{base}.pdf'}")
