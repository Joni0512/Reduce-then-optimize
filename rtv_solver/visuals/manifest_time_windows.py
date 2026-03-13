"""
Visualize the manifest of a solved run as a horizontal bar chart.

For each pickup/dropoff action in the manifest (in manifest order, grouped by vehicle):
- A horizontal bar spans the action's time window (green for pickup, red for dropoff)
- A vertical line marks the scheduled arrival time within that window
- The booking ID is printed inline on the bar
- Vehicle groups are separated by a horizontal divider
- When a "depot" stop with a scheduled_time is present in the manifest, a short grey
  bar is appended showing the depot return leg (last dropoff → depot arrival).
  If the depot stop is absent or has no scheduled_time, no row is added.
"""

import json
import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from rtv_solver.handlers.payload_parser import PayloadParser


PICKUP_COLOR = "#2ecc71"
DROPOFF_COLOR = "#e74c3c"
PICKUP_MARKER_COLOR = "#1a5c32"
DROPOFF_MARKER_COLOR = "#7b241c"
DEPOT_RETURN_COLOR = "#888888"
SEPARATOR_COLOR = "#aaaaaa"

BAR_HEIGHT = 0.55

def _collect_actions(result_data: dict[str, dict[str, Any]]) -> list[dict]:
    """
    Flatten all manifest stops across all driver_runs into an ordered list of dicts.

    Each entry contains:
        run_id, booking_id, action, order, scheduled_time,
        time_window_start, time_window_end

    "depot" stops are skipped as regular rows.  If the manifest contains a depot
    stop with a ``scheduled_time``, a synthetic ``depot_return`` entry is appended
    whose bar spans from the previous stop's ``scheduled_time`` to the depot's
    ``scheduled_time``.  When the depot stop is absent or carries no
    ``scheduled_time``, no depot row is added.
    """
    rows = []
    for driver_run in result_data.get("driver_runs", []):
        run_id = driver_run["state"]["run_id"]
        last_scheduled_time = None
        depot_arrival_time = None

        for stop in driver_run.get("manifest", []):
            if stop["action"] == "depot":
                depot_arrival_time = stop.get("scheduled_time")
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "booking_id": int(stop["booking_id"]),
                    "action": stop["action"],
                    "order": stop["order"],
                    "scheduled_time": stop["scheduled_time"],
                    "time_window_start": stop["time_window_start"],
                    "time_window_end": stop["time_window_end"],
                }
            )
            last_scheduled_time = stop["scheduled_time"]

        if last_scheduled_time is not None and depot_arrival_time is not None:
            rows.append(
                {
                    "run_id": run_id,
                    "booking_id": f"-{run_id}",
                    "action": "depot_return",
                    "order": None,
                    "scheduled_time": depot_arrival_time,
                    "time_window_start": last_scheduled_time,
                    "time_window_end": depot_arrival_time,
                }
            )
    return rows


def plot_manifest_time_windows(
    result_data: dict[str, Any],
    title: str = "Manifest Solution",
    figsize: tuple[int, int] = (14, 10),
    time_grid_interval: int | None = 600,
    save_path: str | None = None,
    show: bool = True,
) -> tuple:
    """
    Plot the manifest of a solved run as a horizontal bar chart.

    Each row corresponds to one pickup or dropoff action in the manifest,
    ordered by manifest sequence and grouped by vehicle. Pickup windows are
    drawn in green, dropoff windows in red. A vertical line marks the
    scheduled arrival time. Vehicle boundaries are separated by a divider.

    Args:
        result_data:         Loaded result_driver_runs.json dict.
        title:               Plot title.
        figsize:             Figure size (width, height) in inches.
        time_grid_interval:  Draw vertical dashed lines every N seconds starting
                             from the earliest pickup time.  Pass ``None`` to
                             disable.  Defaults to 600 (10 minutes).
        save_path:           Optional file path to save the figure.
        show:                Whether to call plt.show().

    Returns:
        (fig, ax) matplotlib objects.
    """
    actions = _collect_actions(result_data)
    if not actions:
        print("No manifest actions found in result data.")
        return None, None

    n = len(actions)
    y_positions = np.arange(n)

    # Determine vehicle group boundaries for separator lines
    vehicle_boundaries: list[int] = []
    prev_run_id = actions[0]["run_id"]
    for idx, action in enumerate(actions):
        if idx > 0 and action["run_id"] != prev_run_id:
            vehicle_boundaries.append(idx)
            prev_run_id = action["run_id"]

    # Compute x-axis bounds across all rows (depot_return rows have real end times)
    all_starts = [a["time_window_start"] for a in actions]
    all_ends = [a["time_window_end"] for a in actions]
    x_min, x_max = min(all_starts), max(all_ends)
    x_range = x_max - x_min
    x_left = x_min - x_range * 0.02
    x_right = x_max + x_range * 0.02

    # Dynamic figure height: at least 6, scales with number of rows
    dynamic_height = max(figsize[1], int(n * 0.45) + 2)
    fig, ax = plt.subplots(figsize=(figsize[0], dynamic_height))

    for i, action in enumerate(actions):
        y = y_positions[i]
        tw_start = action["time_window_start"]
        tw_end = action["time_window_end"]
        bid = action["booking_id"]
        is_depot_return = action["action"] == "depot_return"

        if is_depot_return:
            # Grey bar spanning last dropoff scheduled_time → depot arrival time
            ax.barh(
                y,
                tw_end - tw_start,
                left=tw_start,
                height=BAR_HEIGHT,
                color=DEPOT_RETURN_COLOR,
                alpha=0.45,
                edgecolor=DEPOT_RETURN_COLOR,
                linewidth=0.6,
            )
            label_x = (tw_start + tw_end) / 2
            ax.text(
                label_x,
                y,
                "→ Depot",
                ha="center",
                va="center",
                fontsize=7.5,
                fontstyle="italic",
                color="white",
                zorder=4,
            )
            continue

        is_pickup = action["action"] == "pickup"
        bar_color = PICKUP_COLOR if is_pickup else DROPOFF_COLOR
        marker_color = PICKUP_MARKER_COLOR if is_pickup else DROPOFF_MARKER_COLOR
        sched = action["scheduled_time"]

        # Time window bar
        ax.barh(
            y,
            tw_end - tw_start,
            left=tw_start,
            height=BAR_HEIGHT,
            color=bar_color,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
        )

        # Scheduled time vertical line
        ax.vlines(
            sched,
            y - BAR_HEIGHT / 2,
            y + BAR_HEIGHT / 2,
            color=marker_color,
            linewidth=2.0,
            zorder=3,
        )

        # Inline booking_id label (centered on the bar)
        label_x = (tw_start + tw_end) / 2
        ax.text(
            label_x,
            y,
            str(bid),
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    # Vertical time-grid lines at regular intervals along the x-axis
    if time_grid_interval is not None:
        t = x_min
        while t <= x_right:
            ax.axvline(t, color=SEPARATOR_COLOR, linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
            t += time_grid_interval

    # Vehicle group separators
    for boundary_idx in vehicle_boundaries:
        sep_y = boundary_idx - 0.5
        ax.axhline(sep_y, color=SEPARATOR_COLOR, linewidth=1.2, linestyle="--", zorder=2)

    # Y-axis labels: "P:5" / "D:5" / "→ depot", prefixed by vehicle if multiple
    multi_vehicle = len(set(a["run_id"] for a in actions)) > 1
    y_labels = []
    for action in actions:
        if action["action"] == "depot_return":
            label = f"V{action['run_id']} | depot" if multi_vehicle else "depot"
        else:
            prefix = "P" if action["action"] == "pickup" else "D"
            label = (
                f"V{action['run_id']} | {prefix}:{action['booking_id']}"
                if multi_vehicle
                else f"{prefix}:{action['booking_id']}"
            )
        y_labels.append(label)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_yaxis()

    ax.set_xlim(x_left, x_right)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Manifest Action (in order)", fontsize=11)
    title = title + f"(interval split {time_grid_interval})"
    ax.set_title(title, fontsize=13, fontweight="bold")

    # ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Legend
    pickup_patch = mpatches.Patch(color=PICKUP_COLOR, alpha=0.75, label="Pickup Window")
    dropoff_patch = mpatches.Patch(color=DROPOFF_COLOR, alpha=0.75, label="Dropoff Window")
    pickup_line = Line2D([0], [0], color=PICKUP_MARKER_COLOR, linewidth=2, label="Scheduled Pickup")
    dropoff_line = Line2D([0], [0], color=DROPOFF_MARKER_COLOR, linewidth=2, label="Scheduled Dropoff")
    legend_handles = [pickup_patch, dropoff_patch, pickup_line, dropoff_line]
    has_depot_return = any(a["action"] == "depot_return" for a in actions)
    if has_depot_return:
        depot_patch = mpatches.Patch(color=DEPOT_RETURN_COLOR, alpha=0.45, label="Depot Return")
        legend_handles.append(depot_patch)
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot manifest time windows from a payload file (json/pkl/txt)"
    )
    _project_root = Path(__file__).parent.parent.parent
    parser.add_argument(
        "--result_file",
        type=str,
        default=str(_project_root / "solutions/li_lim/manifests/lc101.json"), # or _project_root / "outputs/storage/optimal_solutions/run_20260302_131028_020e69/result_driver_runs.json"), # or _project_root / "outputs/storage/comp_v1/run_20260224_142551_rhML_extendedCard/result_driver_runs.json"),
        help="Path to payload input file (json/pkl/txt)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=f"Manifest Solution",
        help="Plot title",
    )
    parser.add_argument(
        "--time_grid_interval",
        type=int,
        default=400,
        help="Draw vertical dashed lines every N seconds from the earliest time (default: 600). "
             "Pass 0 to disable.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional path to save the figure (e.g. output.png)",
    )
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Show the plot window",
    )
    show_group.add_argument(
        "--no_show",
        dest="show",
        action="store_false",
        help="Skip plt.show() (useful for batch/headless runs)",
    )
    parser.set_defaults(show=True)
    args = parser.parse_args()

    result_path = Path(args.result_file)
    data = PayloadParser.load_input_data(result_path)
    driver_runs = data.get("driver_runs", [])
    if len(driver_runs) == 0:
        raise ValueError(f"No driver_runs found in payload: {result_path}")

    # Preserve existing manifests as loaded; fail fast if none are available to plot.
    has_manifest_entries = any(len(run.get("manifest", [])) > 0 for run in driver_runs)
    if not has_manifest_entries:
        raise ValueError(
            "No manifest entries found. Li-Lim benchmark instance files contain requests but no solved manifests. Provide a solved payload with non-empty driver manifests."
        )

    grid_interval = args.time_grid_interval if args.time_grid_interval != 0 else None

    plot_manifest_time_windows(
        data,
        title=args.title,
        time_grid_interval=grid_interval,
        save_path=args.save_path,
        show=args.show,
    )
