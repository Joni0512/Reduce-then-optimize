import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config
import numpy as np

from typing import Any
from pathlib import Path

def plot_requests_operating_area(
    payload,
    show: bool = True,
    save_path: str | None = None,) -> None:
    """
    Visualizes the operating area of all requests as a scatter plot.

    - Blue dots: pickup locations
    - Red dots: dropoff locations
    - Green dot: depot location

    The x-axis is longitude, the y-axis is latitude, and the axes are scaled to
    the overall min/max values across all points (including the depot).

    NOTE this has not been tested with plots around the Greenwich line.
    """    
    pickup_lats, pickup_lons, dropoff_lats, dropoff_lons, depot_lat, depot_lon = PayloadParser.get_request_positions(payload)

    # determine bounds using handler logic (keeps the min/max computation centralized)
    (min_lat, max_lat), (min_lon, max_lon) = PayloadParser.get_request_operating_area_limits(payload)

    # create plot
    plt.figure()
    if pickup_lats and pickup_lons:
        plt.scatter(pickup_lons, pickup_lats, c="blue", s=10, label="Pickups")
    if dropoff_lats and dropoff_lons:
        plt.scatter(dropoff_lons, dropoff_lats, c="red", s=10, label="Dropoffs")
    if depot_lat is not None and depot_lon is not None:
        plt.scatter([depot_lon], [depot_lat], c="green", s=40, marker="X", label="Depot")
    
    lon_distance = max_lon - min_lon
    lat_distance = max_lat - min_lat

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.xlim(min_lon - 0.02 * lon_distance, max_lon + 0.02 * lon_distance)
    plt.ylim(min_lat - 0.02 * lat_distance, max_lat + 0.02 * lat_distance)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title("Request Operating Area")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_request_positions_xy(
    payload: dict[str, Any],
    show: bool = True,
    save_path: str | None = None,
    show_legend: bool = True,
    show_title: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    """
    Plot Wilson request positions on a simple XY plane.

    - Pickup nodes are green triangles.
    - Dropoff nodes are red circles.
    - The original depot is a black square.
    - Each pickup->dropoff pair is connected by a purple line behind nodes.

    Coordinates are treated as plain XY values:
    x := lon, y := lat.
    """
    requests = payload.get(PayloadKeys.REQUESTS, [])
    if not requests:
        raise ValueError("No requests found in payload.")

    pickup_xs: list[float] = []
    pickup_ys: list[float] = []
    dropoff_xs: list[float] = []
    dropoff_ys: list[float] = []
    request_pairs: list[tuple[float, float, float, float]] = []

    for req in requests:
        pickup_pt = req.get(PayloadKeys.REQ_PICKUP_PT) or {}
        dropoff_pt = req.get(PayloadKeys.REQ_DROPOFF_PT) or {}

        p_x = pickup_pt.get("lon")
        p_y = pickup_pt.get("lat")
        d_x = dropoff_pt.get("lon")
        d_y = dropoff_pt.get("lat")

        if None in (p_x, p_y, d_x, d_y):
            continue

        pickup_xs.append(p_x)
        pickup_ys.append(p_y)
        dropoff_xs.append(d_x)
        dropoff_ys.append(d_y)
        request_pairs.append((p_x, p_y, d_x, d_y))

    if not request_pairs:
        raise ValueError("No valid pickup/dropoff coordinates found in payload requests.")

    depot_data = payload.get(PayloadKeys.DEPOT, {})
    depot_pt = depot_data.get("loc") or depot_data.get(PayloadKeys.DEPOT_PT) or {}
    depot_x = depot_pt.get("lon")
    depot_y = depot_pt.get("lat")

    fig, ax = plt.subplots()

    # Colors (aligned with plot_request_time_windows_timematrix for consistency)
    pickup_color = '#2ecc71'      # Green for pickup
    dropoff_color = '#e74c3c'     # Red for dropoff
    travel_time_color = '#9b59b6'  # Purple for pickup->dropoff connector

    # keep request connectors behind nodes so points remain visible
    for p_x, p_y, d_x, d_y in request_pairs:
        ax.plot([p_x, d_x], [p_y, d_y], color=travel_time_color, linewidth=1.0, zorder=1)

    ax.scatter(
        pickup_xs,
        pickup_ys,
        c=pickup_color,
        marker="^",
        s=30,
        zorder=3,
        label="Pickups",
    )
    ax.scatter(
        dropoff_xs,
        dropoff_ys,
        c=dropoff_color,
        marker="o",
        s=25,
        zorder=3,
        label="Dropoffs",
    )

    if depot_x is not None and depot_y is not None:
        ax.scatter(
            [depot_x],
            [depot_y],
            c="black",
            marker="s",
            s=60,
            zorder=4,
            label="Depot",
        )

    all_x = pickup_xs + dropoff_xs + ([depot_x] if depot_x is not None else [])
    all_y = pickup_ys + dropoff_ys + ([depot_y] if depot_y is not None else [])
    x_min, x_max = min(0.0, min(all_x)), max(90.0, max(all_x))
    y_min, y_max = min(0.0, min(all_y)), max(90.0, max(all_y))
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_margin = 0.02 * x_span if x_span > 0 else 0.01
    y_margin = 0.02 * y_span if y_span > 0 else 0.01

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    if show_xlabel:
        ax.set_xlabel("X (lon)")
    if show_ylabel:
        ax.set_ylabel("Y (lat)")
    if show_title:
        ax.set_title("Request Positions (XY)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2, linestyle="--")
    if show_legend:
        ax.legend()

    plt.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        if path.suffix == "":
            save_path = str(path.with_suffix(".pdf"))
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_request_time_windows_timematrix(
    data: dict[str, Any],
    title="Request Time Windows",
    figsize=(14, 10),
    max_requests=None,
    save_path: str | None = None,
    show: bool = True,
    show_legend: bool = True,
    show_yticklabels: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = False,
):
    """
    # TODO make this work with the wilson format (we do not have the time_matrix in this data)
    Create a horizontal bar plot showing time windows for each request.
    
    Each request shows:
    - Pickup window highlighted in one color
    - Dropoff window highlighted in another color
    - Travel time bar after pickup window

    Requests are sorted by earliest pickup_start, then pickup_end, then dropoff_start, then dropoff_end.

    Args:
        requests: List of request dicts from parse_li_lim_file
        title: Plot title
        figsize: Figure size tuple
        max_requests: Maximum number of requests to show (None for all)
        
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # TODO move data collection into the payloadParser as a static method (so here no responsibility remains on handling the data keys)
    # TODO add support for the new 'chattanooga' format and 'wilson' format
    requests = data['requests']
    travel_time_matrix = data['travel_time_matrix']

    # Sort by earliest pickup_start, then pickup_end, then dropoff_start, then dropoff_end
    sorted_requests = sorted(
        requests,
        key=lambda r: (
            r['pickup_time_window_start'],
            r['dropoff_time_window_start'],
            r['pickup_time_window_end'],
            r['dropoff_time_window_end'],
        ),
    )
    
    # Limit number of requests if specified
    if max_requests is not None:
        sorted_requests = sorted_requests[:max_requests]
    
    n_requests = len(sorted_requests)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors
    pickup_color = '#2ecc71'      # Green for pickup window
    dropoff_color = '#e74c3c'     # Red for dropoff window
    travel_time_color = '#9b59b6'  # Purple for travel time

    y_positions = np.arange(n_requests)
    bar_height = 0.8  # No gap between lines
    plotted_travel_times = []

    for i, req in enumerate(sorted_requests):
        pickup_start = req['pickup_time_window_start']
        pickup_end = req['pickup_time_window_end']
        dropoff_start = req['dropoff_time_window_start']
        dropoff_end = req['dropoff_time_window_end']

        # Draw pickup window
        ax.barh(y_positions[i], pickup_end - pickup_start, 
                left=pickup_start, height=bar_height, 
                color=pickup_color, alpha=0.6, edgecolor='none')
        
        # Draw dropoff window
        ax.barh(y_positions[i], dropoff_end - dropoff_start,
                left=dropoff_start, height=bar_height,
                color=dropoff_color, alpha=0.6, edgecolor='none')

        # Draw travel time bar directly after pickup window ends
        travel_time = travel_time_matrix[req['pickup_pt']['node_id']][req['dropoff_pt']['node_id']]
        plotted_travel_times.append(float(travel_time))
        ax.barh(y_positions[i], travel_time,
                left=pickup_end, height=bar_height,
                color=travel_time_color, alpha=1.0, edgecolor='none')
    
    # Labels
    ax.set_yticks(y_positions)
    if show_yticklabels:
        ax.set_yticklabels([req['booking_id'] for req in sorted_requests], fontsize=6)
    else:
        ax.set_yticklabels([""] * n_requests)
    if show_xlabel:
        ax.set_xlabel('Time', fontsize=12)
    if show_ylabel:
        ax.set_ylabel('Request ID', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    if show_legend:
        pickup_patch = mpatches.Patch(color=pickup_color, alpha=0.8, label='Pickup Window')
        dropoff_patch = mpatches.Patch(color=dropoff_color, alpha=0.8, label='Dropoff Window')
        if plotted_travel_times:
            avg_travel_time = sum(plotted_travel_times) / len(plotted_travel_times)
            travel_time_patch = mpatches.Patch(
                color=travel_time_color, alpha=0.8,
                label=f"Travel Time (avg {avg_travel_time:.1f}s)"
            )
            legend_handles = [pickup_patch, dropoff_patch, travel_time_patch]
        else:
            legend_handles = [pickup_patch, dropoff_patch]
        ax.legend(handles=legend_handles, loc='upper right')
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Invert y-axis so first request is at top
    ax.invert_yaxis()

    # Remove vertical padding: bars span y-0.5 to y+0.5, so data goes -0.5 to n-0.5
    ax.set_ylim(n_requests - 0.5, -0.5)
    ax.margins(y=0)

    plt.tight_layout(pad=0.2)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.06)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return fig, ax

def plot_request_time_windows_woTimematrix(
        data: dict[str, Any],
        save_path: str | None = None,
        show: bool = True,
        step_size: int = 300):
    """
    Plot the request time windows if the time matrix is not part of the data
    """
    # TODO can possible be combined with the function above, but no priority
    payload = PayloadParser.get_payload_object(data, dwell_pickup_default=15, dwell_alight_default=30, online = True)
    request_handler = RequestHandler(payload.requests, 180, 60)

    requests = request_handler.get_all_requests()
    request_count = len(requests)

    travel_times = []
    waiting_windows = []

    for req in requests:
        travel_times.append(req.get_direct_travel_time())
        waiting_windows.append(req.get_time_window_duration())

    min_travel_time = min(travel_times)
    max_travel_time = max(travel_times)
    avg_travel_time = sum(travel_times) / len(travel_times)

    print(f"DEBUG {request_count} requests: min_travel_time: {min_travel_time}, max_travel_time: {max_travel_time}, avg_travel_time: {avg_travel_time}, min_waiting_window: {min(waiting_windows)}, max_waiting_window: {max(waiting_windows)}, avg_waiting_window: {sum(waiting_windows) / len(waiting_windows)}")

    data = {
        'request_id': [req.id for req in requests],
        'earliest_pickup_time': [req.earliest_pickup_time for req in requests],
        'latest_pickup_time': [req.latest_pickup_time for req in requests],
        'earliest_arrival_time': [req.earliest_arrival_time for req in requests],
        'latest_arrival_time': [req.latest_arrival_time for req in requests],
        'travel_time': [req.get_direct_travel_time() for req in requests],
    }

    df = pd.DataFrame(data)
    # Convert times to datetime
    # df['earliest_pickup_time'] = pd.to_datetime(df['earliest_pickup_time'])
    # df['latest_arrival_time'] = pd.to_datetime(df['latest_arrival_time'])

    df = df.sort_values('earliest_pickup_time').reset_index(drop=True)

    # Define figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # --- Top subplot: Gantt bars ---
    y_pos = range(len(df))
    for i, row in df.iterrows():
        # Green: earliest -> latest_pickup
        ax1.hlines(y=i, xmin=row['earliest_pickup_time'], xmax=row['latest_pickup_time'], color='green', alpha=0.2)
        # Red: latest_pickup -> latest_arrival
        ax1.hlines(y=i, xmin=row['earliest_arrival_time'], xmax=row['latest_arrival_time'], color='red', alpha=0.5)
        # TODO there seems to be a bug because in some lines there are multiple travel times in the visual
        # ax1.hlines(y=i, xmin=row['latest_pickup_time'], xmax=row['latest_pickup_time'] + row['travel_time'], color='black', linestyle='--', alpha=1.0)

    #ax1.set_yticks(y_pos)
    #ax1.set_yticklabels(df['request_id'])

    # --- Align x-axis to the first request time ---
    min_time = df['earliest_pickup_time'].min()
    max_time = df['latest_arrival_time'].max()
    min_id = df['request_id'].min()
    max_id = df['request_id'].max()

    ax1.set_xlim(min_time, max_time)
    ax1.set_ylim(min_id, max_id)

    ax1.invert_yaxis()
    ax1.set_ylabel('Request ID')
    ax1.set_title('Requests Time Windows')
    # Create custom legend handles
    legend_elements = [
        Line2D([0], [0], color='green', lw=8, alpha=0.2, label='Pickup Window'),
        Line2D([0], [0], color='red', lw=8, alpha=0.5, label='Dropoff Window'),
        # Line2D([0], [0], color='black', lw=2, alpha=1.0, linestyle='--', label='Travel Time'),
    ]

    # Add legend to the top subplot
    ax1.legend(handles=legend_elements, loc='upper right', frameon=False)

    # --- Bottom subplot: Active requests count ---
    # Create a fine grid of time steps
    time_grid = np.arange(df['earliest_pickup_time'].min(), df['latest_arrival_time'].max(), step_size)  # step 300 seconds
    active_counts = []

    for t in time_grid:
        active = ((df['earliest_pickup_time'] <= t) & (df['latest_arrival_time'] >= t)).sum()
        active_counts.append(active)

    ax2.bar(time_grid, active_counts, width=300.0, color='skyblue', align='edge')
    ax2.set_ylabel('Active Requests')
    ax2.set_xlabel('Time (hours)')
    ax2.set_xlim(min_time, max_time)

    ax2.set_title('Number of Active Requests Over Time')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return fig, ax1, ax2

def batch_plot_request_time_windows_timematrix(
    input_folder: str | Path,
    output_folder: str | Path,
    *,
    recursive: bool = False,
) -> tuple[int, int]:
    """
    Generate and save request time-window plots for all payload files in a folder.

    Supported file types: .json, .pkl, .txt
    Returns: (processed_count, failed_count)
    """
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input folder does not exist or is not a folder: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    suffixes = {".json", ".pkl", ".txt"}
    glob_pattern = "**/*" if recursive else "*"
    files = sorted(
        p for p in input_dir.glob(glob_pattern)
        if p.is_file() and p.suffix.lower() in suffixes
    )

    if not files:
        raise ValueError(
            f"No supported input files found in {input_dir}. "
            f"Expected one of: {sorted(suffixes)}"
        )

    processed_count = 0
    failed_count = 0
    for input_path in files:
        output_path = output_dir / f"{input_path.stem}_time_windows.pdf"
        try:
            data = PayloadParser.load_input_data(input_path)
            plot_request_time_windows_timematrix(data, title=None, save_path=str(output_path), show=False, show_legend=False, show_xlabel=False, show_yticklabels=False, show_ylabel=False)
            processed_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"Failed for {input_path}: {exc}")

    print(
        f"Batch plotting done. Processed: {processed_count}, "
        f"Failed: {failed_count}, Output: {output_dir}"
    )
    return processed_count, failed_count

def batch_plot_request_positions_xy(
    input_folder: str | Path,
    output_folder: str | Path,
    *,
    recursive: bool = False,
) -> tuple[int, int]:
    """
    Generate and save XY request-pair plots for all payload files in a folder.

    Supported file types: .json, .pkl, .txt
    Returns: (processed_count, failed_count)
    """
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input folder does not exist or is not a folder: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    suffixes = {".json", ".pkl", ".txt"}
    glob_pattern = "**/*" if recursive else "*"
    files = sorted(
        p for p in input_dir.glob(glob_pattern)
        if p.is_file() and p.suffix.lower() in suffixes
    )

    if not files:
        raise ValueError(
            f"No supported input files found in {input_dir}. "
            f"Expected one of: {sorted(suffixes)}"
        )

    processed_count = 0
    failed_count = 0
    for input_path in files:
        output_path = output_dir / f"{input_path.stem}_positions.pdf"
        try:
            data = PayloadParser.load_input_data(input_path)
            plot_request_positions_xy(
                data,
                save_path=str(output_path),
                show=False,
                show_legend=False,
                show_title=False,
                show_xlabel=False,
                show_ylabel=False,
            )
            processed_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"Failed for {input_path}: {exc}")

    print(
        f"Batch XY plotting done. Processed: {processed_count}, "
        f"Failed: {failed_count}, Output: {output_dir}"
    )
    return processed_count, failed_count


if __name__ == "__main__":

    # wilson as it does not have the time matrix
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Arguments for the PayloadParser main script')
    parser.add_argument('--input_file', type=str, default='solutions/li_lim/manifests/lr208.json', help='Path to one input file')
    parser.add_argument('--input_folder', type=str,default='solutions/li_lim/manifests/', help='Optional folder with input files to process in batch')
    parser.add_argument('--output_folder', type=str, default='/Users/jw/Desktop/master_thesis/mt_presentation/li_lim_v2/', help='Output folder for saved batch plots')
    parser.add_argument('--recursive', action='store_true', help='Scan input_folder recursively')
    parser.add_argument('--visual', type=str,
        default='request_pairs_xy',
        choices=['request_pairs_xy', 'time_windows'],
        help="Select visual: 'request_pairs_xy' or 'time_windows'.",
    )
    args = parser.parse_args()

    if args.input_folder:
        if args.visual == 'request_pairs_xy':
            batch_plot_request_positions_xy(
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                recursive=args.recursive,
            )
        else:
            batch_plot_request_time_windows_timematrix(
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                recursive=args.recursive,
            )
    else:
        data = PayloadParser.load_input_data(args.input_file)
        if args.visual == 'request_pairs_xy':
            print(f"Plotting request pairs XY for {args.input_file}")
            plot_request_positions_xy(data, save_path=None, show=True)
        else:
            print(f"Plotting request time windows without time matrix for {args.input_file}")
            plot_request_time_windows_timematrix(data, title=None, save_path=None, show=True, show_legend=False, show_xlabel=False, show_yticklabels=False, show_ylabel=False)
