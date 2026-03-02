import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
import numpy as np

from typing import Any

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

def plot_request_time_windows_timematrix(
    data: dict[str, Any],
    title="Request Time Windows", 
    figsize=(14, 10), 
    sort_by='pickup_start', 
    max_requests=None,
    save_path: str | None = None,
    show: bool = True):
    """
    # TODO make this work with the wilson format (we do not have the time_matrix in this data)
    Create a horizontal bar plot showing time windows for each request.
    
    Each request shows:
    - A bar spanning from earliest pickup time to latest dropoff time
    - Pickup window highlighted in one color
    - Dropoff window highlighted in another color
    
    Args:
        requests: List of request dicts from parse_li_lim_file
        title: Plot title
        figsize: Figure size tuple
        sort_by: How to sort requests - 'pickup_start', 'dropoff_end', 'duration', or 'booking_id'
        max_requests: Maximum number of requests to show (None for all)
        
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # TODO move data collection into the payloadParser as a static method (so here no responsibility remains on handling the data keys)
    # TODO add support for the new 'chattanooga' format and 'wilson' format
    requests = data['requests']
    travel_time_matrix = data['travel_time_matrix']

    # Sort requests
    if sort_by == 'pickup_start':
        sorted_requests = sorted(requests, key=lambda r: r['pickup_time_window_start'])
    elif sort_by == 'dropoff_end':
        sorted_requests = sorted(requests, key=lambda r: r['dropoff_time_window_end'])
    elif sort_by == 'duration':
        sorted_requests = sorted(requests, 
            key=lambda r: r['dropoff_time_window_end'] - r['pickup_time_window_start'])
    elif sort_by == 'booking_id':
        sorted_requests = sorted(requests, key=lambda r: int(r['booking_id']))
    else:
        sorted_requests = requests
    
    # Limit number of requests if specified
    if max_requests is not None:
        sorted_requests = sorted_requests[:max_requests]
    
    n_requests = len(sorted_requests)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors
    pickup_color = '#2ecc71'      # Green for pickup window
    dropoff_color = '#e74c3c'     # Red for dropoff window
    span_color = '#bdc3c7'        # Gray for overall span
    
    y_positions = np.arange(n_requests)
    bar_height = 0.6
    
    for i, req in enumerate(sorted_requests):
        pickup_start = req['pickup_time_window_start']
        pickup_end = req['pickup_time_window_end']
        dropoff_start = req['dropoff_time_window_start']
        dropoff_end = req['dropoff_time_window_end']
        
        # Draw overall span (light gray background)
        ax.barh(y_positions[i], dropoff_end - pickup_start, 
                left=pickup_start, height=bar_height, 
                color=span_color, alpha=0.3, edgecolor='none')
        
        # Draw pickup window
        ax.barh(y_positions[i], pickup_end - pickup_start, 
                left=pickup_start, height=bar_height, 
                color=pickup_color, alpha=0.8, edgecolor='none')
        
        # Draw dropoff window
        ax.barh(y_positions[i], dropoff_end - dropoff_start, 
                left=dropoff_start, height=bar_height, 
                color=dropoff_color, alpha=0.8, edgecolor='none')

        # Draw travel time from pickup to dropoff as a line if started at pickup start
        travel_time = travel_time_matrix[req['pickup_pt']['node_id']][req['dropoff_pt']['node_id']]
        ax.plot([pickup_start,pickup_start + travel_time], [y_positions[i], y_positions[i]], 
                color='blue', linestyle='--', alpha=0.7, label='Travel Time' if i == 0 else "")
    
    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([req['booking_id'] for req in sorted_requests], fontsize=6)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Request ID', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend
    pickup_patch = mpatches.Patch(color=pickup_color, alpha=0.8, label='Pickup Window')
    dropoff_patch = mpatches.Patch(color=dropoff_color, alpha=0.8, label='Dropoff Window')
    span_patch = mpatches.Patch(color=span_color, alpha=0.3, label='Overall Span')
    ax.legend(handles=[pickup_patch, dropoff_patch, span_patch], loc='upper right')
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Invert y-axis so first request is at top
    ax.invert_yaxis()
    
    plt.tight_layout()

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
        show: bool = True):
    """
    Plot the request time windows if the time matrix is not part of the data
    """
    # TODO can possible be combined with the function above, but no priority
    payload = PayloadParser.get_payload_object(data, online = True)
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
        ax1.hlines(y=i, xmin=row['latest_pickup_time'], xmax=row['latest_pickup_time'] + row['travel_time'], color='black', linestyle='--', alpha=1.0)

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
        Line2D([0], [0], color='black', lw=2, alpha=1.0, linestyle='--', label='Travel Time'),
    ]

    # Add legend to the top subplot
    ax1.legend(handles=legend_elements, loc='upper right', frameon=False)

    # --- Bottom subplot: Active requests count ---
    # Create a fine grid of time steps
    time_grid = np.arange(df['earliest_pickup_time'].min(), df['latest_arrival_time'].max(), 300.0)  # step 300 seconds
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


if __name__ == "__main__":
    from rtv_solver.parser.li_lim_parser import LiLimParser
    from rtv_solver.parser.sartori_parser import SartoriParser

    # other data parsers 
    # parser = LiLimParser()       # swap this one line to change parser
    # input_file = 'inputs/li_lim/pdp_100/lc102.txt'
    # data = parser.parse_file(input_file)
    # plot_request_time_windows_timematrix(data, save_path=None, show=True)

    # parser = SartoriParser()
    # input_file = 'inputs/sartori/n100/bar-n100-2.txt'
    # data = parser.parse_file(input_file)
    # plot_request_time_windows_timematrix(data, save_path=None, show=True)

    # wilson as it does not have the time matrix
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Arguments for the PayloadParser main script')
    parser.add_argument('--input_file', type=str, default='inputs/wilson/random_weekeday_2.pkl', help='Path to the input file')
    args = parser.parse_args()

    data = PayloadParser.load_input_data(args.input_file)

    plot_request_time_windows_woTimematrix(data, save_path=None, show=True)
