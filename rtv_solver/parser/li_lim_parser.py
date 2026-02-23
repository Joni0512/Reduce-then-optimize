"""
Parser for Li & Lim PDPTW benchmark instances.
https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/

Format:
Line 1: K (vehicles), Q (capacity), S (speed - not used)
Line 2+: TASK_NO, X, Y, DEMAND, EARLIEST_TIME, LATEST_TIME, SERVICE_TIME, PICKUP_INDEX, DELIVERY_INDEX

Task 0 is the depot.
For pickups: PICKUP_INDEX=0, DELIVERY_INDEX=corresponding delivery task index
For deliveries: PICKUP_INDEX=corresponding pickup task index, DELIVERY_INDEX=0
"""
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

class LiLimParser:

    @staticmethod
    def parse_li_lim_file(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header = lines[0].strip().split('\t')
        num_vehicles = int(header[0])
        vehicle_capacity = int(header[1])
        depot_info = lines[1].strip().split('\t')
        depot_start_time = int(depot_info[4])
        depot_end_time = int(depot_info[5])
        depot_loc = {'lon': float(depot_info[1]), 'lat': float(depot_info[2])}



        driver_runs = []
        for i in range(num_vehicles):
            driver_runs.append({
                'state': {
                    'run_id': i,
                    'start_time': depot_start_time,
                    'end_time': depot_end_time,
                    'am_capacity': vehicle_capacity,
                    'wc_capacity': 0,
                    'locations_already_serviced': 0,
                    'location_dt_seconds': depot_start_time,
                    'loc': copy.deepcopy(depot_loc),
                    'total_locations': 0
                },
                'manifest': [
                ]
            })
    
        # Parse tasks
        tasks = {}
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            task_no = int(parts[0])
            tasks[task_no] = {
                'task_no': task_no,
                'x': float(parts[1]),
                'y': float(parts[2]),
                'demand': int(parts[3]),
                'earliest': int(parts[4]),
                'latest': int(parts[5]),
                'service_time': int(parts[6]),
                'pickup_idx': int(parts[7]),
                'delivery_idx': int(parts[8])
            }
        
        # Task 0 is the depot
        depot_task = tasks[0]
        depot = {'pt': {'lon': depot_task['x'], 'lat': depot_task['y']}}
        
        # Find pickup-delivery pairs
        # Pickups have pickup_idx=0 and delivery_idx>0
        # Deliveries have pickup_idx>0 and delivery_idx=0
        requests = []
        processed = set()
        
        for task_no, task in tasks.items():
            if task_no == 0:  # Skip depot
                continue
            if task_no in processed:
                continue
                
            # Check if this is a pickup (pickup_idx=0, delivery_idx>0)
            if task['pickup_idx'] == 0 and task['delivery_idx'] > 0:
                pickup_task = task
                delivery_task_no = task['delivery_idx']
                delivery_task = tasks[delivery_task_no]
                
                # Mark both as processed
                processed.add(task_no)
                processed.add(delivery_task_no)
                
                # Create request
                request = {
                    'booking_id': str(task_no),
                    'pickup_pt': {
                        'lon': pickup_task['x'],
                        'lat': pickup_task['y'],
                        'node_id': pickup_task['task_no']
                    },
                    'dropoff_pt': {
                        'lon': delivery_task['x'],
                        'lat': delivery_task['y'],
                        'node_id': delivery_task['task_no']
                    },
                    'pickup_time_window_start': pickup_task['earliest'],
                    'pickup_time_window_end': pickup_task['latest'],
                    'dropoff_time_window_start': delivery_task['earliest'],
                    'dropoff_time_window_end': delivery_task['latest'],
                    'am': abs(pickup_task['demand']),  # Use absolute value of demand
                    'wc': 0,
                    # Additional fields that might be useful
                    'pickup_service_time': pickup_task['service_time'],
                    'dropoff_service_time': delivery_task['service_time']
                }
                requests.append(request)
        
        # order requests by pickup_time_window_start
        requests = sorted(requests, key=lambda r: r['pickup_time_window_start'])


        # build the travel time matrix (using Euclidean distance as a proxy for travel time)
        num_tasks = len(tasks)
        travel_time_matrix = np.zeros((num_tasks, num_tasks))

        for i, task_i in tasks.items():
            for j, task_j in tasks.items():
                if i == j:
                    travel_time_matrix[i][j] = 0
                else:
                    distance = np.sqrt((task_i['x'] - task_j['x'])**2 + (task_i['y'] - task_j['y'])**2)
                    travel_time_matrix[i][j] = distance

        # Normalize travel time matrix by vehicle speed (if needed)
        # Assuming speed is 1 unit per time for now
        travel_time_matrix = travel_time_matrix.tolist()
        return {"requests": requests, "depot": depot, "driver_runs": driver_runs, "travel_time_matrix": travel_time_matrix}

    @staticmethod
    def plot_request_time_windows(requests, travel_time_matrix, title="Request Time Windows", figsize=(14, 10), 
                                   sort_by='pickup_start', max_requests=None):
        """
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
        return fig, ax

    @staticmethod
    def plot_from_file(filepath, **kwargs):
        """
        Convenience method to parse a file and plot time windows.
        
        Args:
            filepath: Path to Li & Lim instance file
            **kwargs: Additional arguments passed to plot_request_time_windows
            
        Returns:
            fig, ax: matplotlib figure and axis objects
        """
        data = LiLimParser.parse_li_lim_file(filepath)
        return LiLimParser.plot_request_time_windows(data['requests'], data["travel_time_matrix"], **kwargs)