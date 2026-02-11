import matplotlib.pyplot as plt

from rtv_solver.handlers.payload_parser import PayloadParser


def plot_requests_operating_area(
    payload,
    show: bool = True,
    save_path: str | None = None,
) -> None:
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

