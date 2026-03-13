class PayloadKeys:
    """
    To fix issues with circular imports, we use this class to define the keys for the payload dictionary.
    
    This class is used to define the keys for the payload dictionary that can be used globally.
    """
    DATE = "date"
    TIME_MATRIX = "travel_time_matrix"
    CURRENT_TIME = "current_time"

    DEPOT = "depot"
    DEPOT_PT = 'pt'

    REQUESTS = "requests"
    REQ_BOOKING_ID = "booking_id" # translate all to Booking_ID
    REQ_PICKUP_PT = "pickup_pt"
    REQ_PICKUP_LAT = 'pickup_latitude'
    REQ_PICKUP_LON = 'pickup_longitude'
    REQ_PICKUP_NODE_ID = 'pickup_node_id'
    REQ_DROPOFF_PT = "dropoff_pt"
    REQ_DROPOFF_NODE_ID = 'dropoff_node_id'
    REQ_DROPOFF_LAT = 'dropoff_latitude'
    REQ_DROPOFF_LON = 'dropoff_longitude'
    REQ_PICKUP_WINDOW_START = 'pickup_time_window_start'
    REQ_PICKUP_WINDOW_END = 'pickup_time_window_end'
    REQ_DROPOFF_WINDOW_START = 'dropoff_time_window_start'
    REQ_DROPOFF_WINDOW_END = 'dropoff_time_window_end'
    REQ_AMBULATORY = 'am'
    REQ_WHEELCHAIR = 'wc'
    REQ_DWELL_PICKUP = 'dwell_pickup'
    REQ_DWELL_ALIGHT = 'dwell_alight'

    DRIVERS = "driver_runs"
    DRIVER_STATE = "state"
    DRIVER_STATE_RUN_ID = "run_id"
    DRIVER_STATE_START_TIME = "start_time"
    DRIVER_STATE_END_TIME = "end_time"
    DRIVER_STATE_AM_CAP = "am_capacity"
    DRIVER_STATE_WC_CAP = "wc_capacity"
    DRIVER_STATE_T_LOCS = "total_locations"
    DRIVER_STATE_LOC = "loc"
    DRIVER_STATE_DT_SEC = "location_dt_seconds"
    DRIVER_STATE_LOC_SERV = "locations_already_serviced"

    DRIVER_MANIFEST = "manifest"
    MANIFEST_RUN_ID = "run_id"
    MANIFEST_ORDER = "order"
    MANIFEST_ACTION = "action"
    MANIFEST_BOOKING_ID = "booking_id"
    MANIFEST_LOC = "loc"
    MANIFEST_AMBULATORY = "am"
    MANIFEST_WHEELCHAIR = "wc"
    MANIFEST_SCHED_TIME = "scheduled_time"
    MANIFEST_TIME_WINDOW_START = "time_window_start"
    MANIFEST_TIME_WINDOW_END = "time_window_end"
    # TODO add these changes to the creation of stops from manifests in the parser (otherwise we cannot reach par with LiLimParser solutions) - in LiLim solutions we have pickup_service_time and dropoff_service_time
    MANIFEST_DWELL = "dwell"
    MANIFEST_ARRIVAL_TIME = "arrival_time"
    MANIFEST_SERVICE_START_TIME = "service_start_time"
    MANIFEST_SERVICE_END_TIME = "service_end_time"
    

    STATS_ASSIGNMENT_DEVELOPMENT = "stats_assign_dev"   
    STATS_ASSIGNED = 'assigned_requests'
    STATS_UNSERVED = 'unserved_requests'
    STATS_BOARDED = 'boarded'
    STATS_DROPPED = 'dropped'