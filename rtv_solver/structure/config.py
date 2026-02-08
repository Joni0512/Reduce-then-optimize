from dataclasses import dataclass
from argparse import Namespace
from pathlib import Path

@dataclass
class Config:
    """
    facilitates the usage of arguments in the relevant classes below
    Class is frozen as there should not be any changes to the variables!
    
    no differentiation of separate configs but one object that carries all information
    
    The default values [output_dir, input_file, server_url]should be changed locally  as they also collect the basic information for debug runs. The default values also act as the default for tests. 
    """
    # technical setup
    output_dir: Path = Path("outputs") / "debug"
    input_file: str = "rtv-solver/inputs/wilson_nc_initial.pkl"
    server_url: str = "http://127.0.0.1:5001/"
    max_thread_cnt: int = 16
    rtv_timeout: int = 120
    ilp_timeout: int = 120
    ilp_penalty: int = 1_000_000

    # experiment parameters
    max_cardinality: int = 2
    largest_tsp: int = 8
    share_cost_factor: float = 10
    rebalancing: bool = False
    keep_active: bool = True
    return_depot: bool = False
    dwell_pickup: int = 180
    dwell_alight: int = 60
    walk_distance_cutoff: int = 0
    step_size: int = 300
    batch_interval: int = 1200

    # stats parameters
    travel_time_margin: int = 5

    @classmethod
    def from_args(cls, args: Namespace) -> "Config":
        """
        Initialize from argparse.Namespace or any object with matching attributes.
        Example:
            args = parser.parse_args()
            cfg = Config.from_args(args)
        """
        ROOT_DIR = Path(__file__).resolve().parent.parent.parent
        output_directory = ROOT_DIR / args.output_dir
        output_directory.mkdir(parents=True, exist_ok=True)

        # add the change for the input_file
        return cls(
            output_dir = output_directory,
            input_file = args.input_file,
            server_url = args.server_url,
            max_thread_cnt = args.max_thread_cnt,
            rtv_timeout = args.rtv_timeout,
            ilp_timeout = args.ilp_timeout, 
            ilp_penalty = args.ilp_penalty,
            max_cardinality = args.max_cardinality,
            largest_tsp = args.largest_tsp,
            share_cost_factor = args.share_cost_factor,
            rebalancing = args.rebalancing,
            keep_active = args.keep_active,
            return_depot = args.return_depot,
            dwell_pickup = args.dwell_pickup,
            dwell_alight = args.dwell_alight,
            walk_distance_cutoff = args.walk_distance_cutoff,
            step_size = args.step_size,
            batch_interval = args.batch_interval,
            travel_time_margin = args.travel_time_margin
        )
        # return cls(**vars(args))
