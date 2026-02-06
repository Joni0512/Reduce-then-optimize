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
    output_dir: Path = Path("output_format") / "debug"
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
        return cls(**vars(args))
