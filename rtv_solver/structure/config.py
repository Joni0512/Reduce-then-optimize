import time
import uuid
import argparse
import os

from dataclasses import dataclass, asdict
from pathlib import Path

from rtv_solver.util.helper import save_json, load_json

@dataclass
class Config:
    """
    facilitates the usage of arguments in the relevant classes below
    Class is frozen as there should not be any changes to the variables!
    
    no differentiation of separate configs but one object that carries all information
    
    The default values [output_dir, input_file, server_url]should be changed locally  as they also collect the basic information for debug runs. The default values also act as the default for tests. 
    """
    # technical setup
    config_file: str = ""
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

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """
        Initialize config class from argparse.Namespace
        """
        reproduced = False
        if args.config_file != "":
            cfg_json = load_json(Path(args.config_file))["config_dict"]
            base_output_dir = cls.derive_base_output_dir(cfg_json["output_dir"])
            cfg_json.pop("output_dir", None) # - ignore old rerun

            args = argparse.Namespace(**cfg_json)
            args.output_dir = base_output_dir
            reproduced = True
        
        ROOT_DIR = Path(__file__).resolve().parent.parent.parent
        output_directory = cls.create_output_dir(ROOT_DIR / "outputs", args.output_dir)

        # add the change for the input_file
        config =  cls(
            config_file = args.config_file,
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

        save_json({"config_dict": config.to_dict(),
                "git_commit": os.popen("git rev-parse HEAD").read().strip(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reproduced": reproduced}, 
                config.output_dir / "config.json")

        return config
        # return cls(**vars(args))

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> "Config":
        """Create a Config instance directly from a dictionary."""
        return cls(**cfg_dict)

    @staticmethod
    def create_output_dir(base_dir: Path, experiment_dir: Path | str) -> Path:
        """Create a unique output directory with timestamp or UUID."""
        unique_dir = base_dir / experiment_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        unique_dir.mkdir(parents=True, exist_ok=True)
        return unique_dir
    
    @staticmethod
    def derive_base_output_dir(old_output_dir: str | Path) -> Path:
        old = Path(old_output_dir)
        return old.parent   # ← drop the run_* directory if loaded from a config
