import time
import uuid
import argparse
import os

from dataclasses import dataclass, asdict, field
from typing import List
from pathlib import Path

from rtv_solver.util.helper import save_json, load_json


from rtv_solver.util.logger import BASIC_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)


@dataclass
class Config:
    """
    facilitates the usage of arguments in the relevant classes below
    Class is frozen as there should not be any changes to the variables!
    
    no differentiation of separate configs but one object that carries all information
    
    The default values [output_dir, input_file, server_url]should be changed locally  as they also collect the basic information for debug runs. The default values also act as the default for tests. 
    
    Loggers cannot be used here as they are defined based on the output_dir. For debugging, please use print().
    """
    # FIXME make sure that configs that are missing elements should just take the config values, that should be possibel right (in order to rerun the prior settings with new additional elements)
    # technical setup
    DEBUG: bool = False
    CONFIG_FILE: str = ""
    OVERRIDE: List[str] = field(default_factory=list)
    OUTPUT_DIR: Path = Path("outputs") / "debug"
    INPUT_FILE: str = "rtv-solver/inputs/wilson_nc_initial.pkl"
    INPUT_DIR: str = ""  # coaml mode only: directory of input files to process in batch
    SERVER_URL: str = None # "http://127.0.0.1:5001/"
    MAX_THREAD_CNT: int = 16
    RTV_TIMEOUT: int = 120
    ILP_TIMEOUT: int = 120
    ILP_PENALTY: int = 1_000_000
    # experiment parameters
    MODE: str = 'offline'
    MAX_CARDINALITY: int = 2
    USE_REQUEST_GRAPH_PRUNER: bool = False #new
    REQUEST_GRAPH_MODEL_PATH: str = "outputs/request_graph_gnn.pt" #new
    REQUEST_GRAPH_THRESHOLD: float = 0.5 #new
    LARGEST_TSP: int = 8
    SHARE_COST_FACTOR: float = 10
    REBALANCING: bool = False
    KEEP_ACTIVE: bool = True
    RETURN_DEPOT: bool = False
    INTERMEDIATE_LOCATION: bool = False
    WALK_DISTANCE_CUTOFF: int = 0
    STEP_SIZE: int = 300
    BATCH_INTERVAL: int = 1200
    # backup default values, NOTE currently not the case as it is sometimes the overriding default
    DWELL_PICKUP: int = 180
    DWELL_ALIGHT: int = 60
    # stats parameters
    TRAVEL_TIME_MARGIN: int = 5
    # COAML parameters
    SEED: int = 42
    IMITATION_SOLUTION_FILE: Path | str = 'outputs/test_nc/solution_10r_1v_repeat6_simple/result_driver_runs.json'
    Y_STAR_TYPE: str = "best_ordered_match" # TODO somehow we have circular imports if we imported this from the pipeline module
    EPOCHS: int = 1
    LEARNING_RATE: float = 1e-3
    HIDDEN_DIM: int = 64
    NUM_SAMPLES: int = 20
    SIGMA: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """
        Create a Config instance from the arguments. 
        
        Conversion to lower keys is necessary so both shell scripts and main.py work in parallel.
        """
        reproduced = False

        # print(f"Default arguments: {args}")
        # Convert to dictionary and lower keys; convert back to Namespace
        lower_args_dict = {k.lower(): v for k, v in vars(args).items()}
        args = argparse.Namespace(**lower_args_dict)

        if args.config_file:
            # Load config from file; convert json_dict to lower keys
            cfg_json = load_json(Path(args.config_file))["config_dict"]
            cfg_json = {k.lower(): v for k, v in cfg_json.items()}

            # Extract base output directory
            base_output_dir = cls.derive_base_output_dir(cfg_json["output_dir"])
            cfg_json.pop("output_dir", None)  # ignore old run folder

            # Apply CLI overrides
            override_list = getattr(args, "override", []) or []
            cfg_json = cls.apply_overrides(cfg_json, override_list)

            # Recreate namespace from updated config
            args = argparse.Namespace(**cfg_json)
            args.output_dir = base_output_dir
            args.override = override_list  
            reproduced = True

        # Create new run directory
        ROOT_DIR = Path(__file__).resolve().parent.parent.parent
        input_dir = getattr(args, "input_dir", "") or ""
        if args.mode == "coaml" and input_dir:
            file_name = "batch"
        else:
            file_name = args.input_file.split("/")[-1].split(".")[0]
        output_directory = cls.create_output_dir(ROOT_DIR / "outputs", file_name, args)
        
        # add the change for the input_file
        # if a config is missing an item due to additional config items, one can override it by adding it there and still use the same config settings
        config = cls(
            DEBUG = cls.str_to_bool(args.debug),
            CONFIG_FILE = args.config_file,
            OVERRIDE = args.override, 
            OUTPUT_DIR = output_directory,
            INPUT_FILE = args.input_file,
            INPUT_DIR = getattr(args, "input_dir", "") or "",
            SERVER_URL = getattr(args, "server_url", None), # args.server_url,
            MODE = args.mode,
            USE_REQUEST_GRAPH_PRUNER=cls.str_to_bool(
                getattr(args, "use_request_graph_pruner", "False")
            ),
            REQUEST_GRAPH_MODEL_PATH=getattr(
                args, "request_graph_model_path", "outputs/request_graph_gnn.pt"
            ),
            REQUEST_GRAPH_THRESHOLD=getattr(
                args, "request_graph_threshold", 0.5
            ),
            MAX_THREAD_CNT = args.max_thread_cnt,
            RTV_TIMEOUT = args.rtv_timeout,
            ILP_TIMEOUT = args.ilp_timeout, 
            ILP_PENALTY = args.ilp_penalty,
            MAX_CARDINALITY = args.max_cardinality,
            LARGEST_TSP = args.largest_tsp,
            SHARE_COST_FACTOR = args.share_cost_factor,
            REBALANCING = cls.str_to_bool(args.rebalancing),
            KEEP_ACTIVE = cls.str_to_bool(args.keep_active),
            RETURN_DEPOT = cls.str_to_bool(args.return_depot),
            INTERMEDIATE_LOCATION = cls.str_to_bool(args.intermediate_location),
            DWELL_PICKUP = args.dwell_pickup,
            DWELL_ALIGHT = args.dwell_alight,
            WALK_DISTANCE_CUTOFF = args.walk_distance_cutoff,
            STEP_SIZE = args.step_size,
            BATCH_INTERVAL = args.batch_interval,
            TRAVEL_TIME_MARGIN = args.travel_time_margin,
            SEED = args.seed,
            IMITATION_SOLUTION_FILE = args.imitation_solution_file,
            Y_STAR_TYPE = args.y_star_type,
            EPOCHS = getattr(args, "epochs", 1),
            LEARNING_RATE = getattr(args, "learning_rate", 1e-3),
            HIDDEN_DIM = getattr(args, "hidden_dim", 64),
            NUM_SAMPLES = getattr(args, "num_samples", 20),
            SIGMA = getattr(args, "sigma", 0.2)
        )
        # some checks to fail early if the config is not valid
        assert config.STEP_SIZE <= config.BATCH_INTERVAL, f"MUST: Step size {config.STEP_SIZE} <= batch interval {config.BATCH_INTERVAL}"
        assert config.EPOCHS >= 1, f"MUST: EPOCHS >= 1, got {config.EPOCHS}"

        save_json({"config_dict": config.to_dict(),
                "git_commit": os.popen("git rev-parse HEAD").read().strip(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reproduced": reproduced}, 
                config.OUTPUT_DIR / "config.json")

        return config
        # return cls(**vars(args))

    def enforce_constraints(self):
        """
        Enforce constraints on the config as certain parameters are dependent on other parameters.
        """
        # based on the availability of SERVER_URL, we need to check that we can only calculate with INTERMEDIATE_LOCATION false as the backend server is required in that situation
        if self.SERVER_URL is None:
            console_logger.warning("SERVER_URL is not set, setting INTERMEDIATE_LOCATION to False")
            self.INTERMEDIATE_LOCATION = False

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> "Config":
        """Create a Config instance directly from a dictionary."""
        return cls(**cfg_dict)

    @classmethod
    def from_run_dir(cls, run_dir: Path | str, output_dir: Path | str) -> "Config":
        """
        Rebuild a Config from a previous run's config.json while overriding OUTPUT_DIR.
        """
        config_json = load_json(Path(run_dir) / "config.json")
        config_dict = dict(config_json["config_dict"])
        config_dict["OUTPUT_DIR"] = Path(output_dir)
        return cls.from_dict(config_dict)

    @staticmethod
    def create_output_dir(base_dir: Path, file_name: str, args: argparse.Namespace) -> Path:
        """Create a unique output directory with timestamp or UUID."""
        if args.mode == 'coaml':
            runs_folder = f"batch_lilim_{args.mode}_seed{args.seed}"
            file_folder = f"mc{args.max_cardinality}_bi{args.batch_interval}_ss{args.step_size}_{time.strftime('%Y%m%d_%H%M%S')}"
        else:
            runs_folder = f"run_{args.mode}_mc{args.max_cardinality}_bi{args.batch_interval}_ss{args.step_size}"
            file_folder = f"{file_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        unique_dir = base_dir / args.output_dir / runs_folder / file_folder
        unique_dir.mkdir(parents=True, exist_ok=True)
        return unique_dir
    
    @staticmethod
    def derive_base_output_dir(old_output_dir: str | Path) -> Path:
        old = Path(old_output_dir)
        return old.parent   # drop the run_* directory if loaded from a config
    
    @staticmethod
    def apply_overrides(cfg: dict, overrides: list[str]):
        if not overrides:
            return cfg

        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Invalid override '{item}', use key=value")

            key, value = item.split("=", 1)

            if key not in cfg:
                raise KeyError(f"Unknown config key '{key}'")

            old_value = cfg[key]
            cfg[key] = Config.cast_value(value, type(old_value))
            print(f"Applied override '{key}': {old_value} -> {cfg[key]}")

        return cfg
    
    @staticmethod
    def cast_value(value, target_type):
        if target_type is bool:
            return str(value).lower() in ("true", "1", "yes")
        return target_type(value)

    def str_to_bool(value: str) -> bool:
        return value.strip().lower() == "true"