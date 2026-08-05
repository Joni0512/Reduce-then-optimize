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
    # 2026-07-19: coaml mode, single-payload train/val (NYC has one big request pool, not
    # many Li&Lim-style instance files, so INPUT_DIR's directory split doesn't apply). If
    # set, main.py trains on INPUT_FILE across EPOCHS and evaluates (mode="eval", no
    # gradient) on VAL_INPUT_FILE each epoch, saving the best-val-service-rate checkpoint -
    # see COAMLTrainingLoop._run_train_val_payloads.
    VAL_INPUT_FILE: str = ""
    # 2026-07-19: --epochs 1 coaml runs only - see argparse help in main.py.
    COAML_MODEL_WEIGHTS: str = ""
    COAML_SOLVE_MODE: str = "train"
    SERVER_URL: str = None # "http://127.0.0.1:5001/"
    MAX_THREAD_CNT: int = 16
    RTV_TIMEOUT: int = 120
    ILP_TIMEOUT: int = 120
    ILP_PENALTY: int = 1_000_000
    # experiment parameters
    MODE: str = 'offline'
    MAX_CARDINALITY: int = 2
    USE_REQUEST_GRAPH_PRUNER: bool = False #new
    # 2026-07-07: default now points at the best v2 GNN checkpoint (pos_weight=10,
    # threshold=0.5 chosen by val F3 - see outputs/models_v2_gnnv2/best_config_by_model_val_f3.csv).
    # The old path pointed at a v1-shaped checkpoint, which would now fail to load
    # since RequestGraphPruner instantiates RequestGraphEdgeGNNv2.
    REQUEST_GRAPH_MODEL_PATH: str = "outputs/models_v2_gnnv2/rgnn_mixed_c2_pw10_v2/rgnn_mixed_c2_pw10_v2_best_val_f3.pt" #new
    REQUEST_GRAPH_THRESHOLD: float = 0.5 #new
    # 2026-08-04: NYC pair-pruner test - geographic must match how the loaded
    # checkpoint was trained (RequestGraphFeatureBuilder.add_features), and
    # num_layers must match the checkpoint's message_passing_steps for v1
    # checkpoints (not auto-detected from the state_dict, unlike v2). Defaults
    # keep the existing Li&Lim v2 default checkpoint's behavior unchanged.
    REQUEST_GRAPH_GEOGRAPHIC: bool = False
    REQUEST_GRAPH_NUM_LAYERS: int = 2
    # 2026-07-14: request-only pruner (RequestPruner) - separate from the
    # request-request pair pruner above, runs before it in TripHandler.run()
    # and reduces the request count itself, not just which pairs may share a
    # trip. See rtv_solver/pipeline/request_pruner.py for the model/class,
    # and rtv_solver/pipeline/train_request_pruner.py + sweep_request_pruner.py
    # for how the default checkpoint below was selected (ROC-AUC-ranked,
    # 5-seed sweep, 13 pre-solve-computable features).
    USE_REQUEST_PRUNER: bool = False
    REQUEST_PRUNER_MODEL_PATH: str = "outputs/request_pruner_mlp/request_pruner_mlp_h32_l1_d0p0_pw1p0/request_pruner_mlp_h32_l1_d0p0_pw1p0_best_val_f3.pt"
    REQUEST_PRUNER_THRESHOLD: float = 0.3
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
    # 2026-07-30: additive alongside ScoringMLP (the "mlp" default) - lets
    # training_loop.py/coaml_pipeline.py construct a CandidateScoringGNN
    # instead when no explicit `model=` is passed in, so both can be run
    # side by side for comparison without touching the existing MLP path.
    MODEL_TYPE: str = "mlp"  # "mlp" | "gnn"
    GNN_NUM_MESSAGE_PASSING_LAYERS: int = 1
    # 2026-08-02: aggregator ablation - "gcn" (GCNMeanLayer, Eq. 2), "mean"
    # (GraphSAGEMeanLayer, Algorithm 1), or "pool" (GraphSAGEPoolLayer, Eq. 3).
    # Ignored when MODEL_TYPE="mlp".
    GNN_AGGREGATOR: str = "gcn"  # "gcn" | "mean" | "pool"
    # 2026-08-03: HP-tuning grid stage 2 - threads the existing dropout
    # constructor parameter on CandidateScoringGNN through from the CLI
    # (previously always 0.0, never passed by build_scoring_model() callers).
    DROPOUT: float = 0.0
    # 2026-08-05: HP-tuning stage 2 - optional wandb logging/sweep support.
    USE_WANDB: bool = False
    WANDB_PROJECT: str = "rtv-solver-hp-sweep"
    # Expert label scoring
    IMITATION_SCORING_RULE: str = "legacy"
    # 2026-07-19: comma-separated list of extra instance stems (e.g. "lc207,lc208")
    # to ADD to training_loop.VALIDATION_FILES for this run, without touching
    # TRAINING_FILES or the module-level defaults. Lets us evaluate on LC2/LR2/LRC2
    # test instances - never covered by the standard 6-file val set - without
    # silently changing what every other/future run validates on.
    EXTRA_VALIDATION_FILES: str = ""
    # 2026-07-19: same mechanism as EXTRA_VALIDATION_FILES but for training_loop.
    # TRAINING_FILES - adds instance stems to what the SIL scoring MLP actually
    # trains on (not just evaluates on). See EXTRA_VALIDATION_FILES comment and
    # conversation "aber ist das alte Modell falsch..." for why this matters:
    # without it, any class-2 evaluation gap is confounded with SIL never having
    # trained on class-2 patterns at all, not attributable to the pruner.
    EXTRA_TRAINING_FILES: str = ""
    # 2026-07-26: unlike EXTRA_*_FILES (additive), these REPLACE training_loop's
    # standard TRAINING_FILES/VALIDATION_FILES entirely when non-empty - needed
    # for a "class-2-only" training variant, which a purely additive flag can't
    # express (no way to "add" the class-1 files' absence).
    OVERRIDE_TRAINING_FILES: str = ""
    OVERRIDE_VALIDATION_FILES: str = ""

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
            VAL_INPUT_FILE = getattr(args, "val_input_file", "") or "",
            COAML_MODEL_WEIGHTS = getattr(args, "coaml_model_weights", "") or "",
            COAML_SOLVE_MODE = getattr(args, "coaml_solve_mode", "train") or "train",
            SERVER_URL = getattr(args, "server_url", None), # args.server_url,
            MODE = args.mode,
            USE_REQUEST_GRAPH_PRUNER=cls.str_to_bool(
                getattr(args, "use_request_graph_pruner", "False")
            ),
            # 2026-07-07: fallback default kept in sync with the class default above
            # (new v2 pw10 checkpoint) so an omitted --request_graph_model_path arg
            # doesn't silently fall back to the incompatible old v1 checkpoint.
            REQUEST_GRAPH_MODEL_PATH=getattr(
                args,
                "request_graph_model_path",
                "outputs/models_v2_gnnv2/rgnn_mixed_c2_pw10_v2/rgnn_mixed_c2_pw10_v2_best_val_f3.pt",
            ),
            REQUEST_GRAPH_THRESHOLD=getattr(
                args, "request_graph_threshold", 0.5
            ),
            REQUEST_GRAPH_GEOGRAPHIC=cls.str_to_bool(
                getattr(args, "request_graph_geographic", "False")
            ),
            REQUEST_GRAPH_NUM_LAYERS=getattr(
                args, "request_graph_num_layers", 2
            ),
            # 2026-07-14: request-only pruner flags, mirrors the request-graph
            # (pair) pruner flags above - see USE_REQUEST_PRUNER comment in
            # the dataclass fields for context.
            USE_REQUEST_PRUNER=cls.str_to_bool(
                getattr(args, "use_request_pruner", "False")
            ),
            REQUEST_PRUNER_MODEL_PATH=getattr(
                args,
                "request_pruner_model_path",
                "outputs/request_pruner_mlp/request_pruner_mlp_h32_l1_d0p0_pw1p0/request_pruner_mlp_h32_l1_d0p0_pw1p0_best_val_f3.pt",
            ),
            REQUEST_PRUNER_THRESHOLD=getattr(
                args, "request_pruner_threshold", 0.3
            ),
            EXTRA_VALIDATION_FILES=getattr(args, "extra_validation_files", "") or "",
            EXTRA_TRAINING_FILES=getattr(args, "extra_training_files", "") or "",
            OVERRIDE_TRAINING_FILES=getattr(args, "override_training_files", "") or "",
            OVERRIDE_VALIDATION_FILES=getattr(args, "override_validation_files", "") or "",
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
            # add new config item
            IMITATION_SCORING_RULE = getattr(
                args,
                "imitation_scoring_rule",
                "legacy",
            ),
            EPOCHS = getattr(args, "epochs", 1),
            LEARNING_RATE = getattr(args, "learning_rate", 1e-3),
            HIDDEN_DIM = getattr(args, "hidden_dim", 64),
            NUM_SAMPLES = getattr(args, "num_samples", 20),
            SIGMA = getattr(args, "sigma", 0.2),
            MODEL_TYPE = getattr(args, "model_type", "mlp") or "mlp",
            GNN_NUM_MESSAGE_PASSING_LAYERS = getattr(args, "gnn_num_message_passing_layers", 1),
            GNN_AGGREGATOR = getattr(args, "gnn_aggregator", "gcn") or "gcn",
            DROPOUT = getattr(args, "dropout", 0.0),
            USE_WANDB = cls.str_to_bool(getattr(args, "use_wandb", "False") or "False"),
            WANDB_PROJECT = getattr(args, "wandb_project", "rtv-solver-hp-sweep") or "rtv-solver-hp-sweep",
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