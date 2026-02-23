import dataclasses
import json
import random
import numpy as np
import torch
import os

from pathlib import Path

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import Stats

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def save_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=json_default)

def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Stats):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def set_seed(seed: int = 42, debug=False):
    """
    Sets the seed for the random number generators to make sure it is always aligned and reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    if debug: # slight performance penalty for deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)