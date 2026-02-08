import json
import pickle

from pathlib import Path

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import Stats

def save_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=json_default)

def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Stats):
        return obj.to_dict()
    raise TypeError(f"Type {type(obj)} not serializable")

def load_input_data(input_file: Path):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return PayloadParser.normalize_to_canonical(data)