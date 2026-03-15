from copy import deepcopy
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.util.helper import load_json, save_json


def load_payload_from_file(input_path: str | Path) -> dict:
    """
    Load a payload from JSON or pickle.
    """
    path = Path(input_path)
    extension = path.suffix.lower()

    if extension == ".json":
        return load_json(path)
    if extension in {".pkl", ".pickle"}:
        with open(path, "rb") as file:
            return pickle.load(file)

    raise ValueError(f"Unsupported extension: {extension} for file {path}")


def add_node_ids_to_payload(payload: dict) -> dict:
    """
    Populate node ids on depot, requests and driver manifests.
    """
    depot = payload["depot"]
    depot["pt"]["node_id"] = NetworkHandler.get_next_node_id(
        depot["pt"]["lat"], depot["pt"]["lon"]
    )

    for request in payload["requests"]:
        request["pickup_pt"]["node_id"] = NetworkHandler.get_next_node_id(
            request["pickup_pt"]["lat"], request["pickup_pt"]["lon"]
        )
        request["dropoff_pt"]["node_id"] = NetworkHandler.get_next_node_id(
            request["dropoff_pt"]["lat"], request["dropoff_pt"]["lon"]
        )

    for driver_run in payload["driver_runs"]:
        state_loc = driver_run["state"]["loc"]
        state_loc["node_id"] = NetworkHandler.get_next_node_id(
            state_loc["lat"], state_loc["lon"]
        )
        for stop in driver_run["manifest"]:
            if "loc" in stop:
                stop["loc"]["node_id"] = NetworkHandler.get_next_node_id(
                    stop["loc"]["lat"], stop["loc"]["lon"]
                )

    return payload


def build_travel_time_matrix_payload(
    payload: dict,
    server_url: str = "http://127.0.0.1:5001/",
    euclidean: bool = False,
    mutate: bool = False,
) -> dict:
    """
    Enrich a payload with node ids and travel_time_matrix.
    """
    payload_out = payload if mutate else deepcopy(payload)
    NetworkHandler.init_from_source(server_url=server_url, euclidean=euclidean)
    add_node_ids_to_payload(payload_out)

    travel_time_matrix, no_of_nodes, _, _ = NetworkHandler.initialize_travel_time_matrix()
    travel_time_matrix_np = np.frombuffer(travel_time_matrix, dtype=np.float64).reshape(
        (int(no_of_nodes.value), int(no_of_nodes.value))
    )
    payload_out["travel_time_matrix"] = travel_time_matrix_np.tolist()
    return payload_out


def export_payload(payload: dict, output_dir: str | Path, file_stem: str) -> dict[str, Path]:
    """
    Export processed payload as JSON and pickle.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / f"{file_stem}.json"
    pkl_path = target_dir / f"{file_stem}.pkl"

    save_json(payload, json_path)
    with open(pkl_path, "wb") as file:
        pickle.dump(payload, file)

    return {"json": json_path, "pickle": pkl_path}


def process_payload_file(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    server_url: str = "http://127.0.0.1:5001/",
    euclidean: bool = False,
) -> tuple[dict, dict[str, Path] | None]:
    """
    Full pipeline: load file -> build matrix -> optionally export.
    """
    input_file = Path(input_path)
    payload = load_payload_from_file(input_file)
    processed_payload = build_travel_time_matrix_payload(
        payload=payload, server_url=server_url, euclidean=euclidean
    )

    export_paths = None
    if output_dir is not None:
        export_paths = export_payload(
            payload=processed_payload,
            output_dir=output_dir,
            file_stem=input_file.stem,
        )

    return processed_payload, export_paths


def process_payload_files(
    input_directory: str | Path,
    file_names: Iterable[str],
    output_dir: str | Path | None = None,
    server_url: str = "http://127.0.0.1:5001/",
    euclidean: bool = False,
) -> dict[str, tuple[dict, dict[str, Path] | None]]:
    """
    Batch wrapper around process_payload_file.
    """
    input_dir = Path(input_directory)
    results: dict[str, tuple[dict, dict[str, Path] | None]] = {}

    for file_name in file_names:
        input_path = input_dir / file_name
        if not input_path.is_file():
            print(f"Skipping {file_name}, not found in input directory.")
            continue

        print(f"Processing file: {file_name}")
        results[file_name] = process_payload_file(
            input_path=input_path,
            output_dir=output_dir,
            server_url=server_url,
            euclidean=euclidean,
        )

    return results


if __name__ == "__main__":
    INPUT_DIRECTORY = Path("inputs/test_nc")
    OUTPUT_DIRECTORY = INPUT_DIRECTORY / "ttm" / "test"
    FILES_TO_PROCESS = ["test_10r_1v_repeat6_simple.pkl"]

    process_payload_files(
        input_directory=INPUT_DIRECTORY,
        file_names=FILES_TO_PROCESS,
        output_dir=OUTPUT_DIRECTORY,
    )