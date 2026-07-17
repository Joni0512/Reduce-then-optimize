"""
2026-07-16: standalone runner for the COAML "OPT" baseline (solution
projection with rolling horizon, see COAMLPipeline.solve_iteration mode=
"optimal") for exactly ONE instance and ONE window config.

Why this exists instead of just using `main.py --mode coaml --input_dir ...`:
that path always goes through COAMLTrainingLoop, which (a) requires at least
EPOCHS=1 full training epoch over all 24 hardcoded TRAINING_FILES plus a
"eval" pass over VALIDATION_FILES (config.enforce_constraints asserts
EPOCHS >= 1), and (b) computes OPT for all 6 hardcoded VALIDATION_FILES, not
a single instance of choice. This script instead calls the exact same
mechanism the training loop uses for its OPT precompute
(training_loop.py:145-155: COAMLPipeline(...).solve_pdptw(payload,
mode="optimal")) directly, for one instance/window, with no training at all.

USE_REQUEST_PRUNER / USE_REQUEST_GRAPH_PRUNER are both off by default here,
per the current ask: check the OPT numbers without the pruner first.
"""
import argparse
import json
import time
from pathlib import Path

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.structure.config import Config
from rtv_solver.training_loop import _save_validation_results
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = REPO_ROOT / "solutions" / "li_lim" / "manifests"


def run_opt_single_instance(
    instance: str,
    batch_interval: int,
    step_size: int,
    max_cardinality: int = 2,
    seed: int = 42,
    output_dir: Path | None = None,
    use_request_pruner: bool = False,
    request_pruner_model_path: str | None = None,
    request_pruner_threshold: float = 0.3,
    use_request_graph_pruner: bool = False,
    request_graph_threshold: float | None = None,
) -> Path:
    input_path = MANIFEST_DIR / f"{instance}.json"
    if not input_path.exists():
        raise FileNotFoundError(f"No manifest for instance '{instance}': {input_path}")

    if output_dir is None:
        # 2026-07-16: tag the dir with which pruners were active AND their
        # thresholds, so a threshold-0.3 sweep doesn't silently overwrite a
        # threshold-0.5 sweep's results for the same instance/variant (found
        # the hard way: the first version of this only tagged by variant,
        # which was fine for the single-threshold spot checks but collided
        # the moment a second threshold value was tested).
        def _fmt_t(t: float) -> str:
            return str(t).replace(".", "p")

        effective_graph_threshold = (
            request_graph_threshold if request_graph_threshold is not None else Config.REQUEST_GRAPH_THRESHOLD
        )
        if use_request_pruner and use_request_graph_pruner:
            variant = f"both_pruners_req{_fmt_t(request_pruner_threshold)}_pair{_fmt_t(effective_graph_threshold)}"
        elif use_request_pruner:
            variant = f"request_pruner_t{_fmt_t(request_pruner_threshold)}"
        elif use_request_graph_pruner:
            variant = f"pair_pruner_t{_fmt_t(effective_graph_threshold)}"
        else:
            variant = "baseline"
        output_dir = (
            REPO_ROOT / "outputs" / "opt_single_instance"
            / f"bi{batch_interval}_ss{step_size}" / instance / variant
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        OUTPUT_DIR=output_dir,
        MODE="coaml",
        MAX_CARDINALITY=max_cardinality,
        LARGEST_TSP=max_cardinality * 4,
        BATCH_INTERVAL=batch_interval,
        STEP_SIZE=step_size,
        SEED=seed,
        USE_REQUEST_PRUNER=use_request_pruner,
        REQUEST_PRUNER_MODEL_PATH=request_pruner_model_path or Config.REQUEST_PRUNER_MODEL_PATH,
        REQUEST_PRUNER_THRESHOLD=request_pruner_threshold,
        USE_REQUEST_GRAPH_PRUNER=use_request_graph_pruner,
        REQUEST_GRAPH_THRESHOLD=request_graph_threshold if request_graph_threshold is not None else Config.REQUEST_GRAPH_THRESHOLD,
    )
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)

    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)

    pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path)
    # 2026-07-16: wall-clock runtime of the actual solve, for the pruner-vs-runtime
    # comparison the user asked for - _save_validation_results() doesn't measure this
    # itself (training_loop.py hardcodes total_time=0.0, it's a training-loop artifact,
    # not a per-run timer), so it's measured here and injected into results.json after
    # the fact instead of touching the shared helper.
    start_time = time.time()
    driver_runs = pipeline.solve_pdptw(cleared_payload, mode="optimal")
    runtime_seconds = time.time() - start_time

    _save_validation_results(config, cleared_payload, driver_runs)

    results_path = output_dir / "results.json"
    with open(results_path) as f:
        results = json.load(f)
    results["stats"]["runtime_seconds"] = runtime_seconds
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"OPT run for {instance} @ bi={batch_interval}/ss={step_size} -> {results_path} ({runtime_seconds:.1f}s)")
    return results_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the COAML OPT baseline for one instance/window config.")
    parser.add_argument("--instance", type=str, default="lc108")
    parser.add_argument("--batch_interval", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--max_cardinality", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_request_pruner", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--request_pruner_model_path", type=str, default=None)
    parser.add_argument("--request_pruner_threshold", type=float, default=0.3)
    parser.add_argument("--use_request_graph_pruner", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--request_graph_threshold", type=float, default=None)
    args = parser.parse_args()

    run_opt_single_instance(
        instance=args.instance,
        batch_interval=args.batch_interval,
        step_size=args.step_size,
        max_cardinality=args.max_cardinality,
        seed=args.seed,
        use_request_pruner=args.use_request_pruner == "True",
        request_pruner_model_path=args.request_pruner_model_path,
        request_pruner_threshold=args.request_pruner_threshold,
        use_request_graph_pruner=args.use_request_graph_pruner == "True",
        request_graph_threshold=args.request_graph_threshold,
    )
