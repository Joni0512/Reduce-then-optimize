"""
2026-09-14: first smoke test of the RHO-outcome-advantage method (Option 2
from the srl deck's "Training Signal Design" slide) - see chat. NOT the
same as SIL/behavior cloning: the actor never sees RHO's per-step actions,
only a single episode-level advantage (actor_service_rate - rho_service_rate)
computed after a full rollout, broadcast (uniform, Option A) over the
episode's buffered raw actor scores, and trained with a policy-gradient-
style loss = -advantage * score (NOT MSE - score and advantage are not the
same unit, see chat).

RHO baseline = COAMLPipeline(mode="offline") (pure cost-minimization ILP, no
NN influence - same objective RHO itself optimizes), run at a DIFFERENT
horizon than the actor on purpose for this test: actor at bi200/ss100, RHO
baseline at bi400/ss100 (see chat - explicitly testing whether the actor,
at a shorter/more frequent re-optimization horizon, can beat a RHO baseline
that only sees a longer, less frequent one).

This is a QUICK LOCAL SMOKE TEST (few epochs, ONE instance) to check the
mechanism runs and produces sane numbers - not a full sweep/training run.
"""
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

import random
from pathlib import Path

import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import StatsParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = REPO_ROOT / "solutions" / "li_lim" / "manifests"

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

ACTOR_BATCH_INTERVAL = 200
ACTOR_STEP_SIZE = 100
RHO_BATCH_INTERVAL = 400
RHO_STEP_SIZE = 100
SEED = 42


def _service_rate(config: Config, cleared_payload: dict, driver_runs: list) -> float:
    full_payload_object = PayloadParser.get_payload_object(
        cleared_payload, dwell_pickup_default=config.DWELL_PICKUP, dwell_alight_default=config.DWELL_ALIGHT, online=False,
    )
    all_requests = RequestHandler(full_payload_object.requests, config=config).get_all_requests()
    stats_payload = {
        PayloadKeys.DEPOT: cleared_payload[PayloadKeys.DEPOT],
        PayloadKeys.REQUESTS: cleared_payload[PayloadKeys.REQUESTS],
        PayloadKeys.DRIVERS: driver_runs,
        PayloadKeys.TIME_MATRIX: cleared_payload.get(PayloadKeys.TIME_MATRIX, None),
    }
    _, episode_stats, _ = StatsParser(config, payload=stats_payload).evaluate(stats_payload)
    num_requests = len(all_requests)
    num_serviced = len(episode_stats.serviced_requests)
    return num_serviced / num_requests if num_requests > 0 else 0.0


def run_rho_baseline(instance: str, output_dir: Path, seed: int) -> float:
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    config = Config(OUTPUT_DIR=output_dir / "rho_baseline", MODE="coaml", BATCH_INTERVAL=RHO_BATCH_INTERVAL, STEP_SIZE=RHO_STEP_SIZE, SEED=seed)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)
    pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path)
    driver_runs = pipeline.solve_pdptw(cleared_payload, mode="offline")
    return _service_rate(config, cleared_payload, driver_runs)


def run_actor_episode(instance: str, model: torch.nn.Module | None, output_dir: Path, seed: int, epoch: int) -> tuple[float, torch.nn.Module, list[torch.Tensor]]:
    input_path = MANIFEST_DIR / f"{instance}.json"
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    config = Config(OUTPUT_DIR=output_dir / "actor" / f"epoch_{epoch}", MODE="coaml", BATCH_INTERVAL=ACTOR_BATCH_INTERVAL, STEP_SIZE=ACTOR_STEP_SIZE, SEED=seed)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)

    outcome_advantage_buffer: list[torch.Tensor] = []
    pipeline = COAMLPipeline(
        config, cleared_payload, model=model, imitation_solution_path=input_path,
        outcome_advantage_buffer=outcome_advantage_buffer,
        outcome_advantage_sigma=0.2,
    )
    if model is None:
        pipeline.load_model_weights(ACTOR_CHECKPOINT)
    driver_runs = pipeline.solve_pdptw(cleared_payload, mode="eval")
    service_rate = _service_rate(config, cleared_payload, driver_runs)
    return service_rate, pipeline.model, outcome_advantage_buffer


def main(instance: str = "lc101", epochs: int = 20) -> None:
    output_dir = REPO_ROOT / "outputs" / "srl_rho_outcome_advantage_smoketest" / instance
    rng = random.Random(SEED)

    print(f"=== RHO baseline (bi{RHO_BATCH_INTERVAL}/ss{RHO_STEP_SIZE}) on {instance} ===")
    rho_service_rate = run_rho_baseline(instance, output_dir, SEED)
    print(f"RHO service_rate = {rho_service_rate:.4f}")

    model = None
    actor_optimizer = None

    for epoch in range(1, epochs + 1):
        actor_service_rate, model, buffer = run_actor_episode(instance, model, output_dir, SEED, epoch)
        advantage = actor_service_rate - rho_service_rate
        print(f"epoch {epoch}: actor_service_rate={actor_service_rate:.4f}  rho_service_rate={rho_service_rate:.4f}  advantage={advantage:+.4f}  buffer_size={len(buffer)}")

        if not buffer:
            print(f"epoch {epoch}: empty buffer, skipping optimizer step")
            continue

        if actor_optimizer is None:
            actor_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Option A (uniform broadcast): every buffered score gets the SAME
        # advantage target. loss = -advantage * score (policy-gradient-style
        # direction/magnitude signal, NOT MSE - see module docstring).
        #
        # 2026-09-14 bugfix: this buffer is all ONE stored graph from a
        # single forward pass (the actor rollout above) - a SECOND
        # backward()/step() on it after the first step() has already
        # mutated the model's parameters in-place raises "one of the
        # variables needed for gradient computation has been modified by an
        # inplace operation" (confirmed by running this - see chat). So:
        # exactly ONE backward()/step() per epoch, over the WHOLE buffer at
        # once (not several mini-batch steps reusing the same graph). A real
        # multi-update-per-epoch version would need to re-run the forward
        # pass (or a fresh rollout) between steps, not just resample the
        # buffer - out of scope for this smoke test.
        scores = torch.stack(buffer)
        loss = (-advantage * scores).mean()
        actor_optimizer.zero_grad()
        loss.backward()
        actor_optimizer.step()
        print(f"epoch {epoch}: 1 optimizer step over all {len(buffer)} buffered scores, loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
