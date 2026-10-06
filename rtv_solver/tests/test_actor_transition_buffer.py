"""Tests for the opt-in "collect, then update" SRL actor scheme (2026-10-06): ActorTransitionBuffer
(rtv_solver/pipeline/actor_transition_buffer.py) and COAMLPipeline's collect branch /
compute_srl_actor_loss_for_transition. The old per-window path is covered by the equivalence test."""
import copy
import random
from pathlib import Path

import pytest
import torch

from rtv_solver.pipeline.actor_transition_buffer import ActorTransition, ActorTransitionBuffer


def _dummy_transition(i: int) -> ActorTransition:
    return ActorTransition(
        pipeline=None, feature_tensor=torch.zeros(1, 1), reject_vehicle_ids=[], trip_costs=[], trip_list=[],
        single_trip_map={}, vehicle_to_trips_cost_map={}, trip_to_vehicle_cost_map={}, requests=[],
        vehicles={}, active_requests={}, current_time=float(i),
    )


@pytest.mark.basic
def test_epoch_batches_use_every_transition_exactly_once():
    buf = ActorTransitionBuffer()
    for i in range(37):
        buf.add(_dummy_transition(i))
    batches = buf.epoch_batches(16, random.Random(0))
    assert [len(b) for b in batches] == [16, 16, 5]
    seen = sorted(t.current_time for b in batches for t in b)
    assert seen == [float(i) for i in range(37)]


@pytest.mark.basic
def test_epoch_batches_are_shuffled_but_reproducible():
    buf = ActorTransitionBuffer()
    for i in range(40):
        buf.add(_dummy_transition(i))
    order_a = [t.current_time for b in buf.epoch_batches(8, random.Random(1)) for t in b]
    order_b = [t.current_time for b in buf.epoch_batches(8, random.Random(1)) for t in b]
    assert order_a == order_b
    assert order_a != sorted(order_a)


@pytest.mark.basic
def test_capacity_none_keeps_all_and_clear_empties():
    buf = ActorTransitionBuffer()
    for i in range(500):
        buf.add(_dummy_transition(i))
    assert len(buf) == 500
    buf.clear()
    assert len(buf) == 0


@pytest.mark.basic
def test_capacity_int_is_fifo_ring():
    buf = ActorTransitionBuffer(capacity=3)
    for i in range(5):
        buf.add(_dummy_transition(i))
    assert sorted(t.current_time for b in buf.epoch_batches(10, random.Random(0)) for t in b) == [2.0, 3.0, 4.0]


@pytest.mark.basic
def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        ActorTransitionBuffer(capacity=0)
    with pytest.raises(ValueError):
        ActorTransitionBuffer().epoch_batches(0, random.Random(0))


def _build_srl_pipeline(buffer):
    """Small real SRL setup on lc101 (random actor, twin-free GAT critic), mirrors srl_training_loop."""
    from rtv_solver.pipeline import feat_builder as fb
    from rtv_solver.coaml_pipeline import COAMLPipeline
    from rtv_solver.handlers.payload_parser import PayloadParser
    from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
    from rtv_solver.pipeline.critic_gnn import CriticGNN
    from rtv_solver.pipeline.srl_training_loop import MANIFEST_DIR
    from rtv_solver.structure.config import Config

    import tempfile
    out = Path(tempfile.mkdtemp())
    config = Config(OUTPUT_DIR=out, MODE="coaml", BATCH_INTERVAL=200, STEP_SIZE=100, SEED=42, MAX_CARDINALITY=2, KEEP_ACTIVE=False)
    input_path = MANIFEST_DIR / "lc101.json"
    payload = PayloadParser.clear_vehicle_manifests(PayloadParser.load_input_data(input_path))
    torch.manual_seed(0)
    model = build_scoring_model("mlp", feature_dim=fb.FeatureBuilder.FEATURE_SIZE, hidden_dim=64)
    critic = CriticGNN(aggregator="gat")
    from rtv_solver.pipeline.replay_buffer import ReplayBuffer
    pipeline = COAMLPipeline(
        config, payload, imitation_solution_path=input_path, model=model, critic=critic,
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3), target_critic=copy.deepcopy(critic),
        critic_target_mode="td_bootstrap", replay_buffer=ReplayBuffer(capacity=40),
        actor_transition_buffer=buffer,
    )
    return pipeline, payload


@pytest.mark.integration
@pytest.mark.slow
def test_collect_stores_windows_without_touching_actor_and_loss_matches_inline():
    # --- old inline path: loss of the first window, RNG seeded right before the rollout ---
    old_pipeline, payload = _build_srl_pipeline(buffer=None)
    old_opt = torch.optim.Adam(old_pipeline.model.parameters(), lr=0.0)  # lr=0: weights stay put, we only read the loss
    torch.manual_seed(123)
    old_pipeline.solve_pdptw(payload, mode="srl", optimizer=old_opt, train_critic=False)
    old_first_loss = next(l for l in old_pipeline.loss_history if l is not None)

    # --- new collect path: same weights (same torch seed in builder), actor must not move ---
    buf = ActorTransitionBuffer()
    new_pipeline, payload = _build_srl_pipeline(buffer=buf)
    weights_before = copy.deepcopy(new_pipeline.model.state_dict())
    new_pipeline.solve_pdptw(payload, mode="srl", optimizer=None, train_critic=False)
    assert len(buf) > 0
    assert all(l is None for l in new_pipeline.loss_history)  # no loss during the rollout
    for k, v in new_pipeline.model.state_dict().items():
        assert torch.equal(v, weights_before[k])

    # update-phase loss of the first stored window, same seed placement as the old path
    first = buf.epoch_batches(len(buf), random.Random(0))[0]
    first = min(first, key=lambda t: t.current_time)
    torch.manual_seed(123)
    new_first_loss = new_pipeline.compute_srl_actor_loss_for_transition(first).item()
    assert new_first_loss == pytest.approx(old_first_loss, rel=1e-5, abs=1e-6)

    # and the gradient flows into the actor
    new_pipeline.model.zero_grad()
    new_pipeline.compute_srl_actor_loss_for_transition(first).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in new_pipeline.model.parameters())
