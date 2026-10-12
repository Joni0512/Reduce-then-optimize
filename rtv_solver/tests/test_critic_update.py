"""
2026-10-12: equivalence of critic_td_update_step (joint buffer scheme) with the ORIGINAL inline critic update loop body of
COAMLPipeline.solve_pdptw (coaml_pipeline.py "for _ in range(num_updates)" block), re-implemented here as the reference: same entries, same
initial weights -> identical weights/loss afterwards (single and twin critic). Plus the compressed actor buffer round trip.
"""
import copy
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rtv_solver.pipeline.actor_transition_buffer import ActorTransition, ActorTransitionBuffer
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.critic_update import critic_td_update_step
from rtv_solver.pipeline.episode_buffer import EpisodeStep
from rtv_solver.pipeline.replay_buffer import ReplayEntry
from rtv_solver.pipeline.td_target_builder import td_target_for_step
from rtv_solver.pipeline.match_graph_features import MatchGraphFeatureBuilder as M


def _step(n_req=3, n_veh=2):
    n = n_req + n_veh
    ei = torch.tensor([[i for i in range(n) for j in range(n) if i != j], [j for i in range(n) for j in range(n) if i != j]])
    return EpisodeStep(torch.randn(n_req, M.REQUEST_FEATURE_SIZE), torch.randn(n_veh, M.VEHICLE_FEATURE_SIZE), ei, 0.0)


def _entries(k=6):
    steps = [_step() for _ in range(k + 1)]
    return [ReplayEntry(step=steps[i], reward=float(i) / 3, next_step=steps[i + 1] if i < k - 1 else None) for i in range(k)]


def _reference_inline(entries, critic, opt, target, gamma, critic2=None, opt2=None, target2=None):
    # verbatim logic of the inline block in coaml_pipeline.py (td_bootstrap, train_critic=True)
    preds = [critic(e.step.request_features, e.step.vehicle_features, e.step.edge_index) for e in entries]
    targets = [torch.tensor(td_target_for_step(e.reward, e.next_step, target, gamma, target_critic2=target2), dtype=torch.float32) for e in entries]
    loss = torch.stack([F.mse_loss(q, t) for q, t in zip(preds, targets)]).mean()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0); opt.step()
    if critic2 is not None:
        preds2 = [critic2(e.step.request_features, e.step.vehicle_features, e.step.edge_index) for e in entries]
        loss2 = torch.stack([F.mse_loss(q, t) for q, t in zip(preds2, targets)]).mean()
        opt2.zero_grad(set_to_none=True); loss2.backward()
        torch.nn.utils.clip_grad_norm_(critic2.parameters(), max_norm=1.0); opt2.step()
    return loss.item()


def _same(a, b):
    return all(torch.allclose(x, y, atol=1e-7) for x, y in zip(a.parameters(), b.parameters()))


def test_critic_update_matches_inline_loop_single_and_twin():
    torch.manual_seed(0)
    c1, c2 = CriticGNN(aggregator="gat"), CriticGNN(aggregator="gat")
    t1, t2 = copy.deepcopy(c1), copy.deepcopy(c2)
    entries = _entries()
    for twin in (False, True):
        a1, a2 = copy.deepcopy(c1), copy.deepcopy(c2)
        b1, b2 = copy.deepcopy(c1), copy.deepcopy(c2)
        oa1, oa2 = torch.optim.Adam(a1.parameters(), lr=1e-3), torch.optim.Adam(a2.parameters(), lr=1e-3)
        ob1, ob2 = torch.optim.Adam(b1.parameters(), lr=1e-3), torch.optim.Adam(b2.parameters(), lr=1e-3)
        for _ in range(3):  # several steps: Adam state must match too
            la = critic_td_update_step(entries, a1, oa1, t1, 0.99, critic2=a2 if twin else None, critic_optimizer2=oa2 if twin else None, target_critic2=t2 if twin else None)
            lb = _reference_inline(entries, b1, ob1, t1, 0.99, critic2=b2 if twin else None, opt2=ob2 if twin else None, target2=t2 if twin else None)
            assert abs(la - lb) < 1e-6
        assert _same(a1, b1)
        if twin:
            assert _same(a2, b2)


def test_compressed_buffer_roundtrip_sampling_and_epoch_drop():
    marker = object()
    def tr(v):
        return ActorTransition(pipeline=marker, feature_tensor=torch.full((2, 3), float(v)), reject_vehicle_ids=[1], trip_costs=[v],
                               trip_list=[], single_trip_map={}, vehicle_to_trips_cost_map={}, trip_to_vehicle_cost_map={},
                               requests=[], vehicles={}, active_requests={}, current_time=float(v))
    buf = ActorTransitionBuffer(compress=True)
    for epoch in (1, 2, 3):
        buf.current_epoch = epoch
        for k in range(4):
            buf.add(tr(epoch * 10 + k))
    assert len(buf) == 12
    batch = buf.sample_with_replacement(50, random.Random(0))
    assert len(batch) == 50 and all(t.pipeline is marker for t in batch)
    assert all(t.feature_tensor[0, 0].item() == t.current_time for t in batch)  # content survives compression
    assert len({t.current_time for t in batch}) > 1
    assert buf.drop_epochs_before(3) == 8 and len(buf) == 4
    assert all(t.current_time >= 30 for t in buf.sample_with_replacement(20, random.Random(1)))
    assert sum(len(b) for b in buf.epoch_batches(3, random.Random(2))) == 4  # old path works on compressed items too
