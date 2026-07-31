import types

import pytest
import torch

from rtv_solver.pipeline.candidate_scoring_gnn import (
    CandidateGraph,
    CandidateConflictGraphBuilder,
    CandidateScoringGNN,
)
from rtv_solver.pipeline.feat_builder import FeatureBuilder
from rtv_solver.structure.trip_cost import TripCost


def _make_trip_cost(trip_no, vehicle_id, request_ids):
    """Minimal TripCost with just enough structure for the graph builder:
    vehicle_id and plan.trips (each element needing only a request_id
    attribute, per _extract_request_ids's use of getattr)."""
    plan_trips = [types.SimpleNamespace(request_id=rid) for rid in request_ids]
    plan = types.SimpleNamespace(trips=plan_trips)
    return TripCost(trip_no=trip_no, vehicle_id=vehicle_id, cost=0.0, sequence=[], plan=plan)


def _has_edge(graph: CandidateGraph, a: int, b: int) -> bool:
    edges = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    return (a, b) in edges and (b, a) in edges  # must be undirected (both directions present)


@pytest.fixture
def builder() -> CandidateConflictGraphBuilder:
    return CandidateConflictGraphBuilder()


@pytest.fixture
def mixed_trip_costs():
    """
    2 vehicles, 4 trip candidates, deliberately covering every conflict case:
      - tc0 (veh 0, req {1}) & tc1 (veh 0, req {2}): same vehicle -> connected
      - tc0 (veh 0, req {1}) & tc2 (veh 1, req {1}): same request -> connected
      - tc1 (veh 0, req {2}) & tc2 (veh 1, req {1}): different vehicle, disjoint requests -> NOT connected
      - tc2 (veh 1, req {1}) & tc3 (veh 1, req {3,4}): same vehicle -> connected (tc3 is a cardinality-2 candidate)
      - tc0/tc1 vs tc3: different vehicle, disjoint requests -> NOT connected
    """
    return [
        _make_trip_cost(0, vehicle_id=0, request_ids=[1]),
        _make_trip_cost(1, vehicle_id=0, request_ids=[2]),
        _make_trip_cost(2, vehicle_id=1, request_ids=[1]),
        _make_trip_cost(3, vehicle_id=1, request_ids=[3, 4]),
    ]


@pytest.fixture
def reject_vehicle_ids():
    return [0, 1]


@pytest.fixture
def mixed_graph(builder, mixed_trip_costs, reject_vehicle_ids):
    return builder.build(mixed_trip_costs, reject_vehicle_ids)


# ---------------------------------------------------------------------------
# CandidateConflictGraphBuilder
# ---------------------------------------------------------------------------

@pytest.mark.basic
def test_node_count_and_order(mixed_graph, mixed_trip_costs, reject_vehicle_ids):
    """Nodes must be [trip candidates..., reject actions...] in input order -
    this is the invariant the whole COAML pipeline (scores/ILP vars/y_star)
    depends on."""
    assert mixed_graph.num_nodes == len(mixed_trip_costs) + len(reject_vehicle_ids)
    assert mixed_graph.vehicle_ids == (0, 0, 1, 1, 0, 1)
    assert mixed_graph.is_reject == (False, False, False, False, True, True)
    assert mixed_graph.request_ids[:4] == (
        frozenset({1}), frozenset({2}), frozenset({1}), frozenset({3, 4}),
    )
    assert mixed_graph.request_ids[4] == frozenset()
    assert mixed_graph.request_ids[5] == frozenset()


@pytest.mark.basic
def test_same_vehicle_candidates_connected(mixed_graph):
    assert _has_edge(mixed_graph, 0, 1)  # tc0, tc1: both vehicle 0


@pytest.mark.basic
def test_same_request_across_vehicles_connected(mixed_graph):
    assert _has_edge(mixed_graph, 0, 2)  # tc0, tc2: both request 1


@pytest.mark.basic
def test_unrelated_candidates_not_connected(mixed_graph):
    assert not _has_edge(mixed_graph, 1, 2)  # tc1 (veh0,req2) vs tc2 (veh1,req1)
    assert not _has_edge(mixed_graph, 0, 3)  # tc0 (veh0,req1) vs tc3 (veh1,req{3,4})
    assert not _has_edge(mixed_graph, 1, 3)  # tc1 (veh0,req2) vs tc3 (veh1,req{3,4})


@pytest.mark.basic
def test_cardinality_two_candidate_wires_both_requests(builder):
    """A candidate covering 2 requests must connect to single-request
    candidates sharing EITHER of its own requests."""
    trip_costs = [
        _make_trip_cost(0, vehicle_id=0, request_ids=[3, 4]),  # shared trip, veh 0
        _make_trip_cost(1, vehicle_id=1, request_ids=[3]),     # different veh, shares req 3
        _make_trip_cost(2, vehicle_id=2, request_ids=[4]),     # different veh, shares req 4
        _make_trip_cost(3, vehicle_id=3, request_ids=[9]),     # unrelated
    ]
    graph = builder.build(trip_costs, reject_vehicle_ids=[])
    assert _has_edge(graph, 0, 1)
    assert _has_edge(graph, 0, 2)
    assert not _has_edge(graph, 0, 3)
    assert not _has_edge(graph, 1, 2)  # veh1/req3 vs veh2/req4: unrelated to each other


@pytest.mark.basic
def test_reject_node_connected_only_to_same_vehicle(mixed_graph):
    """Reject nodes (indices 4, 5) must connect to their own vehicle's trip
    candidates, and not to other vehicles or to each other."""
    assert _has_edge(mixed_graph, 4, 0)  # reject(veh0) <-> tc0(veh0)
    assert _has_edge(mixed_graph, 4, 1)  # reject(veh0) <-> tc1(veh0)
    assert not _has_edge(mixed_graph, 4, 2)
    assert not _has_edge(mixed_graph, 4, 3)

    assert _has_edge(mixed_graph, 5, 2)  # reject(veh1) <-> tc2(veh1)
    assert _has_edge(mixed_graph, 5, 3)  # reject(veh1) <-> tc3(veh1)
    assert not _has_edge(mixed_graph, 5, 0)
    assert not _has_edge(mixed_graph, 5, 1)

    assert not _has_edge(mixed_graph, 4, 5)  # different vehicles, no requests on either side


@pytest.mark.basic
def test_edge_index_shape_and_no_self_loops(mixed_graph):
    assert mixed_graph.edge_index.ndim == 2
    assert mixed_graph.edge_index.shape[0] == 2
    source, target = mixed_graph.edge_index[0], mixed_graph.edge_index[1]
    assert not (source == target).any(), (
        "builder must not add self-loops (handled inside the GNN layer via self_linear)"
    )


@pytest.mark.basic
def test_no_edges_for_fully_independent_candidates(builder):
    trip_costs = [
        _make_trip_cost(0, vehicle_id=0, request_ids=[1]),
        _make_trip_cost(1, vehicle_id=1, request_ids=[2]),
    ]
    graph = builder.build(trip_costs, reject_vehicle_ids=[])
    assert graph.edge_index.numel() == 0
    assert graph.edge_index.shape == (2, 0)


@pytest.mark.basic
def test_missing_plan_raises(builder):
    bad = TripCost(trip_no=0, vehicle_id=0, cost=0.0, sequence=[], plan=None)
    with pytest.raises(ValueError, match="plan.trips"):
        builder.build([bad], reject_vehicle_ids=[])


@pytest.mark.basic
def test_missing_request_id_raises(builder):
    plan = types.SimpleNamespace(trips=[types.SimpleNamespace(no_request_id_here=1)])
    bad = TripCost(trip_no=0, vehicle_id=0, cost=0.0, sequence=[], plan=plan)
    with pytest.raises(ValueError, match="request_id"):
        builder.build([bad], reject_vehicle_ids=[])


# ---------------------------------------------------------------------------
# CandidateScoringGNN
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_dim() -> int:
    # Current (not yet redesigned) FeatureBuilder - matches testing the GNN
    # against the existing feature set first, before swapping to
    # feat_builder_new.py's feature set.
    return FeatureBuilder.FEATURE_SIZE


@pytest.fixture
def hidden_dim() -> int:
    return 32


@pytest.fixture
def gnn_model(feature_dim, hidden_dim) -> CandidateScoringGNN:
    torch.manual_seed(0)
    return CandidateScoringGNN(feature_dim=feature_dim, hidden_dim=hidden_dim)


@pytest.mark.basic
def test_gnn_output_shape_matches_graph(gnn_model, mixed_graph, feature_dim):
    x = torch.randn(mixed_graph.num_nodes, feature_dim)
    scores = gnn_model(x, mixed_graph.edge_index)
    assert scores.shape == (mixed_graph.num_nodes,)


@pytest.mark.basic
def test_gnn_handles_empty_edge_index(gnn_model, feature_dim):
    """No conflicts at all (e.g. isolated candidates) must not divide by zero
    or produce NaN - covers the neighbour_count.clamp_min(1.0) guard."""
    x = torch.randn(3, feature_dim)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    scores = gnn_model(x, edge_index)
    assert scores.shape == (3,)
    assert not torch.isnan(scores).any()


@pytest.mark.basic
def test_gnn_raw_scores_unbounded(gnn_model, feature_dim, mixed_graph):
    """No sigmoid/softmax - matches ScoringMLP's contract for the ILP/FY-loss."""
    torch.manual_seed(1)
    x = torch.randn(mixed_graph.num_nodes, feature_dim) * 5.0
    scores = gnn_model(x, mixed_graph.edge_index)
    assert ((scores > 1.0) | (scores < 0.0)).any()


@pytest.mark.basic
def test_gnn_backward_pass_runs(gnn_model, feature_dim, mixed_graph):
    x = torch.randn(mixed_graph.num_nodes, feature_dim)
    scores = gnn_model(x, mixed_graph.edge_index)
    scores.sum().backward()
    for name, param in gnn_model.named_parameters():
        assert param.grad is not None, f"gradient missing for '{name}'"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for '{name}'"


@pytest.mark.basic
def test_gnn_return_embeddings(gnn_model, feature_dim, hidden_dim, mixed_graph):
    x = torch.randn(mixed_graph.num_nodes, feature_dim)
    scores, embeddings = gnn_model(x, mixed_graph.edge_index, return_embeddings=True)
    assert scores.shape == (mixed_graph.num_nodes,)
    assert embeddings.shape == (mixed_graph.num_nodes, hidden_dim)


@pytest.mark.basic
def test_gnn_feature_dim_mismatch_raises(gnn_model, feature_dim, mixed_graph):
    x = torch.randn(mixed_graph.num_nodes, feature_dim + 1)
    with pytest.raises(ValueError, match="Expected"):
        gnn_model(x, mixed_graph.edge_index)


@pytest.mark.basic
def test_requires_graph_flag():
    """coaml_pipeline.py can branch on this to decide whether to build a
    CandidateGraph - see the REVIEW comment in candidate_scoring_gnn.py: this
    branch doesn't exist yet, but the flag it would check is here."""
    assert CandidateScoringGNN.requires_graph is True


@pytest.mark.basic
def test_num_message_passing_layers_must_be_positive(feature_dim):
    with pytest.raises(ValueError):
        CandidateScoringGNN(feature_dim=feature_dim, hidden_dim=16, num_message_passing_layers=0)


@pytest.mark.basic
def test_two_message_passing_layers_still_matches_graph(feature_dim, mixed_graph):
    """Sanity check for the 'try 2 layers' suggestion in the review comment -
    2-hop propagation must not break shapes or introduce NaNs."""
    torch.manual_seed(2)
    model = CandidateScoringGNN(feature_dim=feature_dim, hidden_dim=16, num_message_passing_layers=2)
    x = torch.randn(mixed_graph.num_nodes, feature_dim)
    scores = model(x, mixed_graph.edge_index)
    assert scores.shape == (mixed_graph.num_nodes,)
    assert not torch.isnan(scores).any()
