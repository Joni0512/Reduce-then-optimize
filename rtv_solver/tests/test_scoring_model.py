import pytest
import torch
import torch.nn as nn

from rtv_solver.pipeline import ScoringMLP


@pytest.fixture
def feature_dim() -> int:
    return 8

@pytest.fixture
def hidden_dim() -> int:
    return 32

@pytest.fixture
def model(feature_dim, hidden_dim) -> ScoringMLP:
    torch.manual_seed(0)
    return ScoringMLP(feature_dim=feature_dim, hidden_dim=hidden_dim)


@pytest.mark.basic
def test_output_shape_single_instance(model, feature_dim):
    """Single instance: (num_items, feature_dim) → (num_items,)"""
    num_items = 7
    x = torch.randn(num_items, feature_dim)
    scores = model(x)
    assert scores.shape == (num_items,)

@pytest.mark.basic
def test_output_shape_batch(model, feature_dim):
    """Mini-batch: (batch, num_items, feature_dim) → (batch, num_items)"""
    batch_size, num_items = 4, 7
    x = torch.randn(batch_size, num_items, feature_dim)
    scores = model(x)
    assert scores.shape == (batch_size, num_items)

@pytest.mark.basic
def test_variable_item_count(model, feature_dim):
    """Same model must handle different num_items without reinitialization."""
    for num_items in [1, 5, 12, 20]:
        x = torch.randn(num_items, feature_dim)
        scores = model(x)
        assert scores.shape == (num_items,), f"unexpected shape for num_items={num_items}"

@pytest.mark.basic
def test_raw_scores_can_be_negative(model, feature_dim):
    """Output must be raw (unbounded) — no sigmoid or clamping applied."""
    torch.manual_seed(42)
    x = torch.randn(50, feature_dim)
    scores = model(x)
    has_negative = (scores < 0).any().item()
    has_positive = (scores > 0).any().item()
    assert has_negative, "expected at least some negative scores (no output activation)"
    assert has_positive, "expected at least some positive scores"

@pytest.mark.basic
def test_scores_not_bounded_to_zero_one(model, feature_dim):
    """Scores must not be bounded to [0, 1] — sigmoid would mix up the results"""
    torch.manual_seed(7)
    x = torch.randn(100, feature_dim) * 5.0  # large inputs push raw outputs outside [0,1]
    scores = model(x)
    outside_unit = ((scores > 1.0) | (scores < 0.0)).any().item()
    assert outside_unit, "all scores inside [0,1] — sigmoid added?"

@pytest.mark.basic
def test_backward_pass_runs(model, feature_dim):
    """loss.backward() must not raise and must populate gradients."""
    x = torch.randn(5, feature_dim)
    scores = model(x)
    loss = scores.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"gradient missing for parameter '{name}'"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for '{name}'"

@pytest.mark.basic
def test_zero_grad_clears_gradients(model, feature_dim):
    """Calling zero_grad() must reset all gradients to zero (online update hygiene)."""
    x = torch.randn(5, feature_dim)
    model(x).sum().backward()
    model.zero_grad()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert (param.grad == 0).all(), f"gradient not zeroed for '{name}'"

@pytest.mark.basic
def test_stored_hyperparams(feature_dim, hidden_dim):
    """Constructor arguments must be accessible on the model instance."""
    m = ScoringMLP(feature_dim=feature_dim, hidden_dim=hidden_dim)
    assert m.feature_dim == feature_dim
    assert m.hidden_dim == hidden_dim


@pytest.mark.basic
def test_parameter_count(feature_dim, hidden_dim):
    """Total trainable parameters must match a one-hidden-layer MLP."""
    m = ScoringMLP(feature_dim=feature_dim, hidden_dim=hidden_dim)
    expected = (
        feature_dim * hidden_dim + hidden_dim   # Linear(feature_dim, hidden_dim)
        + hidden_dim * 1 + 1                    # Linear(hidden_dim, 1)
    )
    actual = sum(p.numel() for p in m.parameters())
    assert actual == expected

@pytest.mark.basic
def test_overfit_single_instance(feature_dim, hidden_dim):
    """Model must be able to memorize a fixed input/target after enough SGD steps."""
    torch.manual_seed(0)
    model = ScoringMLP(feature_dim=feature_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    num_items = 6
    x = torch.randn(num_items, feature_dim)
    target = torch.randn(num_items)

    initial_loss = criterion(model(x), target).item()

    for _ in range(500):
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), target).item()
    assert final_loss < initial_loss * 0.01, (
        f"model failed to overfit: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"
    )
