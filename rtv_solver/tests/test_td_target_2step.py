import torch

from rtv_solver.pipeline.episode_buffer import EpisodeStep
from rtv_solver.pipeline.td_target_builder import td_target_for_step


class _ConstCritic(torch.nn.Module):
    """Returns a fixed value per call (ignores its inputs)."""
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, request_features, vehicle_features, edge_index):
        return torch.tensor(self.value)


def _step(t: float) -> EpisodeStep:
    return EpisodeStep(request_features=torch.zeros(1, 1), vehicle_features=torch.zeros(1, 1), edge_index=torch.zeros(2, 0, dtype=torch.long), current_time=t)


G = 0.9
CRIT = _ConstCritic(10.0)
CRIT2 = _ConstCritic(8.0)  # min of the twins = 8


def test_two_step_middle_of_episode():
    y = td_target_for_step(1.0, _step(1), CRIT, G, target_critic2=CRIT2, reward2=2.0, next_next_step=_step(2), n_steps=2)
    assert abs(y - (1.0 + G * 2.0 + G * G * 8.0)) < 1e-9


def test_two_step_second_to_last_has_no_bootstrap():
    y = td_target_for_step(1.0, _step(1), CRIT, G, target_critic2=CRIT2, reward2=2.0, next_next_step=None, n_steps=2)
    assert abs(y - (1.0 + G * 2.0)) < 1e-9


def test_two_step_last_step_is_reward_only():
    assert td_target_for_step(1.0, None, CRIT, G, reward2=None, next_next_step=None, n_steps=2) == 1.0


def test_two_step_gamma_zero_is_reward():
    assert td_target_for_step(1.0, _step(1), CRIT, 0.0, reward2=2.0, next_next_step=_step(2), n_steps=2) == 1.0


def test_one_step_unchanged():
    y = td_target_for_step(1.0, _step(1), CRIT, G, target_critic2=CRIT2, reward2=2.0, next_next_step=_step(2))
    assert abs(y - (1.0 + G * 8.0)) < 1e-9
