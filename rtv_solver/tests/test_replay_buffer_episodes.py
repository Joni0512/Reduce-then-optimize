from rtv_solver.pipeline.replay_buffer import ReplayBuffer


def test_drop_episodes_before():
    b = ReplayBuffer(capacity=100)
    for ep in (1, 2, 3):
        b.current_episode = ep
        for _ in range(4):
            b.add_transition(step=None, reward=float(ep), next_step=None)
    assert len(b) == 12
    assert b.drop_episodes_before(2) == 4
    assert sorted({e.episode for e in b.entries}) == [2, 3]
    b.current_episode = None  # untagged entries (old modes) are never dropped
    b.add_transition(step=None, reward=0.0, next_step=None)
    assert b.drop_episodes_before(10) == 8 and len(b) == 1
