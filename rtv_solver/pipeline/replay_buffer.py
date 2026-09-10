from __future__ import annotations

import random
from dataclasses import dataclass

from rtv_solver.pipeline.episode_buffer import EpisodeStep


@dataclass
class ReplayEntry:
    step: EpisodeStep
    # 2026-09-10: exactly one of the two storage modes below is populated,
    # never both - see ReplayBuffer's docstring for which mode which
    # critic_target_mode uses.
    target: float | None = None          # monte_carlo mode: fixed target (G_t/r_t), used as-is
    reward: float | None = None          # td_bootstrap mode: raw r_t
    next_step: EpisodeStep | None = None  # td_bootstrap mode: raw s_{t+1} (None = terminal step)


class ReplayBuffer:
    """
    2026-08-28: cross-episode replay buffer for critic training (see chat) -
    decorrelates critic updates from the single just-finished episode by
    mixing in steps from earlier episodes of the SAME instance run.

    2026-09-10: two storage modes, picked by critic_target_mode at the call
    site (coaml_pipeline.py) - see chat, fixes a staleness bug:
    - monte_carlo: entries carry a FIXED target (r_t/G_t), via
      add_fixed_target(). Correct to compute once and never touch again -
      G_t only depends on the episode's actual outcome, not on any network
      state, so there is nothing to go stale.
    - td_bootstrap: entries carry the RAW transition (reward, next_step),
      via add_transition() - NOT a precomputed target. The bootstrap target
      (r_t + gamma*target_critic(next_step)) is only evaluated later, once
      per SAMPLE, in coaml_pipeline.py's batch loop (td_target_for_step()),
      using target_critic's weights at that later sampling time. Before
      this fix, td_bootstrap entries used to go through add_fixed_target()
      too (target computed once, at episode end, then frozen for however
      many later episodes the entry survives in the buffer) - matches
      DQN/DDPG's own semantics now (targets computed at minibatch-sample
      time, not at insertion time - see the papers' Algorithm 1 boxes).

    Ring buffer: capacity fixed, oldest entries evicted first once full -
    bounds how "stale" (relative to the current actor) any entry can be,
    without needing an explicit staleness/age check.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: list[ReplayEntry] = []
        self._next_index = 0

    def _add(self, entry: ReplayEntry) -> None:
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
        else:
            self.entries[self._next_index] = entry
            self._next_index = (self._next_index + 1) % self.capacity

    def add_fixed_target(self, step: EpisodeStep, target: float) -> None:
        """monte_carlo mode - see class docstring."""
        self._add(ReplayEntry(step, target=target))

    def add_transition(self, step: EpisodeStep, reward: float, next_step: EpisodeStep | None) -> None:
        """td_bootstrap mode - see class docstring."""
        self._add(ReplayEntry(step, reward=reward, next_step=next_step))

    def sample(self, batch_size: int) -> list[ReplayEntry]:
        n = min(batch_size, len(self.entries))
        return random.sample(self.entries, n)

    def __len__(self) -> int:
        return len(self.entries)
