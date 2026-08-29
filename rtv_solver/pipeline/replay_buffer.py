from __future__ import annotations

import random
from dataclasses import dataclass

from rtv_solver.pipeline.episode_buffer import EpisodeStep


@dataclass
class ReplayEntry:
    step: EpisodeStep
    target: float


class ReplayBuffer:
    """
    2026-08-28: cross-episode Monte Carlo replay buffer for critic training
    (see chat) - decorrelates critic updates from the single just-finished
    episode by mixing in steps from earlier episodes of the SAME instance
    run. Entries carry a fixed target (r_t/G_t), computed once at the end of
    the episode that produced them - unlike TD methods, nothing here ever
    changes after insertion, so no bootstrap target is needed.

    Ring buffer: capacity fixed, oldest entries evicted first once full -
    bounds how "stale" (relative to the current actor) any entry can be,
    without needing an explicit staleness/age check.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: list[ReplayEntry] = []
        self._next_index = 0

    def add(self, step: EpisodeStep, target: float) -> None:
        entry = ReplayEntry(step, target)
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
        else:
            self.entries[self._next_index] = entry
            self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size: int) -> list[ReplayEntry]:
        n = min(batch_size, len(self.entries))
        return random.sample(self.entries, n)

    def __len__(self) -> int:
        return len(self.entries)
