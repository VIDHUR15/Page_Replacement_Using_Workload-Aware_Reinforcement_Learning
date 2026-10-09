"""
LeCaR (Learning Cache Replacement) -- Vietri et al., "Driving Cache
Replacement with ML-based LeCaR", HotStorage '18.

The real published baseline that this project's per-eviction RL agent is
modeled after. Included here as an actual implementation (not just a
citation) so results can be compared head-to-head against a real prior
method, not just against the classic fixed policies and OPT.

Mechanism, in short
--------------------
Two "experts" -- LRU and LFU -- run side by side. Each has a weight
(w_lru, w_lfu, summing to 1) representing how much LeCaR currently trusts
it. On every eviction, LeCaR randomly picks which expert gets to choose
the victim, with probability equal to that expert's weight -- so a better
expert gets consulted more often over time.

To learn, LeCaR keeps a bounded "history" (one per expert) of pages it
recently evicted under that expert's advice. If a page comes back and is
requested again shortly after being evicted, that's direct evidence the
eviction was a mistake ("regret") -- LeCaR looks up which expert is
responsible and multiplicatively shrinks that expert's weight, discounted
by how long ago the eviction happened (a more recent bad call is punished
harder than a distant one). Weights are renormalized after every update.

This implementation follows the mechanism described in the paper
(weighted random expert selection + regret-based multiplicative weight
update via bounded per-expert history) using this project's existing
FrameTable/victim-rule infrastructure. Hyperparameter defaults
(learning_rate=0.45, discount_rate=0.005**(1/capacity)) mirror the values
reported in the paper, but this is an independent reimplementation, not
the authors' original code -- exact numbers may differ slightly from the
published results.
"""

import math
import random
from collections import deque

from .algorithms import FrameTable, victim_lru, victim_lfu

_MIN_WEIGHT = 1e-6  # keeps a losing expert's weight from collapsing to exactly
                    # zero, so it can still recover if the workload shifts back


class LeCaR:
    def __init__(self, capacity: int, learning_rate: float = 0.45,
                 discount_rate: float | None = None, seed: int | None = None):
        self.capacity = capacity
        self.lr = learning_rate
        self.discount_rate = discount_rate if discount_rate is not None else 0.005 ** (1.0 / capacity)
        self.weight = {"LRU": 0.5, "LFU": 0.5}

        # Bounded eviction history per expert: page -> time it was evicted.
        # Capped at `capacity` entries each, oldest dropped first (a ghost
        # cache), matching the paper's history-size choice.
        self._history_order = {"LRU": deque(), "LFU": deque()}
        self._history_time = {"LRU": {}, "LFU": {}}

        self.rng = random.Random(seed)
        self.clock = 0

    def _victim_for(self, table: FrameTable, expert: str) -> int:
        return victim_lru(table) if expert == "LRU" else victim_lfu(table)

    def _choose_expert(self) -> str:
        return "LRU" if self.rng.random() < self.weight["LRU"] else "LFU"

    def _remember_eviction(self, expert: str, page: int) -> None:
        order, times = self._history_order[expert], self._history_time[expert]
        order.append(page)
        times[page] = self.clock
        if len(order) > self.capacity:
            oldest = order.popleft()
            times.pop(oldest, None)

    def _penalize(self, expert: str, evicted_at: int) -> None:
        age = self.clock - evicted_at
        regret = -(self.discount_rate ** age)  # more recent miss -> stronger penalty
        self.weight[expert] *= math.exp(self.lr * regret)
        self.weight[expert] = max(self.weight[expert], _MIN_WEIGHT)
        total = self.weight["LRU"] + self.weight["LFU"]
        self.weight["LRU"] /= total
        self.weight["LFU"] /= total

    def step(self, table: FrameTable, page: int) -> bool:
        """Process one access. Returns True on hit."""
        self.clock += 1

        if page in table:
            table.touch(page)
            return True

        # Regret check: was this page evicted before, and by whom?
        for expert in ("LRU", "LFU"):
            times = self._history_time[expert]
            if page in times:
                evicted_at = times.pop(page)
                try:
                    self._history_order[expert].remove(page)
                except ValueError:
                    pass
                self._penalize(expert, evicted_at)

        if not table.is_full():
            table.insert(page)
        else:
            expert = self._choose_expert()
            victim = self._victim_for(table, expert)
            table.evict_and_insert(victim, page)
            self._remember_eviction(expert, victim)

        return False


def run_lecar(ref_string: list[int], capacity: int, learning_rate: float = 0.45,
              discount_rate: float | None = None, seed: int = 0) -> dict:
    table = FrameTable(capacity)
    lecar = LeCaR(capacity, learning_rate, discount_rate, seed=seed)

    hits = 0
    for page in ref_string:
        if lecar.step(table, page):
            hits += 1

    faults = len(ref_string) - hits
    return {
        "policy": "LeCaR",
        "accesses": len(ref_string),
        "hits": hits,
        "faults": faults,
        "hit_rate": hits / len(ref_string),
        "fault_rate": faults / len(ref_string),
    }
