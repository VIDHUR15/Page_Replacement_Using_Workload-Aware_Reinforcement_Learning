"""
Baseline page-replacement policies (FIFO, LRU, LFU, MRU) plus a shared
frame-metadata manager that lets a single simulator switch strategies
window-by-window without losing frame history.

Design note
-----------
Each classic algorithm only differs in *which piece of metadata* it uses to
pick an eviction victim:

    FIFO -> oldest insertion time
    LRU  -> oldest last-used time
    LFU  -> lowest access frequency (ties -> oldest insertion time)
    MRU  -> newest last-used time

So instead of four unrelated implementations, `FrameTable` tracks all three
pieces of metadata (insertion_time, last_used_time, freq) for every resident
page, and each policy is just a one-line "pick the victim" rule operating on
that shared table. This is what makes it possible for the adaptive/RL
simulator to swap policies every window while keeping the frame contents and
history consistent (see adaptive_simulator.py).
"""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class FrameEntry:
    page: int
    insertion_time: int
    last_used_time: int
    freq: int = 1


class FrameTable:
    """Holds the resident pages and the metadata every policy needs."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.frames: dict[int, FrameEntry] = {}
        self.clock = 0

    def __contains__(self, page: int) -> bool:
        return page in self.frames

    def is_full(self) -> bool:
        return len(self.frames) >= self.capacity

    def touch(self, page: int) -> None:
        """Record a hit: bump recency + frequency."""
        e = self.frames[page]
        self.clock += 1
        e.last_used_time = self.clock
        e.freq += 1

    def insert(self, page: int) -> None:
        """Record a fault-load into a free frame (capacity must allow it)."""
        self.clock += 1
        self.frames[page] = FrameEntry(page, self.clock, self.clock, 1)

    def evict_and_insert(self, victim: int, page: int) -> None:
        del self.frames[victim]
        self.insert(page)


# ---- victim-selection rules -------------------------------------------------

def victim_fifo(table: FrameTable) -> int:
    return min(table.frames.values(), key=lambda e: e.insertion_time).page


def victim_lru(table: FrameTable) -> int:
    return min(table.frames.values(), key=lambda e: e.last_used_time).page


def victim_lfu(table: FrameTable) -> int:
    return min(table.frames.values(), key=lambda e: (e.freq, e.insertion_time)).page


def victim_mru(table: FrameTable) -> int:
    return max(table.frames.values(), key=lambda e: e.last_used_time).page


POLICIES = {
    "FIFO": victim_fifo,
    "LRU": victim_lru,
    "LFU": victim_lfu,
    "MRU": victim_mru,
}


def step(table: FrameTable, page: int, policy_name: str) -> bool:
    """Apply one memory access under `policy_name`. Returns True on hit."""
    if page in table:
        table.touch(page)
        return True

    if not table.is_full():
        table.insert(page)
    else:
        victim = POLICIES[policy_name](table)
        table.evict_and_insert(victim, page)
    return False


def run_fixed_policy(ref_string: list[int], capacity: int, policy_name: str) -> dict:
    """Run one policy, unchanged, over the whole reference string."""
    table = FrameTable(capacity)
    hits = 0
    for page in ref_string:
        if step(table, page, policy_name):
            hits += 1
    faults = len(ref_string) - hits
    return {
        "policy": policy_name,
        "accesses": len(ref_string),
        "hits": hits,
        "faults": faults,
        "hit_rate": hits / len(ref_string),
        "fault_rate": faults / len(ref_string),
    }


def run_opt(ref_string: list[int], capacity: int) -> dict:
    """Belady's optimal (OPT/MIN) algorithm: on every fault with a full
    cache, evict the resident page whose *next* use is furthest in the
    future (or never used again). This requires knowing the whole
    reference string in advance, so it's not implementable online in a
    real OS -- it exists purely as the theoretical best-possible ceiling
    that every other policy is measured against, the same role it plays
    in the research this project is modeled on (e.g. Phoebe, CACHEUS).

    Runtime: O(n * capacity) -- a linear scan over resident pages per
    fault, each with an O(log k) lookahead via binary search. Fine for
    the trace lengths and capacities this project uses.
    """
    n = len(ref_string)

    # For each page value, the sorted list of indices where it occurs.
    occurrences: dict[int, list[int]] = defaultdict(list)
    for i, page in enumerate(ref_string):
        occurrences[page].append(i)

    resident: set[int] = set()
    hits = 0

    for i, page in enumerate(ref_string):
        if page in resident:
            hits += 1
            continue

        if len(resident) < capacity:
            resident.add(page)
            continue

        # Find the resident page used furthest in the future (or never
        # again -> treated as "infinity", i.e. index n).
        victim = None
        farthest_next_use = -1
        for p in resident:
            idx_list = occurrences[p]
            pos = bisect_right(idx_list, i)
            next_use = idx_list[pos] if pos < len(idx_list) else n
            if next_use > farthest_next_use:
                farthest_next_use = next_use
                victim = p
        resident.discard(victim)
        resident.add(page)

    faults = n - hits
    return {
        "policy": "OPT",
        "accesses": n,
        "hits": hits,
        "faults": faults,
        "hit_rate": hits / n,
        "fault_rate": faults / n,
    }
