"""
Synthetic memory-access reference string generators.

Each function returns a list[int] of page numbers. These stand in for a
trace you'd otherwise collect from /proc/<pid>/pagemap, a memory simulator
(e.g. gem5), or a benchmark suite (SPEC, etc.) — swap in real traces later
without touching the rest of the pipeline.
"""

import random


def sequential(length: int, page_range: int, start: int = 0) -> list[int]:
    """Pure sequential scan, wraps around page_range (e.g. streaming a file)."""
    return [(start + i) % page_range for i in range(length)]


def random_uniform(length: int, page_range: int, seed: int | None = None) -> list[int]:
    """Uniformly random accesses (worst case for locality-based policies)."""
    rng = random.Random(seed)
    return [rng.randrange(page_range) for _ in range(length)]


def looping(length: int, loop_size: int, seed: int | None = None) -> list[int]:
    """Repeatedly scans a fixed-size loop of pages (classic LRU/FIFO weak spot
    when loop_size > number of frames — this is the case MRU was designed for)."""
    loop = list(range(loop_size))
    out = []
    i = 0
    while len(out) < length:
        out.append(loop[i % loop_size])
        i += 1
    return out


def locality_based(length: int, page_range: int, hot_fraction: float = 0.2,
                    hot_prob: float = 0.8, seed: int | None = None) -> list[int]:
    """80/20-style working set: a small 'hot' set of pages is accessed most
    of the time, the rest of the address space rarely (typical real-world
    program behaviour, favours LRU/LFU)."""
    rng = random.Random(seed)
    hot_size = max(1, int(page_range * hot_fraction))
    hot_pages = list(range(hot_size))
    cold_pages = list(range(hot_size, page_range))
    out = []
    for _ in range(length):
        if rng.random() < hot_prob or not cold_pages:
            out.append(rng.choice(hot_pages))
        else:
            out.append(rng.choice(cold_pages))
    return out


def mixed(length: int, page_range: int, segment_len: int = 200,
          seed: int | None = None) -> list[int]:
    """Concatenates segments of the other patterns back-to-back, so the
    'correct' policy changes over time — this is the scenario the adaptive
    RL approach specifically targets."""
    rng = random.Random(seed)
    generators = [
        lambda n: sequential(n, page_range, start=rng.randrange(page_range)),
        lambda n: random_uniform(n, page_range, seed=rng.randrange(1_000_000)),
        lambda n: looping(n, loop_size=max(2, page_range // 4)),
        lambda n: locality_based(n, page_range, seed=rng.randrange(1_000_000)),
    ]
    out = []
    while len(out) < length:
        gen = rng.choice(generators)
        out.extend(gen(segment_len))
    return out[:length]


WORKLOADS = {
    "sequential": lambda n, pr, seed=None: sequential(n, pr),
    "random": random_uniform,
    "looping": lambda n, pr, seed=None: looping(n, loop_size=max(2, pr // 3)),
    "locality": locality_based,
    "mixed": mixed,
}
