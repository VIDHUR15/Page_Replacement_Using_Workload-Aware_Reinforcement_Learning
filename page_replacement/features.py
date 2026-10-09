"""
Turns a raw window of page accesses into a small, discretized feature
vector the RL agent can use as its state. Kept deliberately cheap to
compute (all O(window_size)) since this runs on every window — an
expensive feature extractor would defeat the point of an "adaptation
overhead" comparison against static policies.
"""

from collections import Counter


def raw_features(window: list[int]) -> dict:
    n = len(window)
    if n == 0:
        return {"unique_ratio": 0.0, "top_freq_ratio": 0.0,
                "sequential_ratio": 0.0, "repeat_ratio": 0.0}

    counts = Counter(window)
    unique_ratio = len(counts) / n
    top_freq_ratio = counts.most_common(1)[0][1] / n

    # fraction of consecutive steps that are page+1 (streaming behaviour)
    seq_steps = sum(1 for a, b in zip(window, window[1:]) if b == a + 1)
    sequential_ratio = seq_steps / max(1, n - 1)

    # fraction of accesses that are an immediate repeat of a recently
    # seen page within a short lookback (proxy for looping/locality)
    lookback = 8
    repeats = 0
    for i in range(1, n):
        start = max(0, i - lookback)
        if window[i] in window[start:i]:
            repeats += 1
    repeat_ratio = repeats / max(1, n - 1)

    return {
        "unique_ratio": unique_ratio,
        "top_freq_ratio": top_freq_ratio,
        "sequential_ratio": sequential_ratio,
        "repeat_ratio": repeat_ratio,
    }


def _bucket(value: float, edges=(0.2, 0.4, 0.6, 0.8)) -> int:
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def discretize(feat: dict) -> tuple:
    """Bucket each feature into ~5 bins -> compact hashable state for a
    tabular Q-table. (Swap for a neural function approximator later if the
    state space needs to grow.)"""
    return (
        _bucket(feat["unique_ratio"]),
        _bucket(feat["top_freq_ratio"]),
        _bucket(feat["sequential_ratio"]),
        _bucket(feat["repeat_ratio"]),
    )


def extract_state(window: list[int]) -> tuple:
    return discretize(raw_features(window))
