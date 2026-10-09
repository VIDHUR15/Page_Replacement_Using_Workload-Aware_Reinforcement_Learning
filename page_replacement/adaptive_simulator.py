"""
Two ways to run the RL-adaptive policy selector, differing in how often the
agent is allowed to change its mind:

run_adaptive_windowed()
    Decides once per fixed-size window of accesses (the original design).
    Cheap and easy to reason about, but coarser than what published work
    (LeCaR, CACHEUS) actually does.

run_adaptive_per_eviction()
    Decides at every single eviction -- the same granularity LeCaR/CACHEUS
    use, where an "expert" (LRU or LFU there; FIFO/LRU/LFU/MRU here) is
    picked fresh on every cache miss that requires evicting something.
    State is built from a short, causal sliding window of the most recent
    accesses (not the whole trace) so the agent only ever sees the past.

    Reward design: rather than needing a fixed-size lookahead, the reward
    for an eviction decision is the length of the "hit streak" until the
    *next* fault, normalized by cache capacity and capped at 1.0. A good
    eviction (the evicted page wasn't needed again soon) produces a long
    run of hits before the next fault; a bad eviction produces an
    immediate or near-immediate fault. This mirrors the intuition behind
    LeCaR's regret signal (penalize evictions that turn out to be
    "regretted") without needing to maintain a separate ghost-history
    structure.
"""

import time

from .algorithms import FrameTable, POLICIES, step
from .features import extract_state
from .rl_agent import QLearningPolicySelector

DEFAULT_STATE = (0, 0, 0, 0)


# --------------------------------------------------------------------------
# Windowed variant (original design)
# --------------------------------------------------------------------------

def run_adaptive_windowed(ref_string: list[int], capacity: int, window_size: int,
                           agent: QLearningPolicySelector, train: bool = True) -> dict:
    table = FrameTable(capacity)
    n = len(ref_string)
    windows = [ref_string[i:i + window_size] for i in range(0, n, window_size)]

    total_hits = 0
    total_accesses = 0
    policy_usage = {a: 0 for a in agent.actions}
    prev_state = DEFAULT_STATE
    decision_time = 0.0  # time spent on feature-extraction + agent decision

    for idx, window in enumerate(windows):
        t0 = time.perf_counter()
        state = prev_state if idx > 0 else DEFAULT_STATE
        action = agent.select_action(state, greedy=not train)
        decision_time += time.perf_counter() - t0

        policy_usage[action] += 1
        hits = 0
        for page in window:
            if step(table, page, action):
                hits += 1
        hit_rate = hits / len(window)

        total_hits += hits
        total_accesses += len(window)

        next_state = extract_state(window)
        if train:
            agent.update(state, action, hit_rate, next_state)
            agent.decay_epsilon()
        prev_state = next_state

    faults = total_accesses - total_hits
    return {
        "policy": "RL-Adaptive (windowed)",
        "accesses": total_accesses,
        "hits": total_hits,
        "faults": faults,
        "hit_rate": total_hits / total_accesses,
        "fault_rate": faults / total_accesses,
        "policy_usage": policy_usage,
        "decision_time_s": decision_time,
    }


def train_agent_windowed(ref_string_factory, capacity: int, window_size: int,
                          episodes: int, agent: QLearningPolicySelector | None = None,
                          **factory_kwargs) -> QLearningPolicySelector:
    """Train an agent across several freshly-sampled reference strings from
    the same workload family, so it generalizes rather than memorizing one
    trace. Returns the trained agent (Q-table persists across episodes)."""
    if agent is None:
        agent = QLearningPolicySelector()
    for ep in range(episodes):
        ref = ref_string_factory(seed=ep, **factory_kwargs)
        run_adaptive_windowed(ref, capacity, window_size, agent, train=True)
    return agent


# --------------------------------------------------------------------------
# Per-eviction variant (LeCaR/CACHEUS-style granularity)
# --------------------------------------------------------------------------

def run_adaptive_per_eviction(ref_string: list[int], capacity: int, lookback: int,
                               agent: QLearningPolicySelector, train: bool = True,
                               min_commit: int = 1) -> dict:
    """
    min_commit: once the agent picks a policy at an eviction, it stays
    committed to that same policy for at least `min_commit` consecutive
    evictions before it's allowed to reconsider. The whole committed block
    is then scored as ONE reward (total hits accumulated across the block,
    normalized), rather than just the single hit-streak after one eviction.

    Why this exists: some policies (MRU in particular) only pay off after
    many *consecutive* evictions under the same rule -- a single-eviction
    reward horizon can never see that payoff, so the agent could never
    learn to prefer MRU even when it's clearly the best choice (this is
    exactly what happened on the `sequential` workload before this fix:
    MRU needs ~(page_range - capacity) consistent evictions before a
    wraparound access can turn into a hit). Widening the reward horizon to
    span a whole committed block lets that long-horizon payoff actually
    register. min_commit=1 recovers the original eviction-by-eviction
    behaviour exactly.
    """
    table = FrameTable(capacity)
    n = len(ref_string)

    total_hits = 0
    policy_usage = {a: 0 for a in agent.actions}
    decision_time = 0.0

    recent: list[int] = []          # bounded causal history, for state features
    max_recent = max(lookback * 4, lookback + 1)

    block_action = None             # policy currently committed to
    block_state = None              # state at the start of this block
    evictions_in_block = 0
    hits_in_block = 0               # hits observed since the block's FIRST eviction

    def current_state() -> tuple:
        if not recent:
            return DEFAULT_STATE
        return extract_state(recent[-lookback:])

    def close_block() -> None:
        """Score the just-finished commitment block as one reward, spanning
        every eviction and hit that happened while this policy was held."""
        nonlocal block_action, block_state, evictions_in_block, hits_in_block
        if block_action is not None and train:
            denom = capacity * max(evictions_in_block, 1)
            reward = min(hits_in_block / denom, 1.0)
            agent.update(block_state, block_action, reward, current_state())
            agent.decay_epsilon()
        block_action, block_state = None, None
        evictions_in_block, hits_in_block = 0, 0

    for i, page in enumerate(ref_string):
        if page in table:
            table.touch(page)
            total_hits += 1
            if block_action is not None:
                hits_in_block += 1
        else:
            if not table.is_full():
                table.insert(page)
            else:
                need_new_decision = block_action is None or evictions_in_block >= min_commit
                if need_new_decision:
                    close_block()
                    t0 = time.perf_counter()
                    state = current_state()
                    action = agent.select_action(state, greedy=not train)
                    decision_time += time.perf_counter() - t0
                    block_action, block_state = action, state

                policy_usage[block_action] += 1
                victim = POLICIES[block_action](table)
                table.evict_and_insert(victim, page)
                evictions_in_block += 1

        recent.append(page)
        if len(recent) > max_recent:
            del recent[:-lookback]

    close_block()  # score whatever block was still open at the end of the trace

    faults = n - total_hits
    return {
        "policy": "RL-Adaptive (per-eviction)",
        "accesses": n,
        "hits": total_hits,
        "faults": faults,
        "hit_rate": total_hits / n,
        "fault_rate": faults / n,
        "policy_usage": policy_usage,
        "decision_time_s": decision_time,
    }


def train_agent_per_eviction(ref_string_factory, capacity: int, lookback: int,
                              episodes: int, agent: QLearningPolicySelector | None = None,
                              min_commit: int = 1, **factory_kwargs) -> QLearningPolicySelector:
    if agent is None:
        agent = QLearningPolicySelector()
    for ep in range(episodes):
        ref = ref_string_factory(seed=ep, **factory_kwargs)
        run_adaptive_per_eviction(ref, capacity, lookback, agent, train=True, min_commit=min_commit)
    return agent


# Backwards-compatible aliases (original names, windowed behaviour)
run_adaptive = run_adaptive_windowed
train_agent = train_agent_windowed
