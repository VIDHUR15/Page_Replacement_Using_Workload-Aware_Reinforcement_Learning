"""
Compares the RL-adaptive page replacement approaches (windowed and
per-eviction) against fixed FIFO / LRU / LFU / MRU baselines AND against
OPT (Belady's optimal, offline, theoretical ceiling) across several
synthetic workloads.

Also reports "gap to OPT closed relative to LRU" -- the same relative-gap
metric used in published caching-RL papers (e.g. Phoebe): how much of the
distance between LRU and the theoretical best did each policy close.

Run:
    python3 evaluate.py
"""

import time

from page_replacement.algorithms import run_fixed_policy, run_opt
from page_replacement.lecar import run_lecar
from page_replacement.adaptive_simulator import (
    run_adaptive_windowed, train_agent_windowed,
    run_adaptive_per_eviction, train_agent_per_eviction,
)
from page_replacement.rl_agent import QLearningPolicySelector
from page_replacement.workloads import WORKLOADS

CAPACITY = 20          # number of physical frames
TRACE_LENGTH = 3000     # accesses per evaluation trace
PAGE_RANGE = 100        # size of the virtual address space (in pages)
WINDOW_SIZE = 50        # accesses per RL decision window (windowed variant)
LOOKBACK = 20           # accesses looked back at for state (per-eviction variant)
TRAIN_EPISODES = 25     # traces used to train each agent, per workload
MIN_COMMIT = 20         # evictions per commitment block (per-eviction variant) --
                        # see README for why this fixes the sequential/MRU gap


def make_factory(name):
    gen = WORKLOADS[name]

    def factory(seed=None):
        return gen(TRACE_LENGTH, PAGE_RANGE, seed=seed)
    return factory


def gap_to_opt_closed(fault_rate_lru: float, fault_rate_opt: float, fault_rate_policy: float):
    """% of the fault-rate gap between LRU and OPT that `policy` closes.
    100% = matches OPT exactly, 0% = no better than LRU, negative = worse
    than LRU. Returns None if LRU already matches OPT (no gap to measure)."""
    denom = fault_rate_lru - fault_rate_opt
    if abs(denom) < 1e-12:
        return None
    return (fault_rate_lru - fault_rate_policy) / denom * 100.0


def evaluate_workload(name: str) -> list[dict]:
    factory = make_factory(name)
    eval_ref = factory(seed=999_999)  # held-out trace, unseen during training

    rows = []

    # --- fixed baselines --------------------------------------------------
    for policy in ["FIFO", "LRU", "LFU", "MRU"]:
        t0 = time.perf_counter()
        result = run_fixed_policy(eval_ref, CAPACITY, policy)
        result["exec_time_s"] = time.perf_counter() - t0
        result["workload"] = name
        rows.append(result)

    # --- OPT (theoretical ceiling, offline) --------------------------------
    t0 = time.perf_counter()
    opt_result = run_opt(eval_ref, CAPACITY)
    opt_result["exec_time_s"] = time.perf_counter() - t0
    opt_result["workload"] = name
    rows.append(opt_result)

    # --- LeCaR (published baseline: regret-based LRU/LFU mixing) -----------
    t0 = time.perf_counter()
    lecar_result = run_lecar(eval_ref, CAPACITY, seed=0)
    lecar_result["exec_time_s"] = time.perf_counter() - t0
    lecar_result["train_time_s"] = 0.0
    lecar_result["workload"] = name
    rows.append(lecar_result)

    # --- RL-adaptive: windowed ---------------------------------------------
    agent_w = QLearningPolicySelector(seed=42)
    t_train0 = time.perf_counter()
    train_agent_windowed(factory, CAPACITY, WINDOW_SIZE, TRAIN_EPISODES, agent=agent_w)
    train_time_w = time.perf_counter() - t_train0

    t0 = time.perf_counter()
    result_w = run_adaptive_windowed(eval_ref, CAPACITY, WINDOW_SIZE, agent_w, train=False)
    result_w["exec_time_s"] = time.perf_counter() - t0
    result_w["train_time_s"] = train_time_w
    result_w["workload"] = name
    rows.append(result_w)

    # --- RL-adaptive: per-eviction (LeCaR/CACHEUS-style granularity) -------
    agent_pe = QLearningPolicySelector(seed=42)
    t_train0 = time.perf_counter()
    train_agent_per_eviction(factory, CAPACITY, LOOKBACK, TRAIN_EPISODES, agent=agent_pe,
                              min_commit=MIN_COMMIT)
    train_time_pe = time.perf_counter() - t_train0

    t0 = time.perf_counter()
    result_pe = run_adaptive_per_eviction(eval_ref, CAPACITY, LOOKBACK, agent_pe, train=False,
                                           min_commit=MIN_COMMIT)
    result_pe["exec_time_s"] = time.perf_counter() - t0
    result_pe["train_time_s"] = train_time_pe
    result_pe["workload"] = name
    rows.append(result_pe)

    # --- gap-to-OPT closed, relative to LRU ---------------------------------
    lru_fr = next(r["fault_rate"] for r in rows if r["policy"] == "LRU")
    opt_fr = opt_result["fault_rate"]
    for r in rows:
        r["gap_to_opt_pct"] = gap_to_opt_closed(lru_fr, opt_fr, r["fault_rate"])

    return rows


def print_table(rows: list[dict]) -> None:
    header = (f"{'workload':<10} {'policy':<26} {'hit_rate':>9} {'fault_rate':>11} "
              f"{'faults':>7} {'gap_to_OPT':>11} {'exec_ms':>9} {'train_ms':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        gap = r.get("gap_to_opt_pct")
        gap_str = f"{gap:>10.1f}%" if gap is not None else f"{'--':>11}"
        print(f"{r['workload']:<10} {r['policy']:<26} {r['hit_rate']:>9.3f} "
              f"{r['fault_rate']:>11.3f} {r['faults']:>7} {gap_str} "
              f"{r['exec_time_s']*1000:>9.2f} {r.get('train_time_s', 0)*1000:>9.1f}")


def main():
    all_rows = []
    for name in WORKLOADS:
        rows = evaluate_workload(name)
        all_rows.extend(rows)
        windowed = next(r for r in rows if r["policy"] == "RL-Adaptive (windowed)")
        per_evict = next(r for r in rows if r["policy"] == "RL-Adaptive (per-eviction)")
        print(f"\n=== workload: {name} ===")
        print(f"  windowed policy usage:    {windowed['policy_usage']}")
        print(f"  per-eviction policy usage: {per_evict['policy_usage']}")
    print("\n\n=== Summary ===")
    print_table(all_rows)


if __name__ == "__main__":
    main()
