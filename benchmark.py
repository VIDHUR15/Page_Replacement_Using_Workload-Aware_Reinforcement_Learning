"""Repeat the ORIGINAL algorithms across seeds/capacities; no learning changes.

python3 benchmark.py --quick  # original training budget, one seed/capacity
python3 benchmark.py          # 10 seeds x 3 capacities x 5 workloads
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from evaluate import gap_to_opt_closed
from page_replacement.algorithms import run_fixed_policy, run_opt
from page_replacement.lecar import run_lecar
from page_replacement.rl_agent import QLearningPolicySelector
from page_replacement.adaptive_simulator import (
    run_adaptive_windowed, train_agent_windowed,
    run_adaptive_per_eviction, train_agent_per_eviction,
)
from page_replacement.workloads import WORKLOADS

CLASSICS = ("FIFO", "LRU", "LFU", "MRU")


@dataclass(frozen=True)
class Config:
    length: int = 3000
    page_range: int = 100
    episodes: int = 25
    window: int = 50
    lookback: int = 20
    min_commit: int = 20

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        # Original looping/mixed generators have a minimum loop size of two.
        if self.page_range < 2:
            raise ValueError("page_range must be at least 2 for the original workload generators")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "big")


def seed_plan(name, trial, episodes):
    """Trial 0 exactly preserves the original evaluator's seed choices."""
    if trial == 0:
        return dict(training=list(range(episodes)), evaluation=999_999, agent=42, lecar=0)
    return dict(training=[stable_seed("training", name, trial, e) for e in range(episodes)],
                evaluation=stable_seed("evaluation", name, trial),
                agent=stable_seed("agent", name, trial), lecar=stable_seed("lecar", name, trial))


def digest(trace):
    return hashlib.sha256(json.dumps(trace, separators=(",", ":")).encode()).hexdigest()


def run_trial(name, capacity, trial, config):
    config.validate()
    if name not in WORKLOADS:
        raise ValueError("unknown workload")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if isinstance(trial, bool) or not isinstance(trial, int) or trial < 0:
        raise ValueError("trial seeds must be nonnegative integers")
    seeds = seed_plan(name, trial, config.episodes)
    gen = WORKLOADS[name]
    training = [gen(config.length, config.page_range, seed=s) for s in seeds["training"]]
    ref = gen(config.length, config.page_range, seed=seeds["evaluation"])
    train_hashes, eval_hash = [digest(t) for t in training], digest(ref)
    overlap = sum(h == eval_hash for h in train_hashes)
    deterministic = name in ("sequential", "looping")
    rows = []

    def timed(call, training_seconds=0.0, q_states=0):
        t0 = time.perf_counter()
        row = call()
        elapsed = time.perf_counter()-t0
        source_policy = row["policy"]
        if source_policy == "LeCaR":
            row["policy"] = "LeCaR-inspired"
        elif source_policy == "RL-Adaptive (per-eviction)":
            row["policy"] = f"RL-FixedCommit-{config.min_commit}"
        elif source_policy == "RL-Adaptive (windowed)":
            row["policy"] = "RL-Windowed"
        row.update(source_policy=source_policy, workload=name, capacity=capacity, trial_seed=trial,
                   evaluation_seed=seeds["evaluation"], agent_seed=seeds["agent"],
                   lecar_seed=seeds["lecar"], trace_sha256=eval_hash,
                   training_trace_overlap_count=overlap, deterministic_workload=deterministic,
                   exec_time_s=elapsed, train_time_s=training_seconds, q_states_before_eval=q_states)
        rows.append(row)

    for p in CLASSICS:
        timed(lambda p=p: run_fixed_policy(ref, capacity, p))
    timed(lambda: run_opt(ref, capacity))
    timed(lambda: run_lecar(ref, capacity, seed=seeds["lecar"]))

    # Both original trainers request episode indices 0..episodes-1. Returning
    # pre-generated traces preserves their contents/order and all agent calls.
    # No agent RNG reset is performed between training and evaluation.
    def factory(seed=None):
        return training[seed]

    a = QLearningPolicySelector(seed=seeds["agent"])
    start = time.perf_counter()
    train_agent_windowed(factory, capacity, config.window, config.episodes, agent=a)
    elapsed = time.perf_counter()-start
    timed(lambda: run_adaptive_windowed(ref, capacity, config.window, a, train=False), elapsed, len(a.Q))

    a = QLearningPolicySelector(seed=seeds["agent"])
    start = time.perf_counter()
    train_agent_per_eviction(factory, capacity, config.lookback, config.episodes,
                             agent=a, min_commit=config.min_commit)
    elapsed = time.perf_counter()-start
    timed(lambda: run_adaptive_per_eviction(ref, capacity, config.lookback, a, train=False,
                                           min_commit=config.min_commit), elapsed, len(a.Q))
    lru = next(r for r in rows if r["policy"] == "LRU")
    opt = next(r for r in rows if r["policy"] == "OPT")
    best = min(r["faults"] for r in rows if r["policy"] in CLASSICS)
    for row in rows:
        row["best_fixed_faults"] = best
        row["gap_to_opt_pct"] = gap_to_opt_closed(lru["fault_rate"], opt["fault_rate"], row["fault_rate"])
        row["fault_reduction_vs_best_fixed_pct"] = 100*(best-row["faults"])/best if best else None
        row["vs_best_fixed"] = "win" if row["faults"] < best else "tie" if row["faults"] == best else "loss"
    manifest = dict(workload=name, capacity=capacity, trial_seed=trial, seeds=seeds,
                    training_trace_sha256=train_hashes, evaluation_trace_sha256=eval_hash,
                    evaluation_unique_pages=len(set(ref)), deterministic_workload=deterministic,
                    training_trace_overlap_count=overlap,
                    interpretation="Repeated deterministic trace; variation is agent randomness only."
                    if deterministic else "Randomized traces; inspect overlap count before claiming held-out evaluation.")
    return rows, manifest


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["workload"], row["capacity"], row["policy"]), []).append(row)
    output = []
    for (workload, capacity, policy), group in sorted(groups.items()):
        r = dict(workload=workload, capacity=capacity, policy=policy, trials=len(group),
                 deterministic_workload=group[0]["deterministic_workload"],
                 trials_with_training_overlap=sum(x["training_trace_overlap_count"] > 0 for x in group))
        for metric in ("hit_rate", "fault_rate", "faults", "gap_to_opt_pct",
                       "fault_reduction_vs_best_fixed_pct", "exec_time_s", "train_time_s"):
            values = [x[metric] for x in group if x.get(metric) is not None]
            r[metric+"_mean"] = mean(values) if values else None
            r[metric+"_std"] = stdev(values) if len(values)>1 else None
        for outcome, plural in (("win","wins"), ("tie","ties"), ("loss","losses")):
            r[plural+"_vs_best_fixed"] = sum(x["vs_best_fixed"] == outcome for x in group)
        output.append(r)
    return output


def paired_comparisons(rows):
    trials = {}
    for r in rows:
        trials.setdefault((r["workload"], r["capacity"], r["trial_seed"]), {})[r["policy"]] = r
    groups = {}
    for (workload, capacity, trial), policies in trials.items():
        comparators = {p:r["faults"] for p,r in policies.items()}
        comparators["Best-fixed (hindsight)"] = min(policies[p]["faults"] for p in CLASSICS)
        for policy, r in policies.items():
            if not policy.startswith("RL-"):
                continue
            for comparator, faults in comparators.items():
                if comparator != policy:
                    groups.setdefault((workload,capacity,policy,comparator), []).append(
                        100*(faults-r["faults"])/r["accesses"])
    return [dict(workload=w, capacity=c, policy=p, comparator=b, trials=len(v),
                 fault_rate_reduction_pp_mean=mean(v),
                 fault_rate_reduction_pp_std=stdev(v) if len(v)>1 else None,
                 wins=sum(x>0 for x in v), ties=sum(x==0 for x in v), losses=sum(x<0 for x in v))
            for (w,c,p,b),v in sorted(groups.items())]


RAW_FIELDS = ["workload", "capacity", "trial_seed", "policy", "source_policy", "accesses", "hits", "faults",
              "hit_rate", "fault_rate", "gap_to_opt_pct", "best_fixed_faults",
              "fault_reduction_vs_best_fixed_pct", "vs_best_fixed", "exec_time_s", "train_time_s",
              "policy_usage", "q_states_before_eval", "evaluation_seed", "agent_seed", "lecar_seed",
              "trace_sha256", "training_trace_overlap_count", "deterministic_workload"]


def serializable(row):
    return {k:json.dumps(v, sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(serializable(r) for r in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", nargs="+", choices=list(WORKLOADS), default=list(WORKLOADS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--capacities", nargs="+", type=int, default=[10,20,40])
    for name, default in (("length",3000), ("page-range",100), ("episodes",25),
                          ("window",50), ("lookback",20), ("min-commit",20)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--quick", action="store_true",
                        help="One trial (seed 0), 20 frames; retains the full original training budget")
    parser.add_argument("--output", type=Path, help="New output folder; existing folders are never overwritten")
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds, args.capacities = [0], [20]
    config = Config(args.length,args.page_range,args.episodes,args.window,args.lookback,args.min_commit)
    try:
        config.validate()
        if any(s<0 for s in args.seeds) or any(c<=0 for c in args.capacities):
            raise ValueError("seeds must be nonnegative and capacities positive")
        for name in ("workloads","seeds","capacities"):
            if len(getattr(args,name)) != len(set(getattr(args,name))):
                raise ValueError(f"{name} must not contain duplicates")
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output or Path("results")/datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    try:
        output.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        parser.error(f"output already exists: {output}; choose a new folder")
    root = Path(__file__).parent
    files = sorted(root.glob("*.py"))+sorted((root/"page_replacement").glob("*.py"))
    metadata = dict(config=asdict(config), seeds=args.seeds, capacities=args.capacities,
                    workloads=args.workloads, python=sys.version, platform=platform.platform(),
                    source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    started_utc=datetime.now(timezone.utc).isoformat(), status="running",
                    command_arguments=list(argv) if argv is not None else sys.argv[1:],
                    evaluation_note="Original algorithms and generators unchanged; deterministic traces repeat across seeds.")
    metadata_path = output/"config.json"
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    rows, completed, overlapping_trials = [], 0, 0
    total = len(args.workloads)*len(args.capacities)*len(args.seeds)
    try:
        with (output/"raw_results.csv").open("w",newline="",encoding="utf-8") as raw, \
             (output/"trace_manifest.jsonl").open("w",encoding="utf-8") as manifests:
            writer = csv.DictWriter(raw,fieldnames=RAW_FIELDS,extrasaction="ignore")
            writer.writeheader()
            for name in args.workloads:
                for capacity in args.capacities:
                    for seed in args.seeds:
                        batch, manifest = run_trial(name,capacity,seed,config)
                        writer.writerows(serializable(r) for r in batch)
                        rows.extend(batch)
                        manifests.write(json.dumps(manifest)+"\n")
                        raw.flush()
                        manifests.flush()
                        completed += 1
                        overlapping_trials += int(manifest["training_trace_overlap_count"]>0)
                        print(f"[{completed}/{total}] {name}, capacity={capacity}, seed={seed}",flush=True)
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
                        completed_trials=completed,error=str(exc))
        metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
        raise
    summary = summarize(rows)
    write_csv(output/"summary.csv",summary)
    write_csv(output/"paired_comparisons.csv",paired_comparisons(rows))
    metadata.update(status="complete",completed_trials=completed,result_rows=len(rows),
                    trials_with_training_overlap=overlapping_trials,
                    finished_utc=datetime.now(timezone.utc).isoformat())
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    print("\nOriginal RL results (mean hit rate +/- sample SD; W/T/L versus best classic policy):")
    for r in summary:
        if r["policy"].startswith("RL-"):
            sd = "n/a (one trial)" if r["hit_rate_std"] is None else f"{100*r['hit_rate_std']:.2f}%"
            print(f"{r['workload']:<11} C={r['capacity']:<3} {r['policy']:<19} "
                  f"{100*r['hit_rate_mean']:6.2f}% +/- {sd}; "
                  f"{r['wins_vs_best_fixed']}/{r['ties_vs_best_fixed']}/{r['losses_vs_best_fixed']}")
    if overlapping_trials:
        print(f"\nNOTICE: {overlapping_trials}/{completed} trials have training/evaluation trace overlap. "
              "Sequential/looping ignore seeds in the original generators; those runs measure agent randomness, not held-out trace generalization.")
    print(f"\nResults: {output.resolve()}")
    return output


if __name__ == "__main__":
    main()
