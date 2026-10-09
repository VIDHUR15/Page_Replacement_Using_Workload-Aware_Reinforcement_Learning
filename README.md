# Workload-Aware Adaptive Page Replacement Using Reinforcement Learning

This edition preserves the **original implementation** and adds only two
improvements: corrected documentation and an automated evaluation wrapper.
There is **no adaptive-commitment-length agent** in this project.

All nine original Python files, including `evaluate.py`, `gui.py`, and every
module in `page_replacement/`, are byte-for-byte unchanged from the uploaded
ZIP. The original reward functions, Q-update, gamma, exploration schedule,
features, workload generators, and fixed `min_commit` behavior are retained.
`ORIGINAL_SOURCE_SHA256.json` records their original hashes, checked by tests.

## Run the original project

Use Python **3.10 or newer**. The command-line programs and tests need only the
Python standard library.

```bash
python3 evaluate.py
```

This remains the original single-seed evaluation: 20 frames, 3,000 references,
100-page nominal range, 25 training episodes, window 50, lookback 20, and fixed
commitment 20. It prints the original table and policy-usage counts.

For the original desktop GUI:

```bash
python3 -m pip install matplotlib
python3 gui.py
```

Tkinter must also be available in your Python installation. Check with
`python3 -m tkinter`. Installing matplotlib does not install Tkinter.
The GUI is unchanged and runs single-trial comparisons. Use the new benchmark
below for repeated experiments and CSV exports.

## Run automated evaluation

The new `benchmark.py` calls the original simulation and training functions.
It does not replace them or change their learning behavior.

```bash
python3 benchmark.py --quick
```

**Unlike the superseded adaptive-duration ZIP, `--quick` does not reduce the
training budget.** It selects trial seed 0 and capacity 20 while keeping the
default 25 episodes and 3,000 references. With no other parameter overrides,
its hit/fault counts and policy usage reproduce the original `evaluate.py`.
Timing values can differ.

For the full repeated benchmark:

```bash
python3 benchmark.py
```

Defaults: 10 trial seeds (0–9), capacities 10/20/40, five workloads, and all
eight original methods. That is 150 trials and 1,200 method-result rows.
Each RL variant receives 25 training traces of 3,000 references per trial.
A model is trained separately for each workload/capacity/trial. Runtime varies
by machine, and the program prints progress after each completed trial.

Examples:

```bash
python3 benchmark.py --workloads mixed locality --seeds 0 1 2 3 4 \
  --capacities 10 20 40 --output results/my_experiment

python3 benchmark.py --min-commit 1 --output results/commit_1
python3 benchmark.py --min-commit 20 --output results/commit_20
python3 benchmark.py --min-commit 50 --output results/commit_50

python3 benchmark.py --help
```

The commitment sweep uses the same seeds and traces when other options are
unchanged. It compares fixed commitment lengths; the agent does not learn the
length. Use separate validation experiments before choosing a final setting,
and reserve fresh trial seeds for final stochastic-workload tests. The program
does not automatically tune parameters or provide a dedicated validation split.
Deterministic traces still repeat regardless of seeds, as explained below.

If you explicitly change `--episodes` or `--length`, that changes the training
budget or workload length. `--quick` overrides only seeds and capacities.
Every run writes a new timestamped folder by default. `--output` must name a
new folder; existing results are not overwritten.

## How the original method works

A shared `FrameTable` tracks insertion time, last-use time, and resident access
frequency for every page. Policies choose victims using this metadata:

| Policy | Victim |
|---|---|
| FIFO | Earliest insertion |
| LRU | Earliest last use |
| LFU | Lowest resident access frequency; ties use earliest insertion |
| MRU | Most recent last use |
| OPT | Furthest future use, including never used again |

The same frame contents and metadata remain when the RL agent switches policy.
LFU counts reset when a page is reinserted. OPT requires future knowledge and
is an offline bound for this simulation, not an implementable online policy.

The RL state contains four discretized features of **past** accesses:
unique-page ratio, maximum frequency ratio, sequential ratio, and short-lookback
repeat ratio. Each feature has five buckets. The action is one of FIFO, LRU,
LFU, or MRU.

### Windowed RL

Select a policy for the next fixed-size reference window. State comes from the
preceding window, with a default state for the first decision. Reward is the
hit rate of the just-completed window.

### Fixed-commitment eviction-based RL

At a full-cache miss, choose a policy and keep it for `min_commit` evictions.
Hits do not advance the eviction counter. Once the count is reached, the next
eviction triggers another decision; intervening hits remain in the old block.
The final block may be shorter when the trace ends.

The original interface and GUI call this **per-eviction**, but that is literally
true only when `min_commit=1`. At the default 20, it is a fixed-commitment agent.
The new benchmark labels it `RL-FixedCommit-20` to make that distinction clear.

Its actual block reward is:

```text
reward = min(hits_in_block / (capacity * max(evictions_in_block, 1)), 1)
```

For commitment 1, this reduces to a normalized hit streak after one eviction.
For larger commitments, reward covers the whole block. This is a heuristic
signal, not a direct measurement of whether an individual eviction was wrong.

Both variants use the original update:

```text
Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
```

Defaults remain alpha=0.2, gamma=0.5, epsilon=0.2, epsilon decay=0.995, and
minimum epsilon=0.02. Epsilon decays after each learning decision/block.
Evaluation selects greedily without learning updates; ties are randomized.
The code passes `greedy=True` rather than literally setting epsilon to zero.

## Workloads and interpretation

| Workload | Actual implementation | What changes with generator seed? |
|---|---|---|
| Sequential | Repeated scan of the full page range, always starting at zero | Nothing |
| Looping | Repeated scan of `max(2, page_range // 3)` pages | Nothing |
| Random | Uniform page requests | Request sequence |
| Locality | Fixed hot set with default 80/20 access bias | Request sequence; hot-set identity stays fixed |
| Mixed | Randomly selected 200-reference phases | Phase choices and stochastic requests |

The locality hot set does **not** move during the trace. A loss on locality
cannot be explained by rapidly shifting hot pages in this generator.
Sequential and looping are related cyclic patterns at different working-set
sizes; they are not independent real application datasets.

### What a trial seed means

A seed initializes a pseudorandom generator so its choices can be repeated.
The same code, parameters, and seeds reproduce hit/fault counts. Changing seeds
can change both training exploration and stochastic reference strings.

Trial 0 intentionally uses the original seeds: training traces 0..episodes-1,
evaluation trace 999999, agent 42, and LeCaR implementation 0. Later trials use
separately derived seeds for training traces, evaluation traces, and agents.
Seed derivation uses SHA-256 rather than Python's process-randomized `hash()`.

All methods in a trial see the **same evaluation trace** and capacity. Both RL
variants receive the same training traces in the same order. The original
agent RNG continues from training into evaluation; the wrapper does not reset
it or alter exploration. Traces are also shared across capacities for each
workload/trial seed.

### Important distinction: repeated traces versus held-out traces

To preserve the original project, the benchmark does not modify the workload
generators. Therefore **sequential and looping training/evaluation traces are
identical at fixed parameters**, regardless of seed. Their repeated trials
measure sensitivity to agent randomness, not generalization to unseen traces.
Classic fixed policies have identical results across seeds on these traces.

Training/evaluation hashes and exact overlap counts are recorded for every
trial. The CLI prints a notice when overlaps occur. No duplicates are silently
removed and no substitute workload distribution is introduced. Stochastic
traces can also coincide by chance; inspect their overlap counts before
claiming held-out evaluation.

The benchmark trains separately within each workload family. Results do not
show that one universal agent generalizes to unknown applications.

## Output files and statistics

| File | Contents |
|---|---|
| `raw_results.csv` | Every method/trial: hits, faults, rates, timings, seeds, policy usage, and trace overlap flags |
| `summary.csv` | Means, sample standard deviations, and wins/ties/losses against the best classic fixed policy |
| `paired_comparisons.csv` | Paired differences against individual methods, including fixed-commitment versus windowed RL |
| `trace_manifest.jsonl` | Training/evaluation seeds, trace hashes, distinct evaluation pages, and interpretation flags |
| `config.json` | Settings, code hashes, Python/platform details, timestamps, and completion status |

Sample SD is blank for a single trial because it cannot be estimated from one
observation. Mean/SD and win counts are descriptive summaries; they are not
p-values, significance claims, or confidence intervals.

`W/T/L` means wins/ties/losses by fault count. **Best-fixed** means the best of
FIFO/LRU/LFU/MRU on each individual evaluation trace, selected in hindsight.
It does not mean the fixed-commitment RL variant. A deployed system does not
necessarily know that best classic policy beforehand.

In paired comparisons, a positive fault-rate reduction in percentage points
means the first method incurred fewer faults on the same trace. Means of
paired differences are preferable to treating unrelated trials as pairs.

```text
hit_rate = hits / references
fault_rate = faults / references
fault_reduction_pct = 100 * (baseline_faults - method_faults) / baseline_faults
LRU_to_OPT_gap_closed_pct = 100 * (LRU_faults - method_faults) / (LRU_faults - OPT_faults)
```

Gap closed is undefined when LRU matches OPT. A negative gap means worse than
LRU; 100% means matching OPT on that trace. It is not a percentage reduction
in page faults. Compulsory initial misses are included in all fault counts.

`exec_time_s` measures the complete simulation call in Python. Training time
covers calls to the original trainers with pre-generated traces; generation
and hashing are outside the timed region. Neither is real application or
kernel execution time. The original internal `decision_time_s` is omitted
from exported CSVs because its timing boundaries omit some feature/update
work; the original program still returns that historical field.

Completed trial rows are flushed to disk as the run progresses. On interruption,
metadata is marked accordingly; summaries are written only after completion.
Automatic resume is not implemented.

## Original default result: a single run, not a general claim

These hit rates are reproduced with the original default settings and seed
choices. The new benchmark's trial 0 at capacity 20 matches them.

| Workload | Best classic policy | Windowed RL | Fixed commitment 20 | OPT |
|---|---:|---:|---:|---:|
| Sequential | MRU: 19.00% | 18.37% | 19.00% | 19.00% |
| Random | MRU: 21.60% | 20.13% | 19.10% | 51.70% |
| Looping | MRU: 58.90% | 58.27% | 58.90% | 58.90% |
| Locality | LFU: 75.43% | 58.07% | 57.83% | 79.37% |
| Mixed | LFU: 25.33% | 26.83% | 39.97% | 58.57% |

The mixed result reduces faults from 2,240 (LFU) to 1,801 (fixed-commitment RL),
about 19.6%, on this one trace. Matching OPT on cyclic traces is not proof of
universal optimality: fixed MRU also matches it there. On locality, LFU beats
both RL variants and the included LeCaR implementation.

`results/included_benchmark/` contains the full 150-trial original-algorithm
benchmark. See `BENCHMARK_NOTES.md` for the measured multi-seed results and
known interpretation limits. Do not reuse results from the superseded
adaptive-duration ZIP; that version changed the learning objective and data.

## Research context and corrections

- [LeCaR, HotStorage 2018](https://www.usenix.org/conference/hotstorage18/presentation/vietri)
  learns LRU/LFU weights from regret feedback.
- [CACHEUS, FAST 2021](https://www.usenix.org/conference/fast21/presentation/rodriguez)
  includes expert combinations such as scan-resistant LRU and churn-resistant
  LFU; it is not simply ordinary LRU/LFU mixing.
- [Learning-based Page Replacement, 2025](https://www.nature.com/articles/s41598-025-88736-4)
  already studies learned LRU/MRU selection. Including MRU alone is not a new
  research contribution.

The existing `lecar.py` is an **independent LeCaR-inspired implementation**,
not a validated reproduction of the paper. It permits up to capacity history
entries per expert (2C overall), while the paper describes a total C-entry
history. It also uses this project's resident LFU counters and tie rule.
The benchmark labels it `LeCaR-inspired`; the original CLI/GUI retains its
historical `LeCaR` label because those files are unchanged. Validate against
the [authors' implementation](https://github.com/sylab/cacheus) before making
strong claims about published LeCaR performance.

The RL agents receive offline pretraining while LeCaR-inspired starts learning
on the evaluation trace. These are different information budgets and must be
disclosed. This project's results do not establish why prior authors chose
particular experts or that their methods cannot handle other policies.

The README corrects several historical source comments without changing the
original files: the default method is fixed-commitment rather than literally
per-eviction; deterministic traces are not held out; and
`/proc/<pid>/pagemap` exposes mappings/status, **not an ordered memory-access
trace**. See the [kernel documentation](https://docs.kernel.org/admin-guide/mm/pagemap.html).
Real program access traces require suitable instrumentation. Storage block
traces support storage-cache experiments and should be labeled accordingly.

## Preserved limitations and report scope

This is a simulation study, not a kernel implementation. It assumes equal-sized
pages, fixed frame capacity, an initially empty cache, and unit fault counts.
It does not model dirty-page writeback, storage latency, competing processes,
prefetching, or NUMA. Reduced simulated faults do not alone prove application
speedups.

The requested scope preserves known original implementation limitations:

- Q-updates bootstrap at the final block without a terminal flag.
- The state omits frame contents; Q-learning convergence is not established.
- Commitment blocks use a heuristic reward and per-decision discounting.
- Greedy evaluation can add zero-initialized Q-table entries for unseen states,
  although no learning updates occur.
- Original APIs/GUI have limited input validation; the new benchmark validates
  its own inputs before calling them.
- The original OPT implementation has O(N*C*log N) worst-case time, rather than
  the simplified O(N*C) stated in its historical docstring.

These are documented, not silently fixed, so the original numerical behavior
remains reproducible. Future fixes should be evaluated as separate changes.

A suitable report title is **Workload-Aware Adaptive Page Replacement Using
Q-Learning: A Comparative Simulation Study**. Explain the OS concepts, shared
frame design, state/action/reward, training protocol, fair comparisons,
variability, runtime boundaries, deterministic-trace caveat, and failure cases.
The policy selection is adaptive; the commitment length is fixed.

## Files and verification

Original files: `evaluate.py`, `gui.py`, and the seven files in
`page_replacement/`. Added files support automated evaluation: `benchmark.py`,
`tests/test_benchmark.py`, source hashes, benchmark results, and benchmark notes.
Only the original README is replaced.

```bash
python3 -m unittest discover -s tests -v
```

Tests verify original source hashes, exact trial-0 compatibility across all
five workloads, trace pairing, repeated-run determinism, overlap reporting,
summary statistics, input validation, CSV exports, and overwrite protection.
