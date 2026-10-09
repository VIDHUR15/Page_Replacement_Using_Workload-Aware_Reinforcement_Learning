# Benchmark notes — original implementation

The original algorithms, workload generators, training rules, and GUI are preserved. Only the README and a separate automated evaluation wrapper have been added or updated.

## Completed evaluation

- 5 workloads, 3 capacities (10/20/40), and 10 trial seeds (0–9): 150 trials.
- 8 original methods per trial: 1,200 result rows.
- 3,000 references per evaluation trace; 25 training episodes per RL variant.
- Fixed commitment remains 20 evictions; there is no learned-duration agent.
- Trial 0 preserves the original evaluator seeds and reproduces its hit/fault counts and policy usage.
- 7 tests passed, including byte-for-byte source preservation and numerical compatibility across all five original workloads.

## Results at 20 frames

Hit rates below are percentages, reported as mean +/- sample standard deviation across 10 trials. Wins/ties/losses compare fixed-commitment RL with the best classic policy selected in hindsight on each trace.

| Workload | Windowed RL | Fixed commitment 20 | Fixed-commitment W/T/L |
|---|---:|---:|---:|
| sequential | 14.76 +/- 7.78 | 15.20 +/- 8.01 | 0/8/2 |
| random | 20.10 +/- 0.76 | 20.23 +/- 0.77 | 2/0/8 |
| looping | 58.52 +/- 0.33 | 58.90 +/- 0.00 | 0/10/0 |
| locality | 62.42 +/- 4.59 | 67.19 +/- 5.62 | 1/1/8 |
| mixed | 35.90 +/- 9.82 | 36.36 +/- 8.96 | 4/0/6 |

## Interpretation limits

- The original sequential and looping generators ignore their seed. All 60 trials in those two families have identical training/evaluation traces within each configuration. Their variation measures agent randomness, not unseen-trace generalization.
- The other 90 trials have no exact training/evaluation trace overlap in this included run; inspect the manifests for hashes and seeds.
- Workload families retain their original parameters. No new workload distribution, reward, gamma, epsilon schedule, or terminal-update behavior has been introduced.
- Some synthetic configurations favor RL; others favor classic fixed policies. The original 39.97% mixed hit rate is one seed-0 result, not a universal performance guarantee.
- Source comments and original GUI labels are untouched. The corrected README explains the historical per-eviction label, the LeCaR-inspired approximation, and other known limitations.
- Training is separate for each family/capacity/trial. This does not establish a universal agent or real OS/application speedups.
- Statistics are descriptive, not significance tests. Runtime values are machine-dependent.

## Reproduce

```bash
python3 benchmark.py --output results/reproduction
```

To reproduce the original default settings with the new CSV exports:

```bash
python3 benchmark.py --quick
```

`--quick` uses one seed and 20 frames but retains 25 training episodes and 3,000 references unless explicitly overridden. See README.md for full options.
