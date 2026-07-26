# SI-GP: solver-informed dispatching rules for stochastic team orienteering

C++20/CUDA code, benchmark instances and campaign logs for SI-GP, a
solver-informed genetic programming hyper-heuristic that evolves dispatching
rules for the stochastic team orienteering problem with time windows (STOPTW).

Rules are evolved with genetic programming under the protocol of Mei & Zhang
(2018) and Jackson & Mei (2020). A structured-population genetic algorithm
selects, for each instance, which constructed terminals the GP may use. The
terminal ideas come from analyzing reference solver routes (PyVRP, NVIDIA cuOpt).

## Build

```
make cpu       # CPU tools (g++, -std=c++20)
make gpu       # CUDA tools (nvcc)
make verify    # exactness checks against verify/golden/
```

`make verify` runs the CPU gates; `make verify-f4` additionally checks the
CUDA kernel against the CPU simulator on a GPU machine (bit-exact).

## Run

All tools run from the repo root; paths default to `instances/` and
`results/baseline/` (override with `GITC_DATA` / `GITC_CAP2`).

```
bin/ga r105 --universe REGRET,NSROB,MAXN,REACH2,NSTW,FRAGCNT,FRAGSCORE,PBUST,REACH1,REACH3,REACH5,SAT,VFRAC,NFEAS,HARVEST,BOLSAO,ATRISK,REACH2ROB \
  --seeds neutral --nofull --pm 0.5 --gens 50 --stag 10 --top 6 --bmax 13

bin/gen_trace r105 --terms REACH5 --tag Sga --seeds 30
bin/paired r105 results/ga40/results/trace_r105_Sga.json
bin/choice r105 results/ga40/results/trace_r105_Sga.json --backup
bin/redeploy r105 results/ga40/results/trace_r105_Sga.json
bin/lookit r105
```

Campaign scripts: `results/ga40/run_ga40.sh` (full 40-instance GA campaign) and
`s3_generation/run_timing.sh`.

## Structure

```
engine/          shared engine: instance parser, rule trees, CPU/GPU simulator, RNG, GP loop
s1_reference/    replay of reference routes and the per-instance reference gap
s2_terminals/    route readers used to design the constructed terminals
s3_generation/   rule generation (gen_trace), timing
s4_selection/    GA subset search (ga.cu), choice with NS-GP backup, paired tests
verify/          bit-exact checks with golden fixtures
instances/       the 40 STOPTW benchmark instances (Solomon and Cordeau based)
results/         campaign logs, caches, 30-seed traces, reference routes,
                 and the NS-GP baseline (per-instance CSV + 30 per-seed runs)
```

## Evaluation

Every number is a mean over 30 independent GP runs, evaluated once on a held-out
set of 500 scenarios with common random numbers, against the NS-GP baseline
reproduced under the identical protocol. Significance: paired Wilcoxon with Holm
correction.

## References

- Y. Mei, M. Zhang. Genetic Programming Hyper-heuristic for Stochastic Team
  Orienteering Problem with Time Windows. IEEE CEC, 2018.
- J. Jackson, Y. Mei. Genetic Programming Hyper-heuristic with Cluster Awareness
  for Stochastic Team Orienteering Problem with Time Windows. IEEE CEC, 2020.
- C.F.M. Toledo, L. Oliveira, P.M. Franca. Global optimization using a genetic
  algorithm with hierarchically structured population. Journal of Computational
  and Applied Mathematics 261, 2014.
