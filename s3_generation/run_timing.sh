#!/bin/bash
# Timing campaign: the baseline vocabulary vs the budget member on three sizes.
# Run from the repository root after `make gpu`.
export GITC_DATA=instances
export GITC_CAP2=results/baseline/cap2-results.csv
T=bin/timing
BUD="NSTW,MAXN,NFEAS,REACH5"
for inst in c101 r201 pr15; do
  $T $inst --seeds 30 2>&1 | tee -a results/timing/timing_bench.log
  $T $inst --terms $BUD --seeds 30 2>&1 | tee -a results/timing/timing_bench.log
done
echo TIMING_DONE >> results/timing/timing_bench.log
