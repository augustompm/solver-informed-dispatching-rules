#!/bin/bash
# Wave worker for the structured-population GA: each list line is one
# instance; runs the search with the approved configuration (singleton
# seeding without the full chromosome, uniform crossover, pm 0.5/L,
# 50-generation guard, stagnation-10 cutoff) and materialises the champion
# as a 30-seed trace tagged Sga.
# usage: run_ga40.sh <instance-list-file> <worker-name>
set -u
LIST=$1
WK=$2
R=$(cd "$(dirname "$0")/../.." && pwd)
U=REGRET,NSROB,MAXN,REACH2,NSTW,FRAGCNT,FRAGSCORE,PBUST,REACH1,REACH3,REACH5,SAT,VFRAC,NFEAS,HARVEST,BOLSAO,ATRISK,REACH2ROB
W=$R/results/ga40
mkdir -p $W/results $W/logs
cd $W
export GITC_DATA=$R/instances
export GITC_CAP2=$R/results/baseline/cap2-results.csv
while read -r inst; do
  [ -z "$inst" ] && continue
  echo "[$WK] $inst start $(date -u +%FT%TZ)" >> $W/logs/$WK.run
  $R/bin/ga $inst --universe $U --seeds neutral --nofull --pm 0.5 --gens 50 --stag 10 --top 6 --bmax 13 --cache logs/${inst}.evcache > logs/${inst}_search.log 2>&1
  CH=$(grep ^CHAMPION logs/${inst}_search.log | head -n 1)
  TERMS=$(echo "$CH" | sed 's/.*terms: //')
  echo "[$WK] $inst ${CH:-no-champion}" >> $W/logs/$WK.run
  if [ -n "$TERMS" ] && [ "$TERMS" != "(base)" ]; then
    $R/bin/gen_trace $inst --terms "$TERMS" --tag Sga --seeds 30 > logs/${inst}_trace.log 2>&1
    echo "[$WK] $inst trace done $(date -u +%FT%TZ)" >> $W/logs/$WK.run
  fi
  echo "[$WK] $inst end $(date -u +%FT%TZ)" >> $W/logs/$WK.run
done < $LIST
echo "GA40_${WK}_FINISHED $(date -u +%FT%TZ)" >> $W/logs/$WK.run
