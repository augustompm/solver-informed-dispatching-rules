# Paired per-seed analysis of the FULL18 ablation arm (no selection) against
# the deployed SI-GP champion traces and the NS-GP baseline, on the 10 Holm
# instances. Pairing is by seed index under CRN (same RNG streams, same
# held-out draws, seed 1M). Wilcoxon signed-rank mirrors engine/stats.hpp:
# zero diffs dropped, average ranks over ties, tie-corrected variance,
# normal approximation, no continuity correction.
import json
import math
from pathlib import Path

ROOT = Path(".")
F18 = ROOT / "results/ga40/results_full18"
SGA = ROOT / "results/ga40/results"
NSG = ROOT / "results/baseline/nsgp"
INSTS = ["pr01", "c101", "c102", "c103", "c105",
         "r101", "r103", "r104", "r105", "rc104"]


def wilcoxon_paired(diffs):
    d = [x for x in diffs if x != 0.0]
    n = len(d)
    if n == 0:
        return 1.0, 0
    ad = sorted((abs(x), i) for i, x in enumerate(d))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ad[j + 1][0] == ad[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[ad[k][1]] = r
        i = j + 1
    wpos = sum(r for r, x in zip(ranks, d) if x > 0)
    wneg = sum(r for r, x in zip(ranks, d) if x < 0)
    t_stat = min(wpos, wneg)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    # tie correction over groups of equal |d|
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ad[j + 1][0] == ad[i][0]:
            j += 1
        t = j - i + 1
        if t > 1:
            var -= (t ** 3 - t) / 48.0
        i = j + 1
    if var <= 0:
        return 1.0, n
    z = (t_stat - mu) / math.sqrt(var)
    p = 2.0 * 0.5 * math.erfc(-z / math.sqrt(2.0))
    return min(p, 1.0), n


def trace_gains(path):
    j = json.load(open(path))
    return {f["seed"]: f["gain"] for f in j["finals"]}


def ns_gains(inst):
    scores = {}
    for f in (NSG / inst).glob("seed*.json"):
        j = json.load(open(f))
        scores[j["seed"]] = j["test_mean"]
    km = sum(scores.values()) / len(scores)
    return {s: 100.0 * (v - km) / km for s, v in scores.items()}


print(f"{'inst':6} {'F18':>6} {'SI':>6} {'dSI-F18':>8} {'W-L-T':>8} {'p':>9} | "
      f"{'dF18-NS':>8} {'W-L-T':>8} {'p':>9}")
sum_f, sum_s = 0.0, 0.0
for inst in INSTS:
    gf = trace_gains(F18 / f"trace_{inst}_FULL18.json")
    gs = trace_gains(SGA / f"trace_{inst}_Sga.json")
    gn = ns_gains(inst)
    seeds = sorted(gf)
    assert seeds == sorted(gs) == sorted(gn) and len(seeds) == 30
    d_sf = [gs[s] - gf[s] for s in seeds]
    d_fn = [gf[s] - gn[s] for s in seeds]
    p_sf, _ = wilcoxon_paired(d_sf)
    p_fn, _ = wilcoxon_paired(d_fn)
    mf = sum(gf.values()) / 30
    ms = sum(gs.values()) / 30
    sum_f += mf
    sum_s += ms
    wlt_sf = f"{sum(1 for x in d_sf if x > 0)}-{sum(1 for x in d_sf if x < 0)}-{sum(1 for x in d_sf if x == 0)}"
    wlt_fn = f"{sum(1 for x in d_fn if x > 0)}-{sum(1 for x in d_fn if x < 0)}-{sum(1 for x in d_fn if x == 0)}"
    print(f"{inst:6} {mf:+6.2f} {ms:+6.2f} {sum(d_sf)/30:+8.2f} {wlt_sf:>8} {p_sf:9.2e} | "
          f"{sum(d_fn)/30:+8.2f} {wlt_fn:>8} {p_fn:9.2e}")
print(f"{'mean':6} {sum_f/10:+6.2f} {sum_s/10:+6.2f}")
