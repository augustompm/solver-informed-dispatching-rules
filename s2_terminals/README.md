# s2_terminals

Route readers used to design the constructed terminals. `lookit` compares a
reference solver route against the 30 NS-GP rules of an instance, move by move
(coverage, lost customers, route mechanics). `lookit_mech` measures literature
mechanisms (M1-M4) on the same routes before any terminal is implemented.
The routes the reading was performed on are `results/s2_terminals/routes_read/`;
the 60 logs reproduce byte for byte against them. The reading predates the
final reference-gap routes of `results/s1_reference/routes/`, which supersede
them for headroom measurement only.

The reading identifies which operations matter; the terminals implementing them
come from the scheduling and orienteering literature, plus two novel composites.
Sixteen of the eighteen are grounded in published terminals. Terminal code
lives in `engine/sim_terms.hpp` (code ids are kept: the kernels are verified
bit-exact against them); the paper defines every terminal algebraically.
The kernel exposes 61 terminal slots in total (`engine/vocab.hpp`); the
campaign menu is the 18 below plus the 11 baseline terminals. The remaining
slots are exploration-era terminals that are not part of the final delivery,
kept so the kernels stay bit-exact against the recorded goldens and traces.

| code id | paper name | origin |
|---|---|---|
| REGRET | REGRET | expected regret, Bian & Liu; algebraically identical to SLOST (two menu genes, never elected together) |
| ATRISK | ATRISK | expected prize at risk downstream, each open successor weighted by one minus its in-time arrival probability (Bian & Liu); chance-constrained counterpart of SLOST |
| NFEAS | NFEAS | analogue of NIQ (number in queue), JSS terminal tables |
| REACH1/2/3/5 | REACHk | k-step greedy lookahead, generalizing 1-step NPT-style lookahead |
| NSTW | NSTW | NS (Jackson & Mei) with each successor weighted by the unspent fraction of its window |
| NSROB | NSROB | NS under +1 sigma service overrun |
| REACH2ROB | REACH2ROB | REACH2 under +1 sigma |
| MAXN | MAXNS | NS with max aggregation (best single successor) |
| PBUST | IAPD | Gaussian miss probability of the best downstream successor (one minus its in-time arrival probability) |
| FRAGCNT | NLOST | count of the window cascade killed by a choice (structural regret) |
| FRAGSCORE | SLOST | the same cascade weighted by prize (identical to REGRET by construction) |
| SAT | HFRAC | fraction of the horizon consumed (from RemT, Mei & Zhang) |
| VFRAC | VISFRAC | fraction of customers already visited (after the satisfied fraction of Liu et al. 2017, GPHH for UCARP) |
| HARVEST | LOSSR (novel) | SLOST/(NS+1), cascade damage per reachable value |
| BOLSAO | NETLOSS (novel) | SLOST - NS, net damage of the choice |

The 11 baseline terminals (SCORE ... RemT, NS) are inherited unchanged from
Mei & Zhang (2018) and Jackson & Mei (2020).
