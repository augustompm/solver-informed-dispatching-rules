# s2_terminals

Route readers used to design the constructed terminals. `lookit` compares a
reference solver route against the 30 NS-GP rules of an instance, move by move
(coverage, lost customers, route mechanics). `lookit_mech` measures literature
mechanisms (M1-M4) on the same routes before any terminal is implemented.

The reading identifies which operations matter; the terminals implementing them
come from the scheduling and orienteering literature, plus two novel composites.
Terminal code lives in `engine/sim_terms.hpp` (code ids are kept: the kernels
are verified bit-exact against them).

| code id | paper name | origin |
|---|---|---|
| REGRET | REGRET | expected regret, Bian & Liu |
| ATRISK | IAP | in-time arrival probability, Bian & Liu |
| NFEAS | NFEAS | analogue of NIQ (number in queue), JSS terminal tables |
| REACH1/2/3/5 | REACHk | k-step greedy lookahead, generalizing 1-step NPT-style lookahead |
| NSTW | NSTW | NS (Jackson & Mei) restricted to reachable time windows |
| NSROB | NSROB | NS under +1 sigma service overrun |
| REACH2ROB | REACH2ROB | REACH2 under +1 sigma |
| MAXN | MAXNS | NS with max aggregation (best single successor) |
| PBUST | IAPD | IAP of the best downstream successor (closed-form Gaussian tail) |
| FRAGCNT | NLOST | count of the window cascade killed by a choice (structural regret) |
| FRAGSCORE | SLOST | the same cascade weighted by prize |
| SAT | HFRAC | fraction of the horizon consumed (from RemT, Mei & Zhang) |
| VFRAC | VISFRAC | fraction of customers already visited |
| HARVEST | LOSSR (novel) | SLOST/(NS+1), cascade damage per reachable value |
| BOLSAO | NETLOSS (novel) | SLOST - NS, net damage of the choice |

The 11 baseline terminals (SCORE ... RemT, NS) are inherited unchanged from
Mei & Zhang (2018) and Jackson & Mei (2020).
