# Literature search for the first-work claim

The claim (Sections 1 and 2 of the paper): no prior study reads the routes of a reference
solver to decide which terminals a GPHH should compose from, and none selects that
vocabulary instance by instance.

The search was made on 4 August 2026, on Scopus (TITLE-ABS-KEY, exports under `scopus/`)
and on Google Scholar (first page of results for each query, and a second pass restricted
to 2021 onwards for Q1). Screening was made by title and abstract.

## Q1: GPHH on orienteering

Scopus:
```
TITLE-ABS-KEY ( "orienteering" AND ( "genetic programming" OR "hyper-heuristic*" ) )
```
Scholar:
```
orienteering "genetic programming" OR "hyper-heuristic"
```

Scopus returned 8 records (`scopus/F1_4ab6d593.csv`). Three are the GPHH line on the
stochastic TOPTW: Mei 2018, Karunakaran 2019 and Jackson 2020. The query finds all three,
so it is not too narrow. The other records are four proceedings volumes, a genetic
algorithm that searches solutions (Wang 2024) and the deterministic Thief OP (Santos 2018).
On Scholar the same three papers are the first three results. Restricting to 2021 onwards,
no GPHH paper on the stochastic TOPTW appears. The two closest are a simheuristic for the
same problem (Rabe 2021, from the group already cited in the paper) and a GPHH for the
dynamic dial-a-ride problem (Huang 2025).

## Q2: rules or vocabulary informed by a solver

Scopus:
```
TITLE-ABS-KEY ( ( "genetic programming" OR "hyper-heuristic*" ) AND ( "dispatching rule*" OR "priority function*" OR "terminal set" OR "feature construction" ) AND ( "solver" OR "optimal solution*" OR "imitation" ) )
```
Scholar:
```
"genetic programming" OR "hyper-heuristic" "dispatching rule" OR "priority function" OR "terminal set" OR "feature construction" solver OR imitation
```

Scopus returned 13 records (`scopus/F2_84806b7d.csv`). They are the usual GPHH
shop-scheduling papers, where rules are evolved from scratch: Tay 2007, Park
2013/2015/2018, Đurasević 2018, Masood 2018/2022, Sun 2025. The "reference point" in
Masood is an NSGA-III concept, not a reference solver. The rest is out of scope
(machining, symbolic regression, proceedings volumes). Scholar brings the Zhang 2024
survey and Ingimundardottir 2018, both cited in the paper, and Braune 2022, where the
terminals are the usual job and machine attributes and the CP solver only benchmarks the
results. Nothing here takes its terminals from a solver's solutions.

## Q3: instance-specific heuristic generation

Scopus:
```
TITLE-ABS-KEY ( ( "instance-specific" OR "per-instance" ) AND ( "heuristic generation" OR "hyper-heuristic*" OR "dispatching rule*" ) )
```
Scholar:
```
"instance-specific" OR "per-instance" "heuristic generation" OR "hyper-heuristic" OR "dispatching rule"
```

Scopus returned 3 records (`scopus/F3_cf8e2bee.csv`): Mısır 2021, which generates
heuristics per instance by parameter configuration for the 2D HP protein model, Kaleta
2026, a neural constructive heuristic for the FJSP, and one proceedings volume. Scholar
adds the Burke 2013 survey, Zhang 2025 (the instance-specific LLM line, cited in the
paper; Scopus does not index the arXiv version yet), the SPOT thesis (Xue 2013, machine
learning that modifies existing heuristics using suboptima of subproblems) and Bacha
2019, which configures a genetic algorithm per instance. None of them selects a GP
terminal vocabulary per instance.

## Notes

- Shen 2025, the PRISMA survey of the OP (2017-2025, Web of Science, 112 studies), does
  not mention genetic programming or hyper-heuristics.
- A web search on 26 June 2026 located the nearest neighbours, all cited in the paper:
  Sim and Hart 2016, Wei 2026, Zhang 2025/2026, Yska 2018, Mei 2017, Zhang 2021,
  Ferreira 2022, Acero 2024.
