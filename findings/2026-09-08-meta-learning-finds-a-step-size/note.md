# 2026-09-08 — Meta-learning finds a step size: the outer loop lands on the optimum the landscape names, online, and its controls on theirs

*Opened 2026-09-08 with the step-2 chunk (loom PR #9); redone 2026-09-16 after the first review
round, which sent the draft back as a lab notebook: no comparison on shared axes, and the loop run
in a regime where there was nothing to find. The words are `docs/meta.md`'s glossary: a pass
qualifies a signal, the inner loop consumes a residual, "objective" is the outer loop's word.*

**Question.** With the rule Δ = −η·s and only η learned, does the outer loop, differentiating the
objective J = L^soft(z_K) through K steps of the rule, find from a non-functional start the step
size the objective's own landscape in η names; does it fail to under a sign-flipped signal, and
find only "nothing moves" under a shuffled or an output-only one; and does the η it finds adapt a
fresh tile on a task the loop never saw?

**Floor and shape.** The flat tile `(4, 16, 8, 2)` at arity 4 on 2-juntas (4 → 2), the smoke shape
of steps 0 and 1; the inner cell `soft.relay.entry`, the exact adjoint to the entry, the one a soft
fabric could carry; the objective L^soft of the final state. The non-learned baseline is the same
objective at a fixed η, on the same members, across η (`recipes.landscape`): plain descent at a
fixed rate, which is the floor since `findings/2026-09-08-the-floors-under-plain-descent/note.md`.

**Conditions.** `ONLINE_META` (`soft.relay.entry -> L^soft`): one case per step (W = 1), K = 64
steps of the rule per rollout, a batch of 16 fresh tiles on 16 fresh tasks per outer step, 500
outer steps of Adam at 0.03 on log η from η₀ = 0.01, the gradient's global norm clipped at 1.
Rates are per vote, so the untrained objective is 0.25 (the soft read of a random table is 0.5 on
every output) and the untrained deployed loss is about 0.5 (its bits are wrong half the time). The
controls swap the recipe's `control` and nothing else. `BATCHED_META` (every case per step, K =
16, 200 outer steps at 0.1), the first draft's regime, appears in the path only. Every run on the
CPU; the sweep on three recipe seeds, the learned column on six.

## The finding

One table, every row on the same axes: online, K = 64, the same members for the sweep and the
loop's first step, the held-out check 500 steps of the rule at the η found on a fresh tile and a
fresh task from the seed's held-out half. The sweep column is the mean over three seeds; the
learned columns are the range over six seeds of the tail, the median over the last 200 outer
steps, where the loop settles (the deployed loss is that of the final states).

| signal | swept objective: shape, best η, J there | learned η | J at the learned η | deployed loss there | held-out A^hard, driven by this signal | the same η, driven by the true signal |
|---|---|---|---|---|---|---|
| true | an interior optimum: **30**, 0.0025 (20 → 0.0041, 50 → 0.0033, 100 → 0.092, 200 → untrained) | **23.0 to 33.0** | 0.0019 to 0.0035 | 0.0020 to 0.0137 | **1.000 ×6** | 1.000 ×6 |
| flipped | monotone worse: 0.43 at η = 3, 0.50 from 100 on; best **0** | 2.3e-04 to 2.5e-04, ×40 below the start | 0.253 to 0.253 | 0.46 to 0.54 | 0.34 to 0.53 | 0.34 to 0.53 |
| shuffled | monotone worse: 0.254 at 3, 0.44 at 400; best **0** | 0.60 to 0.65 | 0.2523 to 0.2524 | 0.46 to 0.53 | 0.34 to 0.59 | 0.69 to 0.97 |
| output-only | flat to worse: 0.251 at 3, 0.278 at 1000; best **0** | 1.15 to 1.28 | 0.2506 to 0.2507 | 0.43 to 0.51 | 0.34 to 0.66 | 1.000 ×6 |

The outer loop finds the minimum of whatever landscape the signal hands it, and only the true
signal hands it a landscape with an interior minimum. Under the true signal the sweep's optimum is
30, the floor (J within four times its minimum) is η from 20 to 50, and the loop lands at 29 to 33
on every seed, with the objective and the deployed loss both at that floor and the held-out tile
at 1.000. Under the three controls the sweep says the best η is zero, and the loop goes there: it
walks η down under the flip, and parks it at a size where nothing moves under the two information
controls, the objective at untrained on every seed. The two held-out columns say two different
things. Driven by its own signal, every control leaves the fresh tile at chance. Driven by the true
signal, the η a control parked at is a slow but working step size, which is a fact about the true
signal and not about the control; the first draft reported only that column, and it read as a
control result.

**The landscape** (`2026-09-16-landscape.py`; `recipes.landscape` at each η, online; cells: J per
seed, then the mean J and the mean deployed loss in bold):

| η | true | flipped | shuffled | output-only |
|---|---|---|---|---|
| 3 | 0.2150 · 0.2145 · 0.2148 (**0.2148 / 0.3008**) | 0.4510 · 0.4140 · 0.4318 (**0.4322 / 0.5566**) | 0.2558 · 0.2540 · 0.2538 (**0.2545 / 0.4948**) | 0.2520 · 0.2504 · 0.2512 (**0.2512 / 0.4616**) |
| 10 | 0.0281 · 0.0229 · 0.0401 (**0.0303 / 0.0254**) | 0.5643 · 0.6036 · 0.5307 (**0.5662 / 0.5768**) | 0.2641 · 0.2629 · 0.2587 (**0.2619 / 0.4831**) | 0.2567 · 0.2560 · 0.2509 (**0.2545 / 0.4251**) |
| 20 | 0.0033 · 0.0038 · 0.0052 (**0.0041 / 0.0052**) | 0.5220 · 0.6080 · 0.5468 (**0.5589 / 0.5638**) | 0.2737 · 0.2734 · 0.2624 (**0.2698 / 0.5117**) | 0.2613 · 0.2658 · 0.2526 (**0.2599 / 0.4082**) |
| 30 | 0.0010 · 0.0053 · 0.0011 (**0.0025 / 0.0046**) | 0.5312 · 0.5451 · 0.5448 (**0.5404 / 0.5417**) | 0.2771 · 0.2734 · 0.2698 (**0.2734 / 0.4896**) | 0.2651 · 0.2717 · 0.2558 (**0.2642 / 0.4062**) |
| 50 | 0.0004 · 0.0044 · 0.0052 (**0.0033 / 0.0039**) | 0.5312 · 0.5312 · 0.5625 (**0.5416 / 0.5417**) | 0.3069 · 0.2869 · 0.2996 (**0.2978 / 0.4857**) | 0.2702 · 0.2772 · 0.2612 (**0.2695 / 0.4089**) |
| 100 | 0.0872 · 0.0885 · 0.1008 (**0.0922 / 0.1100**) | 0.5156 · 0.5000 · 0.5000 (**0.5052 / 0.5052**) | 0.3709 · 0.3870 · 0.3462 (**0.3680 / 0.4753**) | 0.2753 · 0.2800 · 0.2676 (**0.2743 / 0.4056**) |
| 200 | 0.2121 · 0.2132 · 0.2142 (**0.2131 / 0.2337**) | 0.5000 · 0.5000 · 0.5000 (**0.5000 / 0.5000**) | 0.4262 · 0.3699 · 0.4108 (**0.4023 / 0.4772**) | 0.2792 · 0.2782 · 0.2712 (**0.2762 / 0.4102**) |
| 400 | 0.3359 · 0.2272 · 0.2212 (**0.2614 / 0.2806**) | 0.5000 · 0.5000 · 0.5000 (**0.5000 / 0.5000**) | 0.4381 · 0.4454 · 0.4406 (**0.4414 / 0.4928**) | 0.2807 · 0.2766 · 0.2729 (**0.2767 / 0.4115**) |
| 1000 | 0.3744 · 0.3084 · 0.3313 (**0.3380 / 0.3548**) | 0.5000 · 0.5000 · 0.5000 (**0.5000 / 0.5000**) | 0.4420 · 0.4385 · 0.4208 (**0.4338 / 0.4811**) | 0.2798 · 0.2768 · 0.2759 (**0.2775 / 0.4095**) |

Online the true signal's landscape is sharp: a factor of three in η either side of 30 costs a
factor of ten in the objective, and by 200 the loss is back at untrained, because a single case's
exact signal at a large step overwrites what the other fifteen cases wrote. The flipped landscape
climbs to 0.5, every bit wrong. The shuffled and the output-only landscapes rise from untrained at
every η above zero: the output gates alone, one case at a time, cannot lower the all-case loss,
where with every case per step they bought a fourteenth (the path).

**The runs** (`2026-09-16-learned.py`; `recipes.train` under `ONLINE_META` and its control swaps;
cells: η / J at outer steps 100, 200, 300, 400, 500; the tail, the medians of η and J over the last
200 outer steps, where the loop settles; the deployed loss of the final states at 500; the held-out
accuracy at the tail-median η, under the control and under the true signal):

| control | seed | η / J along the run | tail: median η / J over steps 300 to 500 | deployed at 500 | held-out |
|---|---|---|---|---|---|
| none | 0 | 0.719/0.2475 · 27.7/0.0027 · 24.8/0.0031 · 27.5/0.0012 · 29.1/0.0027 | 28.3 / 0.0019 | 0.0039 | 1.000 / 1.000 |
| none | 1 | 0.733/0.2486 · 26/0.0037 · 24.8/0.0028 · 23.8/0.0008 · 32.5/0.0020 | 24.4 / 0.0019 | 0.0020 | 1.000 / 1.000 |
| none | 2 | 0.739/0.2482 · 28.1/0.0080 · 21.1/0.0032 · 24.7/0.0035 · 29.9/0.0065 | 24.7 / 0.0022 | 0.0098 | 1.000 / 1.000 |
| none | 3 | 0.718/0.2479 · 25.7/0.0058 · 21.2/0.0050 · 26.4/0.0019 · 30.7/0.0015 | 26.5 / 0.0019 | 0.0020 | 1.000 / 1.000 |
| none | 4 | 0.717/0.2495 · 27/0.0012 · 22.6/0.0017 · 22.9/0.0069 · 29.1/0.0100 | 23 / 0.0028 | 0.0137 | 1.000 / 1.000 |
| none | 5 | 0.718/0.2487 · 34.1/0.0143 · 65.7/0.0322 · 33.1/0.0162 · 29.6/0.0036 | 33 / 0.0035 | 0.0039 | 1.000 / 1.000 |
| flipped | 0 | 0.00146/0.2526 · 0.000632/0.2513 · 0.000372/0.2545 · 0.00025/0.2536 · 0.000182/0.2523 | 0.00025 / 0.2529 | 0.5195 | 0.469 / 0.469 |
| flipped | 1 | 0.00144/0.2522 · 0.000622/0.2528 · 0.000361/0.2526 · 0.000241/0.2530 · 0.000173/0.2527 | 0.00024 / 0.2530 | 0.4570 | 0.438 / 0.438 |
| flipped | 2 | 0.00144/0.2526 · 0.000635/0.2519 · 0.000376/0.2530 · 0.00025/0.2532 · 0.000182/0.2520 | 0.00025 / 0.2529 | 0.4707 | 0.344 / 0.344 |
| flipped | 3 | 0.00142/0.2525 · 0.000604/0.2530 · 0.000355/0.2523 · 0.000239/0.2543 · 0.000173/0.2531 | 0.000239 / 0.2529 | 0.5352 | 0.406 / 0.406 |
| flipped | 4 | 0.00146/0.2538 · 0.000628/0.2526 · 0.000367/0.2542 · 0.000245/0.2539 · 0.000178/0.2521 | 0.000245 / 0.2531 | 0.4941 | 0.531 / 0.531 |
| flipped | 5 | 0.00138/0.2532 · 0.000597/0.2522 · 0.000347/0.2513 · 0.000231/0.2534 · 0.000169/0.2529 | 0.000231 / 0.2528 | 0.5000 | 0.438 / 0.438 |
| shuffled | 0 | 0.45/0.2520 · 0.593/0.2504 · 0.654/0.2538 · 0.67/0.2526 · 0.636/0.2517 | 0.644 / 0.2523 | 0.5137 | 0.406 / 0.969 |
| shuffled | 1 | 0.421/0.2518 · 0.718/0.2518 · 0.687/0.2524 · 0.579/0.2521 · 0.693/0.2517 | 0.636 / 0.2524 | 0.4570 | 0.531 / 0.969 |
| shuffled | 2 | 0.432/0.2519 · 0.558/0.2516 · 0.553/0.2528 · 0.563/0.2528 · 0.636/0.2519 | 0.596 / 0.2524 | 0.4727 | 0.344 / 0.688 |
| shuffled | 3 | 0.44/0.2515 · 0.79/0.2529 · 0.594/0.2520 · 0.589/0.2535 · 0.672/0.2525 | 0.648 / 0.2523 | 0.5273 | 0.500 / 0.906 |
| shuffled | 4 | 0.433/0.2529 · 0.606/0.2522 · 0.503/0.2537 · 0.61/0.2532 · 0.694/0.2524 | 0.599 / 0.2524 | 0.5195 | 0.594 / 0.844 |
| shuffled | 5 | 0.421/0.2525 · 0.673/0.2517 · 0.589/0.2511 · 0.639/0.2520 · 0.624/0.2522 | 0.624 / 0.2523 | 0.4922 | 0.469 / 0.750 |
| output-only | 0 | 0.6/0.2504 · 0.985/0.2492 · 1.63/0.2507 · 1.27/0.2508 · 1.24/0.2507 | 1.25 / 0.2507 | 0.4980 | 0.594 / 1.000 |
| output-only | 1 | 0.622/0.2511 · 1.24/0.2497 · 1.2/0.2505 · 1.07/0.2495 · 1.36/0.2499 | 1.21 / 0.2506 | 0.4258 | 0.531 / 1.000 |
| output-only | 2 | 0.616/0.2504 · 1.27/0.2502 · 1/0.2504 · 1.16/0.2507 · 1.36/0.2511 | 1.22 / 0.2506 | 0.4551 | 0.344 / 1.000 |
| output-only | 3 | 0.595/0.2506 · 1.28/0.2514 · 1.01/0.2510 · 1.12/0.2510 · 1.32/0.2504 | 1.19 / 0.2506 | 0.5059 | 0.438 / 1.000 |
| output-only | 4 | 0.613/0.2515 · 1.06/0.2511 · 1.21/0.2517 · 1.14/0.2511 · 1.1/0.2507 | 1.28 / 0.2507 | 0.4727 | 0.656 / 1.000 |
| output-only | 5 | 0.601/0.2512 · 1.39/0.2502 · 1.09/0.2508 · 1.3/0.2496 · 1.07/0.2505 | 1.15 / 0.2506 | 0.4707 | 0.594 / 1.000 |

Under the true signal η climbs from 0.01 by a constant factor per outer step (0.72 at step 100 is
six doublings), reaches the optimum near step 200 and stays, one seed making a transient to 66 at
step 300 and returning; the deployed loss at the end is the objective's size, so the deploy gap on
this tile is closed by the rule that trained it. Under the flip η can only shrink, and does. Under
the shuffled signal η rises from 0.01 to 0.6 and stops: the layer's mean, which a permutation
keeps, is a real component of the gradient and carries the loop to the edge of the landscape's
flat part, and no further. Under output-only the same, at 1.

## Path

1. *The regime.* The first draft ran the loop with every case per step (`BATCHED_META`). That
   landscape is shallow: `2026-09-16-landscape.py` under `BATCHED_META` gives, for the true
   signal, J = 0.2468 · 0.2238 · 0.1349 · 0.0578 · 0.0079 · 0.0006 · 0.0032 · 0.0136 · 0.0225 at
   η = 3 to 1000 (means over the seeds), a minimum at 100 with the loss ten times below untrained
   from 50 to 1000; and the loop landed at η = 127, 130 and 119 (J = 0.0004 to 0.0005, the deployed loss 0.0000) (`2026-09-16-learned.py`, three
   seeds, 200 outer steps at 0.1), inside a band where any η would have done. The draft read the
   flatness as a property of the tile ("the exact soft signal cannot overshoot what a sixteen-entry
   table absorbs"): it is a property of the window. Online the same signal overshoots from 100 on
   and the landscape has an optimum, so online is where a found η is a finding. The batched
   regime keeps a line here and nothing more.
2. *The objective.* The first design carried the mean of the online losses, on the reading that a
   deployed tile experiences the stream. Gabriel asked whether back-propagation through the K
   steps did not make the final loss enough, and whether the mean did not weight the steps. It
   does (`docs/meta.md`): the final loss became the objective, the online losses a record, and
   since this round the deployed loss of the final states is a second record, so the deploy gap is
   a column of every run and never a footnote.
3. *The parameterisation and the outer step.* η = exp(raw) rather than softplus: at a
   non-functional start the two agree, but a constant factor per Adam step reaches a working scale
   in log(scale)/lr steps, and the flipped control then drives raw down geometrically. Adam on the
   host is kept for the same reason: the gradient on raw is η times the gradient on η and vanishes
   exactly where the check starts. The draft climbed at 0.1 for 200 steps
   (`2026-09-16-outer-step.py`, `ONLINE_META` at that schedule, ten seeds; cells: η / J at outer
   steps 25, 50, 75, 100, 150, 200; then the held-out accuracy at the η found):

   | seed | η / J along the run | held-out |
   |---|---|---|
   | 0 | 0.144/0.253 · 4.12/0.188 · 59.1/0.001 · 172/0.211 · 80.8/0.053 · 417/0.273 | 0.531 |
   | 1 | 0.148/0.252 · 4.23/0.206 · 35.9/0.003 · 74.1/0.036 · 29.1/0.001 · 32.6/0.007 | 1.000 |
   | 2 | 0.146/0.252 · 4.34/0.197 · 52.3/0.014 · 32.5/0.005 · 29.1/0.002 · 24.3/0.008 | 1.000 |
   | 3 | 0.146/0.252 · 4.23/0.177 · 44.3/0.008 · 23.5/0.003 · 21.6/0.001 · 26.1/0.006 | 1.000 |
   | 4 | 0.147/0.252 · 4.34/0.195 · 49.2/0.017 · 27.6/0.001 · 26.9/0.008 · 27.2/0.001 | 1.000 |
   | 5 | 0.142/0.250 · 4.16/0.197 · 45.6/0.006 · 28/0.001 · 24.6/0.006 · 26.8/0.023 | 1.000 |
   | 6 | 0.145/0.251 · 4.29/0.200 · 61.1/0.015 · 85/0.077 · 51.4/0.039 · 31/0.001 | 1.000 |
   | 7 | 0.145/0.251 · 4.32/0.195 · 42.7/0.010 · 19.4/0.003 · 22.5/0.020 · 36.4/0.003 | 1.000 |
   | 8 | 0.146/0.254 · 4.24/0.194 · 46.4/0.001 · 20.3/0.004 · 23.5/0.016 · 17/0.003 | 1.000 |
   | 9 | 0.147/0.252 · 4.4/0.187 · 27.2/0.015 · 22.7/0.002 · 22.5/0.003 · 30.4/0.004 | 1.000 |

   Nine seeds find the optimum; one overshoots it (172 at step 100), comes back to 81, and runs
   away to 417, where the objective is worse than untrained and the outer gradient sign-unreliable
   (item 5). At 0.03 for 500 steps every seed settles (the runs above; seed 5's transient to 66
   recovers). The runaway is the outer schedule, not the signal: the recipe climbs at 0.03.
4. *What the shuffled control leaks, and why its claim is on the loss.* A permutation within a
   layer keeps the layer's mean, and the mean is a real component of the gradient. With every case
   per step that leak carried the loop to η between 18 and 102 (the first draft), on a landscape
   flat there; online it carries it to 0.6, the edge of the landscape's flat part. The claim is
   worded on the loss on both regimes and η is reported; the sign flip is the only control whose
   η is a result by construction.
5. *The derivative path, per cell* (`2026-09-16-meta-gradient-paths.py`, K = 8; cells: the full
   outer gradient / the first-order one, the signal as data / a central finite difference with
   h = 0.5; three tiles). Which cells carry a second-order term:

   | cell | η | tile 0 | tile 1 | tile 2 |
   |---|---|---|---|---|
   | soft.relay.entry | 3 | −7.08e−4 / −7.85e−4 / −7.09e−4 | −4.74e−4 / −5.15e−4 / −4.74e−4 | −9.39e−4 / −9.44e−4 / −9.39e−4 |
   | soft.relay.entry | 30 | −1.50e−3 / −7.79e−4 / −1.50e−3 | −3.11e−3 / −1.40e−3 / −3.12e−3 | −5.17e−3 / −2.74e−3 / −5.17e−3 |
   | soft.relay.logit | 3 | −2.13e−4 / −2.18e−4 / −2.13e−4 | −1.43e−4 / −1.46e−4 / −1.43e−4 | −2.09e−4 / −2.10e−4 / −2.09e−4 |
   | soft.relay.logit | 30 | −1.01e−4 / −1.18e−4 / −1.01e−4 | −8.30e−5 / −8.92e−5 / −8.30e−5 | −2.12e−4 / −2.05e−4 / −2.12e−4 |
   | hard.uniform.entry | 3 | −1.79e−2 / −1.79e−2 / −1.10e−2 | −1.12e−2 / −1.12e−2 / −7.92e−3 | −1.01e−2 / −1.01e−2 / +2.25e−3 |
   | hard.uniform.entry | 30 | −1.49e−3 / −1.49e−3 / −1.49e−3 | −4.76e−4 / −4.76e−4 / −4.77e−4 | −1.55e−3 / −1.55e−3 / −1.55e−3 |
   | hard.relay.entry | 3 | −6.32e−4 / −6.32e−4 / +7.96e−4 | −4.34e−5 / −4.34e−5 / +2.56e−4 | +2.35e−5 / +2.35e−5 / +4.81e−4 |
   | hard.relay.entry | 30 | +3.85e−4 / +3.85e−4 / −1.30e−2 | −6.12e−4 / −6.12e−4 / +4.51e−3 | +1.41e−3 / +1.41e−3 / +8.17e−3 |
   | hard.flip.entry | 3 | −9.22e−3 / −9.22e−3 / −1.13e−2 | −2.82e−2 / −2.82e−2 / −3.62e−2 | −1.04e−2 / −1.04e−2 / −1.65e−2 |
   | hard.flip.entry | 30 | −5.65e−4 / −5.65e−4 / +3.78e−4 | −1.24e−3 / −1.24e−3 / +7.04e−3 | −9.99e−4 / −9.99e−4 / +1.81e−2 |
   | hard.relay.logit | 3 | −2.66e−4 / −2.52e−4 / −2.34e−4 | −8.08e−5 / −7.45e−5 / +4.50e−4 | −2.82e−4 / −2.71e−4 / −2.31e−4 |
   | hard.relay.logit | 30 | +4.28e−5 / +2.04e−5 / −1.25e−3 | +3.28e−4 / +2.47e−4 / +1.68e−4 | +1.07e−4 / +1.03e−4 / −8.05e−4 |
   | hard.autodiff.logit | 3 | −2.86e−4 / −2.52e−4 / −2.34e−4 | −1.80e−4 / −7.45e−5 / +4.50e−4 | −3.41e−4 / −2.71e−4 / −2.31e−4 |
   | hard.autodiff.logit | 30 | +1.88e−4 / +2.04e−5 / −1.25e−3 | +4.61e−4 / +2.47e−4 / +1.68e−4 | +2.12e−4 / +1.03e−4 / −8.05e−4 |

   On the soft pass the full gradient is the finite difference to three digits and the term
   through the signal, a Hessian, is a tenth of the total at η = 3 and half at η = 30. On the
   `entry` cells of the hard pass the full and the first-order gradient are the same number to the
   last digit, on every via: the signal has no derivative in z. On the `logit` cells of the hard
   pass, the straight-through cells, they are not: σ′ of the stored logit is differentiable, and
   the surrogate tables of `hard.autodiff.logit` add a second path, together a factor of nine at
   η = 30 with nothing to check them against. The first draft's doc said "on the bits the
   meta-gradient is exactly the first-order term"; that is true of the `entry` cells and the doc
   now says so, and `first_order` defaults to on for any hard-pass cell. The finite difference on
   the bits disagrees with both on most cells: the objective on the bits is a staircase in η, and a
   difference of ±0.5 that crosses a flip measures a jump, not a slope; the mechanics test uses the
   soft pass for that reason.
6. *The seed and the signal on the hard pass, exact.* Asked whether the residual is ∂ℓ/∂r only on
   the soft pass: it is ∂ℓ/∂r at the read on both, and the signal to the entry is ∂L/∂T at the
   pass's point on both, exact on the bits because the read is multilinear (now a test in
   `test_signals`). The one derivative the hard pass lacks is ∂T/∂z, the rounding; the hop to the
   logit is the surrogate. The glossary of `docs/meta.md` came out of this exchange.
7. *The held-out set.* Tasks are draws from a key; the seed's key is split once into a training
   half and a held-out half, and `adapt` draws from the second. Fixed here, never looked at while
   step 3's rule is designed.

**Measured and read.** Measured: online the objective has an optimum in η at 30 and a floor a
factor of 2.5 wide, and from a non-functional start the loop lands on it on every seed; the three
controls' landscapes have their minimum at zero and the loop goes there; the η found adapts a
held-out task on every seed; the deploy gap of the trained states is closed on this tile; with
every case per step the landscape is shallow and a found η is not a finding; the outer step of 0.1
runs one seed in ten away; which cells carry a second-order term. Read, not tested: that at first
order the check measures the cosine of the accumulated steps with the soft gradient at the end,
which is why a sign-consistent signal is found and a shuffled one is not; that the shuffled loop's
η is the leak of the layer's mean. What η means on the bits, with the stored logit bounded, is the
next chunk's question.

**The claim** (`test_claim.py`, re-run on every push), under `ONLINE_META` and its `control`
swaps, three recipe seeds: the swept objective has a floor (J within four times its minimum)
narrower than a factor of five and its ends ten times above it; the loop settles on that floor on
every seed, the median of η over the last 200 outer steps on it and the median objective ten times
below the start (the endpoint alone is not the claim: the tail carries transients to about 66 that
return within a hundred steps, and on another machine's float path one of them sat at step 500
while every local seed had returned; the finding is where the loop *settles*); the tail-median η
adapts a held-out task to 1.000 in 500 steps of the rule; under the sign flip η ends below its
start and the objective stays untrained; under the shuffled and output-only signals the objective
stays closer to untrained than to trained. Its cells are the "true" row of the finding's table
(the floor from the landscape table, the tail-median η and the held-out column of the runs) and
the controls' J and η columns.

**What would change it.** The deep shape, where the exact relay on the bits does not train and the
signal is the wiring-shaped feedback or the flip credit (step 1), with the stored logit bounded:
the same table for a hard-pass cell, and the deploy-gap column for the first time not trivially
closed (the next chunk). A pool of states of every age, which asks the same loop for an η that
also holds what it built. The rule as a function of the three factors (step 3), where the outer
loop has more than one number to move and the trajectory term returns on every cell.
