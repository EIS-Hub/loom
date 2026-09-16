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
steps of the rule per rollout, a batch of 16 fresh tiles on 16 fresh tasks per outer step, 1000
outer steps of Adam at 0.01 on log η from η₀ = 0.01, the gradient's global norm clipped at 1.
Rates are per vote, so the untrained objective is 0.25 (the soft read of a random table is 0.5 on
every output) and the untrained deployed loss is about 0.5 (its bits are wrong half the time). The
controls swap the recipe's `control` and nothing else. `BATCHED_META` (every case per step, K =
16, 200 outer steps at 0.1), the first draft's regime, appears in the path only. Every run on the
CPU; the sweep on three recipe seeds, the learned column on six.

## The finding

One table, every row on the same axes: online, K = 64, the same members for the sweep and the
loop's first step, the held-out check 500 steps of the rule at the η found on a fresh tile and a
fresh task from the seed's held-out half. The sweep column is the mean over three seeds; the
learned columns are the range over six seeds of the tail, the median over the last 200 of the
1000 outer steps, where the loop settles (the deployed loss is that of the final states).

| signal | swept objective: shape, best η, J there | learned η | J at the learned η | deployed loss there | held-out A^hard, driven by this signal | the same η, driven by the true signal |
|---|---|---|---|---|---|---|
| true | an interior optimum: **30**, 0.0025 (20 → 0.0041, 50 → 0.0033, 100 → 0.092, 200 → untrained) | **25.4 to 38.2** | 0.0020 to 0.0022 | 0.0000 to 0.0137 | **1.000 ×6** | 1.000 ×6 |
| flipped | monotone worse: 0.43 at η = 3, 0.50 from 100 on; best **0** | 3.6e-04 to 3.8e-04, ×26 below the start | 0.253 to 0.253 | 0.48 to 0.51 | 0.34 to 0.53 | 0.34 to 0.53 |
| shuffled | monotone worse: 0.254 at 3, 0.44 at 400; best **0** | 0.60 to 0.63 | 0.2522 to 0.2524 | 0.46 to 0.51 | 0.34 to 0.59 | 0.69 to 0.97 |
| output-only | flat to worse: 0.251 at 3, 0.278 at 1000; best **0** | 1.15 to 1.26 | 0.2505 to 0.2507 | 0.46 to 0.51 | 0.34 to 0.66 | 1.000 ×6 |

The outer loop finds the minimum of whatever landscape the signal hands it, and only the true
signal hands it a landscape with an interior minimum. Under the true signal the sweep's optimum is
30, the floor (J within four times its minimum) is η from 20 to 50, and the loop settles at 23 to
33 on every seed (the tail medians; the endpoints 29 to 33), with the objective and the deployed
loss both at that floor and the held-out tile at 1.000. Under the three controls the sweep says the best η is zero, and the loop goes there: it
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
cells: η / J at outer steps 200, 400, 600, 800, 1000; the tail, the medians of η and J over the
last 200 outer steps, where the loop settles; the deployed loss of the final states at 1000; the
held-out accuracy at the tail-median η, under the control and under the true signal):

| control | seed | η / J along the run | tail: median η / J over steps 800 to 1000 | deployed at 1000 | held-out |
|---|---|---|---|---|---|
| none | 0 | 0.186/0.2498 · 16.9/0.0036 · 23.5/0.0072 · 25.5/0.0055 · 29.2/0.0010 | 26.2 / 0.0022 | 0.0020 | 1.000 / 1.000 |
| none | 1 | 0.188/0.2513 · 17.2/0.0018 · 24.7/0.0011 · 26.3/0.0023 · 28.5/0.0008 | 26.8 / 0.0020 | 0.0000 | 1.000 / 1.000 |
| none | 2 | 0.184/0.2506 · 16.6/0.0054 · 26/0.0015 · 30.6/0.0012 · 43.4/0.0022 | 38.2 / 0.0020 | 0.0039 | 1.000 / 1.000 |
| none | 3 | 0.194/0.2519 · 16.6/0.0052 · 24.4/0.0056 · 23.8/0.0028 · 28/0.0021 | 25.4 / 0.0021 | 0.0000 | 1.000 / 1.000 |
| none | 4 | 0.186/0.2514 · 17.1/0.0089 · 23.3/0.0076 · 29.6/0.0007 · 29.7/0.0019 | 29.6 / 0.0020 | 0.0000 | 1.000 / 1.000 |
| none | 5 | 0.19/0.2508 · 16.8/0.0074 · 23.9/0.0027 · 25.1/0.0078 · 26.6/0.0088 | 25.9 / 0.0021 | 0.0137 | 1.000 / 1.000 |
| flipped | 0 | 0.00244/0.2513 · 0.00115/0.2536 · 0.000678/0.2519 · 0.000447/0.2531 · 0.000318/0.2524 | 0.000375 / 0.2530 | 0.4922 | 0.469 / 0.469 |
| flipped | 1 | 0.00242/0.2529 · 0.00112/0.2530 · 0.000656/0.2515 · 0.000433/0.2529 · 0.000309/0.2542 | 0.000363 / 0.2529 | 0.5039 | 0.438 / 0.438 |
| flipped | 2 | 0.00245/0.2519 · 0.00115/0.2532 · 0.000679/0.2538 · 0.000452/0.2516 · 0.000325/0.2543 | 0.000379 / 0.2528 | 0.5098 | 0.344 / 0.344 |
| flipped | 3 | 0.00238/0.2530 · 0.00112/0.2543 · 0.000662/0.2535 · 0.00044/0.2524 · 0.000315/0.2541 | 0.00037 / 0.2529 | 0.4805 | 0.406 / 0.406 |
| flipped | 4 | 0.00244/0.2526 · 0.00113/0.2539 · 0.00067/0.2528 · 0.000449/0.2524 · 0.000322/0.2521 | 0.000377 / 0.2528 | 0.4980 | 0.531 / 0.531 |
| flipped | 5 | 0.0024/0.2522 · 0.00111/0.2534 · 0.000657/0.2535 · 0.000438/0.2529 · 0.000314/0.2535 | 0.000367 / 0.2529 | 0.4961 | 0.438 / 0.438 |
| shuffled | 0 | 0.143/0.2509 · 0.665/0.2526 · 0.599/0.2512 · 0.605/0.2525 · 0.626/0.2525 | 0.626 / 0.2524 | 0.4863 | 0.406 / 0.938 |
| shuffled | 1 | 0.142/0.2525 · 0.588/0.2521 · 0.636/0.2511 · 0.652/0.2523 · 0.554/0.2530 | 0.628 / 0.2523 | 0.4844 | 0.531 / 0.969 |
| shuffled | 2 | 0.124/0.2517 · 0.608/0.2527 · 0.636/0.2534 · 0.593/0.2515 · 0.561/0.2542 | 0.597 / 0.2524 | 0.5137 | 0.344 / 0.688 |
| shuffled | 3 | 0.157/0.2528 · 0.615/0.2535 · 0.608/0.2533 · 0.617/0.2521 · 0.61/0.2540 | 0.629 / 0.2523 | 0.4648 | 0.500 / 0.906 |
| shuffled | 4 | 0.133/0.2523 · 0.618/0.2532 · 0.612/0.2524 · 0.599/0.2518 · 0.638/0.2519 | 0.62 / 0.2522 | 0.4961 | 0.594 / 0.844 |
| shuffled | 5 | 0.141/0.2520 · 0.625/0.2520 · 0.679/0.2527 · 0.601/0.2522 · 0.617/0.2523 | 0.623 / 0.2523 | 0.5000 | 0.469 / 0.750 |
| output-only | 0 | 0.169/0.2505 · 1.27/0.2508 · 1.23/0.2500 · 1.21/0.2510 · 1.24/0.2513 | 1.25 / 0.2505 | 0.4688 | 0.594 / 1.000 |
| output-only | 1 | 0.175/0.2519 · 1.17/0.2494 · 1.19/0.2494 · 1.11/0.2505 · 1.17/0.2506 | 1.21 / 0.2506 | 0.4941 | 0.531 / 1.000 |
| output-only | 2 | 0.165/0.2513 · 1.22/0.2507 · 1.3/0.2509 · 1.13/0.2501 · 1.16/0.2528 | 1.17 / 0.2506 | 0.5098 | 0.344 / 1.000 |
| output-only | 3 | 0.179/0.2524 · 1.11/0.2510 · 1.2/0.2523 · 1.15/0.2503 · 1.13/0.2506 | 1.15 / 0.2507 | 0.4648 | 0.438 / 1.000 |
| output-only | 4 | 0.174/0.2519 · 1.23/0.2511 · 1.18/0.2508 · 1.17/0.2514 · 1.15/0.2499 | 1.19 / 0.2506 | 0.4590 | 0.594 / 1.000 |
| output-only | 5 | 0.176/0.2515 · 1.23/0.2496 · 1.33/0.2507 · 1.25/0.2504 · 1.25/0.2505 | 1.26 / 0.2505 | 0.4902 | 0.656 / 1.000 |

Under the true signal η climbs from 0.01 by a constant factor per outer step (0.72 at step 100 is
six doublings), reaches the optimum near step 200 and stays, one seed making a transient to 66 at
step 300 and returning (on the CI runner's float path such a transient sat at step 500 on one
seed, which is why the claim pins the tail and not the endpoint); the deployed loss at the end is the objective's size, so the deploy gap on
this tile is closed by the rule that trained it. Under the flip η can only shrink, and does. Under
the shuffled signal η rises from 0.01 to 0.6 and stops: the layer's mean, which a permutation
keeps, is a real component of the gradient and carries the loop to the edge of the landscape's
flat part, and no further. Under output-only the same, at 1.

## Path

1. *The regime.* The first draft ran the loop with every case per step (`BATCHED_META`). That
   landscape is shallow: `2026-09-16-landscape.py` under `BATCHED_META` gives, for the true
   signal, J = 0.2468 · 0.2238 · 0.1349 · 0.0578 · 0.0079 · 0.0006 · 0.0032 · 0.0136 · 0.0225 at
   η = 3 to 1000 (means over the seeds), a minimum at 100 with the loss ten times below untrained
   from 50 to 1000; and the loop landed at η = 127, 130 and 119 at step 200 (J = 0.0004 to 0.0005, the deployed loss 0.0000) (`2026-09-16-learned.py`, three
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
   (item 5). The second draft climbed at 0.03 for 500 steps, and every local seed settled; the CI
   runner, on another float path, had one of the three claim seeds sit at 66 and then, with the
   claim on the tail median, at 83: the loop does not settle reliably at that step. The reason is
   Adam: it rescales whatever gradient it sees into a step of the outer step's size, so at the
   floor, where the gradient is tiny and its sign only mostly right, the loop random-walks with a
   restoring drift, and the spread at rest shrinks with the square root of the outer step while the
   climb lengthens with its inverse. The schedule was then chosen on ten seeds
   (`2026-09-16-outer-schedule.py`; one process per setting and seed; cells: the tail medians of
   η and J over the last 200 outer steps, the tail's range; the floor is η from 20 to 50):

| setting | seeds 0 to 9: tail-median η / J [tail min, tail max] (· the median on the floor but not the whole tail; ✗ off) | whole tail on the floor |
|---|---|---|
| A: 0.03, 500 steps, batch 16 (round 2's first schedule) | 28.3/0.0019 [24.9, 33.2] · 24.4/0.0019 [22.5, 35.1] · 24.7/0.0022 [21.0, 30.6] · 26.5/0.0019 [21.2, 31.2] · 23.0/0.0028 [19.6, 29.1] · · 33.0/0.0035 [29.6, 70.3] · · 28.1/0.0020 [25.4, 35.3] · 29.9/0.0017 [28.2, 44.2] · 26.5/0.0021 [25.0, 29.1] · 51.6/0.0101 [38.8, 92.4] ✗ | **7/10**, off 1/10 |
| B: 0.015, 800, 16 | 25.4/0.0021 [22.6, 28.3] · 28.2/0.0022 [25.2, 36.0] · 27.8/0.0016 [26.1, 30.0] · 26.8/0.0016 [26.0, 33.9] · 28.0/0.0017 [26.2, 32.7] · 28.3/0.0019 [25.8, 33.9] · 73.3/0.0389 [35.5, 88.2] ✗ · 25.8/0.0021 [24.9, 26.6] · 25.7/0.0020 [23.7, 29.2] · 29.5/0.0017 [28.2, 32.9] | **9/10**, off 1/10 |
| C: 0.03, 500, 32 | 25.7/0.0027 [23.2, 30.4] · 24.5/0.0025 [22.2, 33.6] · 30.0/0.0023 [23.9, 41.2] · 24.5/0.0029 [19.2, 29.6] · · 31.7/0.0024 [26.1, 50.3] · · 24.3/0.0027 [22.7, 26.6] · 24.8/0.0026 [20.8, 32.3] · 25.7/0.0026 [22.3, 28.2] · 27.0/0.0026 [23.7, 34.9] · 26.6/0.0022 [23.7, 37.7] | **8/10**, off 0/10 |
| **D: 0.01, 1000, 16 (the recipe)** | 26.2/0.0022 [25.2, 29.4] · 26.8/0.0020 [26.2, 28.5] · 38.2/0.0020 [30.7, 44.9] · 25.4/0.0021 [23.9, 28.5] · 29.6/0.0020 [25.8, 30.9] · 25.9/0.0021 [25.1, 26.8] · 25.2/0.0022 [23.8, 27.5] · 29.0/0.0018 [25.3, 31.0] · 26.3/0.0017 [24.8, 30.7] · 29.9/0.0017 [26.6, 35.3] | **10/10**, off 0/10 |
| E: 0.015, 800, 32 | 32.9/0.0020 [29.6, 45.1] · 42.1/0.0048 [38.0, 59.4] · · 127.5/0.1338 [102.1, 140.4] ✗ · 27.5/0.0022 [26.4, 29.7] · 67.6/0.0267 [36.0, 89.5] ✗ · 31.6/0.0029 [27.0, 59.9] · · 33.2/0.0026 [25.1, 39.0] · 27.2/0.0025 [24.0, 30.3] · 34.7/0.0033 [26.5, 54.7] · · 49.3/0.0095 [30.6, 77.3] · | **4/10**, off 2/10 |

   At 0.03 one seed in ten ends off the floor, the rate CI met. Halving the step alone still loses
   a seed; doubling the batch alone ends none off the floor but the tails still wander to the
   floor's edges; both together is worse, with one seed carried to 127 (read, not tested: a larger
   batch makes the gradient's sign more consistent, and past the optimum the first-order term's
   sign is wrong, so a decisive loop follows it up where a noisy one would have wandered back). At
   0.01 for 1000 steps every seed keeps its whole tail on the floor, the medians between 25 and 38:
   the recipe.
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
every case per step the landscape is shallow and a found η is not a finding; at an outer step of
0.1 one seed in ten runs away and at 0.03 one in ten ends off the floor, at 0.01 none of ten;
which cells carry a second-order term. Read, not tested: that at first
order the check measures the cosine of the accumulated steps with the soft gradient at the end,
which is why a sign-consistent signal is found and a shuffled one is not; that the shuffled loop's
η is the leak of the layer's mean. What η means on the bits, with the stored logit bounded, is the
next chunk's question.

**The claim** (`test_claim.py`, re-run on every push, one test per seed and control so the
runner spreads them), under `ONLINE_META` and its `control` swaps, three recipe seeds: the swept
objective has a floor (J within four times its minimum) narrower than a factor of five and its ends
ten times above it; the loop settles on that floor on every seed, the median of η over the last 200
outer steps on it and the median objective ten times below the start (the endpoint alone is not the
claim: the loop's tail wanders, and where it settles is the finding); the tail-median η adapts a
held-out task to 1.000 in 500 steps of the rule; under the sign flip η ends below its start and the
objective stays untrained; under the shuffled and output-only signals the objective stays closer to
untrained than to trained. Its cells are the "true" row of the finding's table (the floor from the
landscape table, the tail-median η and the held-out column of the runs) and the controls' J and η
columns. The claim's first two schedules failed on the CI runner (an endpoint at 66, then a tail
median at 83, on one seed, on another machine's float path with a newer JAX); the schedule above
and the pin of JAX's version are what made it hold there.

**What would change it.** The deep shape, where the exact relay on the bits does not train and the
signal is the wiring-shaped feedback or the flip credit (step 1), with the stored logit bounded:
the same table for a hard-pass cell, and the deploy-gap column for the first time not trivially
closed (the next chunk). A pool of states of every age, which asks the same loop for an η that
also holds what it built. The rule as a function of the three factors (step 3), where the outer
loop has more than one number to move and the trajectory term returns on every cell.
