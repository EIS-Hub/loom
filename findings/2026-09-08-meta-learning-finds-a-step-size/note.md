# 2026-09-08 — Meta-learning finds a step size: the outer loop from a non-functional start, under the true signal and its controls

*From the step-2 chunk 4a (loom PR #TBD) and the probes of the same date, on the flat tile, before
the pool. The path is recorded: the objective chosen and the one not chosen, the parameterisation,
what the controls leak, and where the window moves the step size.*

**Question.** With the rule Δ = −η·s and only η learned, does the outer loop, differentiating the
loss of the tile the rule ends on through K steps of the rule, find a working step size from a
start where nothing moves; does it fail to under a sign-flipped signal; does nothing train under
a shuffled or an output-only signal; and does the η found adapt a fresh tile on a task the loop
never saw? Underneath: what does the η found *mean*?

**Floor and shape.** The flat tile `(4, 16, 8, 2)` at arity 4 on 2-juntas (4 → 2), the smoke shape of
steps 0 and 1; the signal `soft.relay.entry`, the exact adjoint to the entry, the one a soft fabric
could carry. Plain descent is the floor since the record chunk of the same date
(`notes/2026-09-08-the-floors-under-plain-descent.md`): the rule at a fixed η is that floor.

**Conditions.** `SOFT_META`: K = 16 steps of the rule per rollout, every case per step, a batch of
16 fresh tiles on 16 fresh tasks per outer step, 200 outer steps of Adam at 0.1 on log η from
η₀ = 0.01, the gradient's global norm clipped at 1. Rates are per vote (the seed is one vote per
case, summed over the task's two outputs), so the untrained loss on this tile is 0.25. The claims swap the recipe's `control` and
nothing else. The probes also visited windows W = 1 and W = 4 and rollouts of 32 and 64 steps.
Every run on the CPU.

**The objective, and the one not chosen.** J = the soft loss on every case of the tile after K
steps. The alternative, the mean of the online losses along the rollout (each step's window
scored before its update), weights the steps: an early update is credited through every later
loss, the last through none. `probes/2026-09-08-meta-objective-landscape.py` measured both across
η (cells: final loss / online mean, three probe seeds, **bold** the mean over seeds):

| η | W = 1, K = 64 | W = 4, K = 32 | W = all, K = 16 |
|---|---|---|---|
| 1 | 0.2504/0.2524 · 0.2513/0.2488 · 0.2441/0.2520 (**0.2486 / 0.2511**) | 0.2511/0.2533 · 0.2517/0.2481 · 0.2469/0.2513 (**0.2499 / 0.2509**) | 0.2526/0.2535 · 0.2512/0.2519 · 0.2489/0.2500 (**0.2509 / 0.2518**) |
| 3 | 0.2355/0.2520 · 0.2383/0.2486 · 0.2211/0.2495 (**0.2316 / 0.2500**) | 0.2472/0.2523 · 0.2499/0.2480 · 0.2385/0.2484 (**0.2452 / 0.2496**) | 0.2500/0.2521 · 0.2495/0.2509 · 0.2451/0.2482 (**0.2482 / 0.2504**) |
| 10 | 0.1354/0.1969 · 0.0040/0.1389 · 0.0217/0.1829 (**0.0537 / 0.1729**) | 0.1625/0.2383 · 0.0975/0.2234 · 0.1145/0.2180 (**0.1248 / 0.2265**) | 0.2434/0.2488 · 0.2415/0.2480 · 0.2216/0.2403 (**0.2355 / 0.2457**) |
| 30 | 0.0004/0.1041 · 0.0002/0.0693 · 0.0003/0.0951 (**0.0003 / 0.0895**) | 0.0018/0.1413 · 0.0011/0.0977 · 0.0055/0.1130 (**0.0028 / 0.1173**) | 0.0651/0.2110 · 0.0129/0.1800 · 0.0267/0.1676 (**0.0349 / 0.1862**) |
| 100 | 0.3661/0.3841 · 0.0001/0.1430 · 0.0000/0.1321 (**0.1221 / 0.2197**) | 0.0002/0.1477 · 0.0001/0.0711 · 0.0002/0.0955 (**0.0002 / 0.1048**) | 0.0004/0.0882 · 0.0004/0.0743 · 0.0009/0.0689 (**0.0006 / 0.0772**) |
| 300 | 0.2187/0.2466 · 0.1021/0.1268 · 0.3087/0.3262 (**0.2098 / 0.2332**) | 0.3855/0.2956 · 0.0000/0.1037 · 0.0000/0.0617 (**0.1285 / 0.1537**) | 0.0001/0.1300 · 0.0000/0.0695 · 0.0001/0.0434 (**0.0001 / 0.0810**) |
| 1000 | 0.4447/0.3790 · 0.1382/0.2211 · 0.2138/0.1981 (**0.2656 / 0.2661**) | 0.3389/0.3099 · 0.0000/0.0829 · 0.3745/0.3018 (**0.2378 / 0.2315**) | 0.0000/0.2028 · 0.0000/0.0692 · 0.0000/0.0715 (**0.0000 / 0.1145**) |
| 3000 | 0.5022/0.4303 · 0.3318/0.4695 · 0.2189/0.2315 (**0.3510 / 0.3771**) | 0.5158/0.4097 · 0.0000/0.0739 · 0.2550/0.3221 (**0.2569 / 0.2686**) | 0.5617/0.3396 · 0.0000/0.1090 · 0.1875/0.2355 (**0.2497 / 0.2280**) |

Two things are in the table. The step size the objective wants depends on the window: fully
online (W = 1) the final loss has its optimum at η ≈ 30 and degrades from 100 on, at W = 4 it sits
at 30 to 100, and with every case per step the loss is at its floor from 100 to 1000 and collapses
at 3000 (one seed at 0.56): a band, three times wider than the online optimum and shifted up. And
the two objectives disagree where they should: with every case per step the online mean turns up
again past η ≈ 100 (0.077 → 0.081 → 0.115) while the final loss stays at zero, the mean punishing
the early overshoot of a large step that the final tile has long recovered from. The final loss
asks where the rule *arrives*; the mean asks how it *travels*.

**The derivative path, measured** (`probes/2026-09-08-meta-gradient-paths.py`, K = 8; cells: the
full meta-gradient / the first-order one, the signal as data / a central finite difference with
h = 0.5; three probe seeds):

| signal | η | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| soft.relay.entry | 3 | −7.08e−4 / −7.85e−4 / −7.09e−4 | −4.74e−4 / −5.15e−4 / −4.74e−4 | −9.39e−4 / −9.44e−4 / −9.39e−4 |
| soft.relay.entry | 30 | −1.50e−3 / −7.79e−4 / −1.50e−3 | −3.11e−3 / −1.40e−3 / −3.12e−3 | −5.17e−3 / −2.74e−3 / −5.17e−3 |
| soft.relay.entry | 300 | +1.61e−3 / −6.60e−5 / +1.61e−3 | −8.50e−7 / −2.47e−6 / −8.50e−7 | −3.10e−6 / −5.80e−6 / −3.10e−6 |
| hard.uniform.entry | 3 | −1.79e−2 / −1.79e−2 / −1.10e−2 | −1.12e−2 / −1.12e−2 / −7.92e−3 | −1.01e−2 / −1.01e−2 / +2.25e−3 |
| hard.uniform.entry | 30 | −1.49e−3 / −1.49e−3 / −1.49e−3 | −4.76e−4 / −4.76e−4 / −4.77e−4 | −1.55e−3 / −1.55e−3 / −1.55e−3 |
| hard.uniform.entry | 300 | −5.8e−11 / −5.8e−11 / 0 | 0 / 0 / 0 | −1.8e−10 / −1.8e−10 / 0 |

On the soft pass the full meta-gradient is the finite difference to three digits, and the
first-order path is not the whole of it: the term through the signal is a tenth at η = 3, half at
η = 30, and past the optimum, at η = 300, it can change the gradient's sign (seed 0: +1.6e−3 against
−6.6e−5 first order), because the signal changes more along a longer trajectory. On the bits the
full and the first-order meta-gradient are the same number to the last digit, as the declaration
in `docs/meta.md` says they must be: nothing differentiates through rounded tables. There the finite
difference disagrees on four cells of nine: the objective on the bits is piecewise smooth in η,
and a difference of ±0.5 in η that crosses a flip inside the rollout measures a jump, not a slope.
The mechanics test uses the soft relay for that reason.

**The controls** (`probes/2026-09-08-controls-trajectories.py`; batch 16, K = 16, every case per
step, 200 outer steps from η₀ = 0.01; cells: η / final loss at outer steps 50, 100, 200; then the
η the run ends on, driven by the *true* signal for 500 plain steps on a fresh held-out 2-junta,
hard accuracy; three probe seeds):

| control | seed 0 | seed 1 | seed 2 | held-out at the η found |
|---|---|---|---|---|
| none (the true signal) | 5.08/0.2451 · 250/0.0001 · 127/0.0004 | 5.04/0.2465 · 170/0.0002 · 130/0.0004 | 5.13/0.2457 · 246/0.0001 · 119/0.0005 | 1.000 · 1.000 · 1.000 |
| flipped | 5.6e−4/0.2536 · 2.6e−4/0.2526 · 1.1e−4/0.2513 | 5.3e−4/0.2537 · 2.4e−4/0.2521 · 1.0e−4/0.2528 | 5.4e−4/0.2535 · 2.4e−4/0.2525 · 1.0e−4/0.2519 | 0.469 · 0.438 · 0.344 |
| shuffled | 3.98/0.2527 · 26.9/0.2507 · 32.8/0.2499 | 3.64/0.2525 · 102/0.2503 · 54.5/0.2504 | 4.00/0.2525 · 63.1/0.2514 · 17.6/0.2512 | 1.000 · 1.000 · 1.000 |
| output-only | 4.88/0.2497 · 166/0.2331 · 143/0.2327 | 4.91/0.2498 · 101/0.2324 · 129/0.2305 | 4.92/0.2495 · 144/0.2337 · 124/0.2321 | 1.000 · 1.000 · 1.000 |

Under the true signal η rises by a constant factor per outer step (5 at step 50 means nine
doublings from 0.01 in fifty steps), overshoots to about 250, into the region where the final loss
is already at its floor, and settles near 125, where some batch members begin to overshoot and
pull it down: the loss is at its floor from step 100 on, and the η found adapts a fresh tile on a
task the loop never saw to 1.000 in 500 plain steps, on every seed. Under the flipped signal η can
only shrink, and does, geometrically, to a hundredth of its start; nothing trains and the tile
stays at chance. Under the shuffled signal the loop finds an η of the same order as the true
signal's (18 to 102) and the loss does not move: the numbers carry no information about which
entry they reach. Under output-only the output gates learn what they can on their own residual, a
drop of a fourteenth in the loss (0.250 → 0.233) and no more. The last column for the two
information controls says only that the η they wandered to is a working step size *for the true
signal*, which every η between ten and a few hundred is on this tile; it is not a result about the
controls.

**Path.**

1. *The objective.* The first design carried the mean of the online losses, on the reading that a
   deployed tile experiences the stream. Gabriel asked whether back-propagation through the K
   steps did not make the final loss enough, and whether the mean did not weight the steps. It
   does (above): the final loss became the objective, the online losses a record.
2. *The parameterisation.* η = exp(raw) rather than softplus: at a non-functional start the two
   agree, but a constant factor per Adam step reaches a working scale in log(scale)/lr steps, and
   the flipped control then drives raw to −∞ geometrically, which the table shows (η falls by a
   factor 2.2 every fifty steps). Adam on the host is kept for the same reason: the gradient on
   raw is η times the gradient on η and vanishes exactly where the check starts.
3. *What the shuffled control leaks.* A permutation within a layer keeps the layer's mean, and the
   mean is a real component of the gradient, so the loop's η under it is not zero and not
   meaningful; the claim is therefore worded on the loss, never on η. The sign flip is the only
   control whose η is a result.
4. *The held-out set.* Tasks are draws from a key; the seed's key is split once into a training
   half and a held-out half, and `adapt` draws from the second. Fixed here, never looked at while
   step 3's rule is designed.
5. *The second-order term.* Read as negligible on the soft pass before it was measured; the
   derivative-path table says it grows with η and dominates it past the optimum. It is
   kept in the objective on the soft pass (the rollout differentiates through the signal unless
   told not to); the bits never have it.

**Measured and read.** Measured: the meta-gradient is the finite difference on the soft pass and
identical to its first-order term on the bits; from a non-functional start the loop finds a step
size under the true signal on every seed, fails to under the sign flip, and leaves the loss
untrained under the shuffled and output-only signals; the η found adapts a held-out task; the step
size the objective wants depends on the window. Read, not tested: that at first order the check
measures the cosine of the accumulated steps with the soft gradient at the end, which is why a
sign-consistent signal is found and a shuffled one is not, and that on this benign tile the
final-tile objective has no interior optimum in η because the exact soft signal cannot overshoot
what a sixteen-entry table can absorb. What η means on the bits, where the design pass saw the
final loss go flat above a threshold and the loop walk η into the thousands, is the next chunk's
question, with the stored logit bounded.

**Claims left behind.** `claims/test_2026_09_08_meta_learning_finds_a_step_size.py`, under
`SOFT_META` and its `control` swaps, three recipe seeds: the loop finds a step size that adapts a
held-out task to 1.000 in 500 plain steps; under the sign flip η ends below its start and the loss
stays untrained; under the shuffled and output-only signals the loss stays closer to untrained
than to trained.

**What would change it.** A shape where the exact soft signal can overshoot, so that the
final-tile objective has an interior optimum at every window (the deep shape, next chunk, records
this on the bits). A pool of states of every age, which asks the same loop for an η that also
holds what it built. The rule as a function of the three factors (step 3), where the outer loop
has more than one number to move and Adam's normalisation is no longer a footnote.
