# 2026-09-08 — The floors under plain descent: the step is the signal's size, and at depth the plain reference is no longer the floor

*From chunk 4 of step 1 (the record chunk that makes descent plain), the five probes of the same
date, and the re-run of every claim of steps 0 and 1 under the re-pinned recipes. This note records
the path: the grids that had no edge, the plateau that was a point, the reading that fell.*

**Question.** Every statement of the four earlier notes about which signal descent can follow was
made under Adam. Which of them hold when descent is the plain update, Δ = −lr·s, nothing
normalised, and at what rate does each signal train? X against its floor: each signal's hard
accuracy at fixed budgets over a log grid of rates, against the reference's on the same tiles.

**Why plain descent.** Decided 2026-09-08 (Gabriel): `descent.descend` is now `logits − lr·s`
per layer, with `s = compute(signal, …)` unchanged; `optax` left the module. Three reasons. Adam
normalises per coordinate, so a signal's magnitude, and the σ′ factor that separates the partial
to the logit from the partial to the entry, never entered any statement about which signal
descent can follow. It carries two moments per logit, a memory the cost table of
`docs/signals.md` explicitly excludes from a signal's price. And at window 1 on the bits it
amplifies rarely addressed entries, whose second moment decays toward zero, and its momentum
carries one case's vote into the cases that follow. Step 2's rule is exactly −η·s with η learned,
so the plain update is the only honest baseline from here on, and Adam never enters a signal
claim again. This note supersedes, by pointer, the numbers of
`notes/2026-09-03-one-tile-computes.md`, `notes/2026-09-07-straight-through-is-a-signal-cell.md`,
`notes/2026-09-07-the-relay-is-autodiff-and-the-bits-learn-by-alignment.md` and
`notes/2026-09-07-direct-feedback-and-the-flip-credit.md`, which are never edited; their
qualitative statements are each marked held or moved below.

**Conditions.** Two shapes. The flat tile, hidden `(16, 8)` at arity 4, built to the task's width
as `recipes.setup` builds it (`(4, 16, 8, 2)` on 2-juntas 4→2, `(4, 16, 8, 3)` on 2-bit addition
with carry; the earlier notes wrote `(4, 16, 8, 2)` for both, which the addition claim never
was). The deep tile `(6, 32, 32, 16, 4)` at arity 3, step 1's landmark shape, on 6-bit addition
and 2-juntas 6→4. Signals: on the flat tile the reference (`soft.autodiff.logit`),
straight-through (`hard.autodiff.logit`) and the relay to the entry on both passes, batched; the
reference and the soft relay online (window 1). On the deep tile, on the soft pass the reference
and the relay, uniform split, random bus and reachable bus to the entry; on the bits the same
five and the flip credit. Rates on a ×3 grid, 1 to 100000 (the brief's grid stopped at 3000; on
the flat tile the reference had no upper edge there, so every sweep went on to 100000), then ×1.5
refinements where a plateau was narrow or edge-bound, and 12000 steps where a low edge could be
a budget edge. Budgets: 2000 steps on the flat tile (3000 online), 4000 on the deep one, accuracy
recorded every 50 (flat) or 250 (deep) steps, so "every seed at 1.000 by" is known to that grain.
Probe seeds 0–2: the tile from `key(seed)` as the earlier deep probes drew it, so the addition
tiles of the deep tables are the same tiles as the earlier notes'; the task from a fold of the
same key. The claims run under the recipe seeds, which `recipes.setup` splits differently, and
one probe checks the deep pins on those tiles too. Device: CPU, `jax.vmap` over seeds and over
rates, so a cell is one process. The probes: `probes/2026-09-08-plain-descent-flat.py` (29 s),
`probes/2026-09-08-plain-descent-deep.py` (254 s),
`probes/2026-09-08-plain-descent-rate-each-signal-wants.py` (113 s),
`probes/2026-09-08-plain-descent-deploy-gap-at-depth.py` (15 s),
`probes/2026-09-08-plain-descent-recipe-seeds-at-depth.py` (20 s); seven minutes of compute in
all, far under the two hours budgeted, which is why no grid was shrunk.

*How a pin is chosen.* The working plateau of a signal that reaches the target is the run of
rates at which every seed sits at 1.000 by the budget; of one that does not, the rates whose mean
at the budget is within 0.02 of its best. A recipe's rate is the geometric centre of the plateau,
never an edge; its budget twice the first step at which every probe seed reached the target at
that rate, rounded to the record's grain; then the claims re-run under the recipe seeds.

**Measured.** The sweeps in full, every rate and every seed on both tiles with the plateaus
refined, are the appendix at the end of this note; the argument reads from their summary, the
table of the rate each signal wants below.

**The path.** Three things were not what the brief expected. (1) The flat reference had no upper
edge on the brief's grid: at 3000 it reaches the target in 50 steps, and the edge is at 100000
on the junta (one seed at 0.750) and at 10000 on addition, so the grid was extended a further two
decades everywhere. (2) At depth the plain reference's plateau, defined as every probe seed at
1.000 by 4000 steps, is one grid point at ×1.5 resolution, 3000: its neighbours 2000 and 4500
leave one seed at 0.980 or 0.988, and at 100 to 1000 two seeds sit at 0.89 to 0.98 for 12000
steps. The first reading, a deploy gap that the plain update never closes because its push
vanishes with the residual, was probed and fell for the reference: the soft read is wrong too
(0.90 to 0.98), the loss sits at 0.007 to 0.027 and the gradient at 0.2 to 2.6 % of its initial
size, a plateau or shallow minimum of the soft loss that the larger step does not enter. Why the
larger step avoids it is not measured (a reading: it saturates the tables before the soft
landscape flattens, 0.41 of the entries unsaturated at 3000 against 0.53 to 0.67 below). The
reading held for the relay to the entry at 700 on one seed: soft 1.000, bits 0.992, loss 0.0015,
gradient zero, a deploy gap that nothing closes. (3) Under the recipe seeds the same rate misses:
at 3000 two tiles sit at 0.988 and 0.992 for 4000 steps while 2000 and 4500 reach by 1000 and
1250. Over the six tiles tried no single rate takes the plain reference to the target on all of
them; the relay to the entry at 1500 does, on all six, with its own turbulence on the way (one
recipe seed at 0.652 at 1000 steps, 1.000 by 1750). So at depth the floor of this repo is no longer
what it was: the reference comes within two bits of the target on every tile and reaches it on
most; the signal that reaches it on every tile is the relay without σ′, at a rate of its own.

**The rate each signal wants.** The plateau per (shape, pass, signal), its centre, and the pin.

| shape, task | signal | plateau (rates) | centre | pinned | recipe |
|---|---|---|---|---|---|
| flat, both tasks | soft.autodiff.logit, batched | [10, 3000] (junta [3, 30000], addition [10, 3000]) | 173 | 100 | `SOFT_FLOOR`, and `descend`'s default |
| flat, both tasks | hard.autodiff.logit (straight-through) | [300, 700] (junta [1, 1000], addition [300, 700]) | 458 | 450 | `HARD_FLOOR` |
| flat, both tasks | soft.relay.entry, batched | [3, 1000] (junta [1, 100000]: above 3000 the bits regime, which the junta allows) | 55 | — | none swaps to it on the flat tile |
| flat | hard.relay.entry | junta [1, 100000]; addition none (chance at every rate) | — | — | none |
| flat, both tasks | soft.autodiff.logit, window 1 | [10, 1000] (refined [30, 300] on the junta) | 100 | 100 | `ONLINE_FLOOR` |
| flat | soft.relay.entry, window 1 | addition [1, 30], junta [1, 100], erratic above | 5–10 | — | none |
| deep addition | soft.autodiff.logit | the point 3000 on the probe seeds; 2000 and 4500 on the recipe seeds | 3000 | 3000 | `DEEP_FLOOR` |
| deep addition | soft.relay.entry | the point 1500 (700 to 1000 leave one tile at 0.992 to 0.996 for 12000 steps; 2000 collapses) | 1500 | 1500 | `DEEP_RELAY` |
| deep addition | soft.uniform.entry | short of the target at every rate; best 0.954 at 300, 0.957 at 10000 | — | — | swapped in under `DEEP_RELAY` |
| deep addition | soft.direct.entry | short; best 0.79 to 0.80 at 1 to 30, falling with the rate | — | — | — |
| deep addition | soft.reachable.entry | short; 0.89 to 0.94 at every rate | — | — | — |
| deep addition | hard.autodiff.logit (straight-through) | short; best 0.939 ± 0.011 at 1000, 0.90 to 0.94 on [700, 2000], chance at ≤ 300 and ≥ 30000 | 1000 | 1000 | `DEEP_HARD_FLOOR` |
| deep addition | hard.relay.entry | chance at every rate | — | — | swapped in under `DEEP_HARD_FLOOR` |
| deep addition | hard.uniform / direct / reachable / flip .entry | rate-free above 1000 (every digit identical from 1000 to 100000), 0.86 to 0.96 / 0.62 to 0.72 / 0.88 to 0.99 / 0.85 to 0.91 | — | 1000 | swapped in under `DEEP_HARD_FLOOR` |
| deep junta | soft, every via | reference [10, 10000]; relay [3, 100000]; uniform and reachable at every rate; direct 2 of 3 seeds at most rates | — | — | — |
| deep junta | hard, every via | straight-through [150, 1500] with a hole at 200 (one seed at 0.488); uniform and reachable at every rate; direct [3, 100]; flip 0.97 to 1.000, all seeds at 300; the exact relay 2 of 3 seeds at best, seed 2 never | — | — | — |

Do the best rates differ across signals by more than a decade? On the soft pass at depth, no:
the reference at 3000 and the relay to the entry at 1500, a factor of two, but their plateaus
are points that do not overlap, so the swap a claim makes has to change the recipe, not only the
signal (`DEEP_RELAY`). On the flat tile the relay to the entry sits a decade below the reference
(centre 55 against 173) and both plateaus are wide enough to share a rate. On the bits, the
entry cells are rate-free above a threshold and any rate above it serves them all, so one recipe
(`DEEP_HARD_FLOOR`, whose own signal, straight-through, wants exactly the band around 1000)
serves every bits claim. Why the bits cells to the entry are rate-free: a bits signal reads the
bits, not the logits' size, so with z' = z / lr the update is z' ← z' − s from an initial
z₀ / lr that vanishes as lr grows; every trajectory above the threshold is the one started from
zero at unit rate. Straight-through keeps σ′(z) and is not: its band is narrow.

**What held, what moved.** Every qualitative statement of the four earlier notes and of
`docs/signals.md` §"What the transports do at depth", with the numbers behind it.

*Of the step-0 note.* One tile reaches 2-juntas by direct descent, soft and hard agreeing once
trained: **held** (`SOFT_FLOOR` at 100, every probe seed at 1.000 by 100 steps, the claim with
soft accuracy 1.000 too). Two-bit addition with carry: **held** (by 200 steps at 100). Online,
one case per step: **held** (`ONLINE_FLOOR` at 100, by 100 steps on the junta and 250 on
addition, against Adam's 3000-step budget).

*Of the straight-through note.* The two gradients nearly orthogonal at initialisation, the bits'
sparser: **unchanged**, a statement about the signals, not the optimiser, not re-measured. On the
bits at the soft rate two seeds miss at 500 steps, one at 2000, and a fifth of the rate reaches:
**moved**; the plain junta reaches at every rate from 1 to 1000 and fails above (0.66 to 0.88 at
3000 and beyond), and 2-bit addition reaches only in [300, 700], short below (means 0.48 to 0.69 from 1 to 200,
single seeds at 1.000 at 1 and 200) and above (0.993 ± 0.010 at 1000). "Descent on the hard pass wants a smaller
step" is therefore a narrower band, not a smaller step: its top (1000) lies below the
reference's (3000 to 30000), its bottom on addition (300) above the reference's (10). The
chattering read as Adam's normalised steps: **moved**; the plain update chatters too, the first
step at 1.000 on the bits being noisy (the junta at 150 steps at rate 10 and 450 at rate 30; on
addition at 450 one seed hovers at 0.979 until 1450), so chattering is the piecewise-constant
gradient's, not the optimiser's. The deploy gap, zero on the bits at every step, opening and
closing on the soft pass: **held** (claims pass, the first by construction).

*Of the relay-and-alignment note.* The relay is autodiff: **unchanged** (a theorem, tested). The
deep shape as the floor, the reference at 1.000 on every seed within 1000 steps: **moved**, the
path above; under the plain update no single rate reaches on all six tiles, and the plateau
below is a soft-pass solution short of the target. The relay reaches with or without σ′ because
Adam does not see a positive factor: **moved**; with σ′ it is the reference, above; without it
the relay reaches on all six tiles at 1500, on none at 700 to 1000 (one tile at 0.992 to 0.996
for 12000 steps) and collapses to 0.65 to 0.72 at 2000: the factor now sets the band, and the
two bands do not overlap. The blind split stalls short on the soft pass: **held on addition**
(uniform 0.87 to 0.96 at every rate, single seeds at 1.000 at five rates, never every seed; under
`DEEP_RELAY` the claim passes on the recipe seeds) and, as the audit already scoped on the bits,
**not on the junta**, where the soft uniform split reaches at every rate by 250 to 500 steps. On
the bits the exact relay to the logit or to the entry does not leave chance at any rate, budget
or optimiser: **moved for the logit, held for the entry**; straight-through at 1000 reaches
0.939 ± 0.011 (0.926 · 0.953 · 0.938 at 4000 steps; 0.867 · 0.953 · 0.934 at 2000), 0.90 to
0.94 on [700, 2000], chance below 300 and above 30000; the relay to the entry is at 0.46 to 0.51
at every rate on addition, and the only difference between the two cells is σ′, which the plain
update keeps as a weight (the reading, that σ′ gives a saturated entry inertia, a "stay" the bare
vote lacks, is not measured). The blind split trains to 0.84 to 0.98 on the bits and never to
1.000: **held**, 0.86 to 0.96, with one run of thirty-three at 1.000 (rate 10, seed 2, from 1000
steps on). Dead paths, thrash, who learns, alignment, the votes: **not re-measured**; they are
statements about the signals under Adam's flips, and the ones that involve the optimiser (flips
per step) are owed a plain-update run. On the junta at depth the blind split reaches on every
run and the relay does not (0.47 to 0.61): **held for the split** (every rate, by 250 to 500),
**moved for the relay**, which now reaches on two seeds of three at some rates (seed 0 above
1000, seed 1 at 30) and never on the third (0.40 to 0.56).

*Of the direct-feedback note.* The flip credit trains the bits and stops exactly: **held**
(0.85 to 0.91, identical at 1000, 2000 and 4000 steps at every rate from 10 up, 0.910 ± 0.023
from 3000 up; on the junta 0.97 to 1.000, every seed at 1.000 at 300; the fixed-point claim
passes). The random bus trains worst: **held** (bits 0.62 to 0.72, soft 0.63 to 0.80, best at the
lowest rates and falling with the rate; the ordering claim passes). Masked to reachable outputs
the bus trains like the wiring-shaped split: **held** (bits 0.88 to 0.99, 0.986 ± 0.020 at 100
with two seeds at 1.000, 0.947 ± 0.039 rate-free above 1000; soft 0.89 to 0.94; the claim
passes). Alignment's own quantity and the depth boundary (one hidden layer): **not re-measured**.
The per-gate signal does not leave chance: **held** (the claim passes). "Three signals train at
depth and the one autodiff would suggest is the one that does not": **moved**, four train, any
fixed feedback on the reachability, the flip credit, and the exact relay with σ′ kept; the exact
relay without it is the one that does not.

*Of `docs/signals.md` §depth.* Reworded where the above moved: the relay to the entry at its own
rate; straight-through trains at depth under the plain update; the flip credit rate-free; four
signals. The section's ranges from the Adam notes became words with a pointer here.

*New, not in any earlier note.* The bits cells to the entry are rate-free above a threshold
(above). The soft relay to the entry at a high enough rate saturates every table in the first
steps and is thereafter the bits' relay: at chance on addition from 3000 up on the flat tile and
2000 up on the deep one, at the target on the junta at every rate, exactly where
`hard.relay.entry` is. And the flat tile's straight-through reaches 2-bit addition, which no
earlier claim asserted, on every seed in [300, 700]: `HARD_FLOOR` now holds on both flat tasks.

**The online row.** With window 1 the reference's plateau on the flat tile, [10, 1000], is nearly
the batched one, [10, 3000]: the seed divides by the window's size and the address sum grows with
the cases, so the per-entry signal has the same size online and batched, and the rate carries
over. At 100 every probe seed is at 1.000 by 100 steps on the junta and 250 on addition, which
`ONLINE_FLOOR` pins at 500; under Adam the same claim needed 3000. The soft relay to the entry
online wants a decade less ([1, 30] on addition, [1, 100] on the junta) and is erratic above,
consistent with its σ′-free push overshooting one case at a time. The deep shape was not run
online: that row belongs to the workshop, where the window is an axis of its own.

**Claims left behind.** Recipes, before → after:

| recipe | before (Adam) | after (plain) | from |
|---|---|---|---|
| `SOFT_FLOOR` | 0.1, 500 steps | 100, 400 steps | the flat reference's plateau [10, 3000], 200 steps ×2 |
| `HARD_FLOOR` | 0.02, 2000 | 450, 3000 | straight-through's band on addition [300, 700] inside the junta's; addition's slowest seed at 1450, ×2 |
| `ONLINE_FLOOR` | 0.05, 3000 | 100, 500 | the online reference's plateau [10, 1000], 250 steps ×2 |
| `DEEP_FLOOR` | 0.1, 4000 | 3000, 4000 | the point 3000 on the probe seeds; the full budget, since no rate reaches on every tile and the claim is about the landscape |
| `DEEP_HARD_FLOOR` | 0.02, 2000 | 1000, 2000 | straight-through's band at depth; the junta's 750 ×2, rounded to the tables' grain, where every bits cell has come to rest or plateaued |
| `DEEP_RELAY` | — | 1500, 4000 | new: the relay to the entry's own point; 2000 steps ×2 |

Claims, by file. `test_2026_09_03_one_tile_computes.py`: all three **unchanged** apart from the
recipes they name. `test_2026_09_07_straight_through.py`: all four (five with the matrix cell)
**unchanged**. `test_2026_09_07_transports.py`: the uniform split losing the sign after one hop
**unchanged** (no training in it); the exact relay dead on the bits where the blind split is not
**unchanged**; descent on the relay reaching where the blind split stalls **re-stated** under
`DEEP_RELAY` (under `DEEP_FLOOR`'s rate the relay is at chance, the second new claim);
`test_addition_under_the_deep_floor` **superseded**: it asserted 1.000 on every seed (it held
under Adam 0.1 for 4000 steps) and under `DEEP_FLOOR` the recipe seeds give 1.000 · 0.992 · 0.988,
with no rate of 2000, 3000 or 4500 reaching on all six tiles tried; it is removed from the file
and re-earned as the first claim of `claims/test_2026_09_08_the_floors_under_plain_descent.py`:
the plain reference comes within two bits of the target on every seed and the relay to the entry
under `DEEP_RELAY` reaches it. `test_2026_09_07_direct_feedback_and_the_flip_credit.py`: all four
**unchanged**. New, in the new file: the same signal short under the reference's rate and at the
target under its own; straight-through trains the bits at depth where the relay to the entry
does not; two-bit addition under the hard floor. Nineteen claims, all green under the recipe
seeds; 22 s at `-n 8` on this CPU (the `-n 2` time is in the chunk's report).

**What would change it.** More tiles at depth: six is few for a plateau that is a point, and a
finer grid than ×1.5, or a rate schedule, might find a band the reference shares across tiles;
until then the deep floor is the relay to the entry. Why the large step avoids the soft plateau,
which was read, not measured. The σ′-as-inertia reading of straight-through on the bits (flips
per step against the relay to the entry at their own rates). The mechanism probes of the two
2026-09-07 notes re-run under the plain update, alignment and votes included. The deep shape
online. And the workshop of step 2, whose learned η is exactly the pin this note sets by hand,
which is what makes the plain update the floor it meta-learns against.

## Appendix: the sweeps in full

**Measured, the flat tile.** Per cell, every row is a rate, every seed cell is the accuracy at
the two budgets, and rows whose every digit is identical are one row with a rate range.

*The soft pass, batched.*

*2-junta 4→2: soft.autodiff.logit* (hard accuracy at 500 / 2000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.438 / 0.688 | 0.469 / 0.969 | 0.531 / 0.750 | **0.802 ± 0.121** | never |
| 3 | 0.625 / 1.000 | 0.875 / 1.000 | 0.625 / 1.000 | **1.000 ± 0.000** | 1700 |
| 10 | 1.000 / 1.000 | 1.000 / 1.000 | 0.938 / 1.000 | **1.000 ± 0.000** | 550 |
| 30 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 200 |
| 100 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 100 |
| 300–30000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 50 |
| 100000 | 1.000 / 1.000 | 0.750 / 0.750 | 1.000 / 1.000 | **0.917 ± 0.118** | never |

*2-junta 4→2: soft.relay.entry* (hard accuracy at 500 / 2000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.750 / 1.000 | 0.969 / 1.000 | 0.719 / 1.000 | **1.000 ± 0.000** | 1100 |
| 3 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 400 |
| 10 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 150 |
| 30–1000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 50 |
| 3000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 100 |
| 10000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 50 |
| 30000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 100 |
| 100000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 50 |

*addition 2-bit: soft.autodiff.logit* (hard accuracy at 500 / 2000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.521 / 0.646 | 0.604 / 0.625 | 0.396 / 0.583 | **0.618 ± 0.026** | never |
| 3 | 0.562 / 0.938 | 0.625 / 0.812 | 0.521 / 0.875 | **0.875 ± 0.051** | never |
| 10 | 0.896 / 1.000 | 0.729 / 1.000 | 0.792 / 1.000 | **1.000 ± 0.000** | 1550 |
| 30 | 1.000 / 1.000 | 1.000 / 1.000 | 0.979 / 1.000 | **1.000 ± 0.000** | 550 |
| 100 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 200 |
| 300 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 100 |
| 1000–3000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 50 |
| 10000 | 1.000 / 1.000 | 0.979 / 0.979 | 1.000 / 1.000 | **0.993 ± 0.010** | never |
| 30000 | 1.000 / 1.000 | 0.979 / 0.979 | 0.875 / 0.875 | **0.951 ± 0.055** | never |
| 100000 | 0.875 / 0.875 | 0.688 / 0.708 | 0.854 / 0.875 | **0.819 ± 0.079** | never |

*addition 2-bit: soft.relay.entry* (hard accuracy at 500 / 2000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.646 / 0.979 | 0.625 / 0.979 | 0.583 / 0.938 | **0.965 ± 0.020** | never |
| 3 | 0.979 / 1.000 | 0.917 / 1.000 | 0.875 / 1.000 | **1.000 ± 0.000** | 900 |
| 10 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 300 |
| 30 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 100 |
| 100–300 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 50 |
| 1000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 150 |
| 3000 | 0.458 / 0.438 | 0.625 / 1.000 | 0.479 / 0.625 | **0.688 ± 0.234** | never |
| 10000 | 0.583 / 0.521 | 0.500 / 0.500 | 0.458 / 0.500 | **0.507 ± 0.010** | never |
| 30000 | 0.542 / 0.500 | 0.583 / 0.583 | 0.375 / 0.417 | **0.500 ± 0.068** | never |
| 100000 | 0.500 / 0.479 | 0.354 / 0.438 | 0.646 / 0.500 | **0.472 ± 0.026** | never |

*The bits, batched.*

*2-junta 4→2: hard.autodiff.logit* (hard accuracy at 500 / 2000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 1.000 / 1.000 | 1.000 / 1.000 | 0.719 / 1.000 | **1.000 ± 0.000** | 900 |
| 3 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 350 |
| 10 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 150 |
| 30 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 450 |
| 100 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 350 |
| 300–1000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 100 |
| 3000 | 0.875 / 0.875 | 0.844 / 0.844 | 0.656 / 0.656 | **0.792 ± 0.097** | never |
| 10000 | 0.781 / 0.812 | 0.812 / 0.844 | 0.688 / 0.688 | **0.781 ± 0.068** | never |
| 30000–100000 | 0.781 / 0.781 | 0.812 / 0.812 | 0.688 / 0.688 | **0.760 ± 0.053** | never |

*2-junta 4→2: hard.relay.entry* (hard accuracy at 500 / 2000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 250 |
| 3 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 150 |
| 10 | 1.000 / 1.000 | 1.000 / 1.000 | 0.500 / 1.000 | **1.000 ± 0.000** | 750 |
| 30 | 0.562 / 1.000 | 1.000 / 1.000 | 0.781 / 1.000 | **1.000 ± 0.000** | 1200 |
| 100 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 450 |
| 300–100000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 400 |

*addition 2-bit: hard.autodiff.logit* (hard accuracy at 500 / 2000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.542 / 0.479 | 0.625 / 1.000 | 0.542 / 0.542 | **0.674 ± 0.232** | never |
| 3 | 0.604 / 0.542 | 0.562 / 0.583 | 0.542 / 0.562 | **0.562 ± 0.017** | never |
| 10 | 0.354 / 0.438 | 0.542 / 0.500 | 0.458 / 0.500 | **0.479 ± 0.029** | never |
| 30 | 0.542 / 0.500 | 0.458 / 0.521 | 0.375 / 0.521 | **0.514 ± 0.010** | never |
| 100 | 0.583 / 0.521 | 0.646 / 0.562 | 0.500 / 0.625 | **0.569 ± 0.043** | never |
| 300 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 350 |
| 1000 | 0.958 / 0.979 | 0.979 / 1.000 | 0.979 / 1.000 | **0.993 ± 0.010** | never |
| 3000 | 0.750 / 0.833 | 0.625 / 0.625 | 0.771 / 0.812 | **0.757 ± 0.094** | never |
| 10000 | 0.604 / 0.604 | 0.562 / 0.562 | 0.646 / 0.646 | **0.604 ± 0.034** | never |
| 30000–100000 | 0.604 / 0.604 | 0.562 / 0.562 | 0.625 / 0.625 | **0.597 ± 0.026** | never |

*addition 2-bit: hard.relay.entry* (hard accuracy at 500 / 2000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.479 / 0.479 | 0.521 / 0.583 | 0.500 / 0.458 | **0.507 ± 0.055** | never |
| 3 | 0.479 / 0.583 | 0.417 / 0.292 | 0.438 / 0.625 | **0.500 ± 0.148** | never |
| 10 | 0.542 / 0.542 | 0.688 / 0.771 | 0.542 / 0.479 | **0.597 ± 0.125** | never |
| 30 | 0.500 / 0.625 | 0.375 / 0.542 | 0.500 / 0.458 | **0.542 ± 0.068** | never |
| 100 | 0.479 / 0.542 | 0.562 / 0.479 | 0.479 / 0.562 | **0.528 ± 0.035** | never |
| 300–100000 | 0.583 / 0.500 | 0.521 / 0.562 | 0.521 / 0.458 | **0.507 ± 0.043** | never |

*Online, window 1.*

*2-junta 4→2: soft.autodiff.logit W=1* (hard accuracy at 1000 / 3000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.500 / 0.938 | 0.594 / 0.969 | 0.531 / 0.812 | **0.906 ± 0.068** | never |
| 3 | 0.938 / 1.000 | 0.969 / 1.000 | 0.781 / 1.000 | **1.000 ± 0.000** | 1800 |
| 10 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 600 |
| 30 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 200 |
| 100 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 100 |
| 300 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 50 |
| 1000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 150 |
| 3000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.969 / 0.969 | **0.990 ± 0.015** | never |
| 10000 | 0.500 / 0.500 | 0.500 / 0.500 | 0.750 / 0.750 | **0.583 ± 0.118** | never |
| 30000–100000 | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 | **0.500 ± 0.000** | never |

*2-junta 4→2: soft.relay.entry W=1* (hard accuracy at 1000 / 3000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.969 / 1.000 | 0.969 / 1.000 | 0.906 / 1.000 | **1.000 ± 0.000** | 1250 |
| 3 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 450 |
| 10 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 150 |
| 30–100 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 100 |
| 300 | 0.688 / 0.781 | 1.000 / 1.000 | 0.562 / 1.000 | **0.927 ± 0.103** | never |
| 1000 | 0.562 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 1350 |
| 3000 | 0.562 / 0.656 | 0.875 / 1.000 | 0.812 / 1.000 | **0.885 ± 0.162** | never |
| 10000 | 0.594 / 1.000 | 0.844 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 2750 |
| 30000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 850 |
| 100000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.875 / 0.812 | **0.938 ± 0.088** | never |

*addition 2-bit: soft.autodiff.logit W=1* (hard accuracy at 1000 / 3000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.542 / 0.729 | 0.625 / 0.604 | 0.458 / 0.604 | **0.646 ± 0.059** | never |
| 3 | 0.729 / 0.979 | 0.667 / 0.958 | 0.583 / 0.917 | **0.951 ± 0.026** | never |
| 10 | 0.979 / 1.000 | 1.000 / 1.000 | 0.938 / 1.000 | **1.000 ± 0.000** | 1800 |
| 30 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 650 |
| 100 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 250 |
| 300 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 100 |
| 1000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 200 |
| 3000 | 0.958 / 0.979 | 0.958 / 0.958 | 0.812 / 0.896 | **0.944 ± 0.035** | never |
| 10000 | 0.667 / 0.667 | 0.625 / 0.625 | 0.562 / 0.583 | **0.625 ± 0.034** | never |
| 30000–100000 | 0.542 / 0.542 | 0.542 / 0.542 | 0.458 / 0.458 | **0.514 ± 0.039** | never |

*addition 2-bit: soft.relay.entry W=1* (hard accuracy at 1000 / 3000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.896 / 1.000 | 0.688 / 1.000 | 0.771 / 1.000 | **1.000 ± 0.000** | 2850 |
| 3 | 1.000 / 1.000 | 1.000 / 1.000 | 0.958 / 1.000 | **1.000 ± 0.000** | 1300 |
| 10 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 400 |
| 30 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 ± 0.000** | 200 |
| 100 | 0.646 / 0.583 | 0.729 / 0.646 | 0.583 / 0.583 | **0.604 ± 0.029** | never |
| 300 | 0.417 / 0.583 | 0.542 / 0.667 | 0.604 / 0.521 | **0.590 ± 0.060** | never |
| 1000 | 0.688 / 0.521 | 0.562 / 0.479 | 0.542 / 0.583 | **0.528 ± 0.043** | never |
| 3000 | 0.583 / 0.458 | 0.479 / 0.583 | 0.604 / 0.625 | **0.556 ± 0.071** | never |
| 10000 | 0.604 / 0.562 | 0.500 / 0.667 | 0.625 / 0.688 | **0.639 ± 0.055** | never |
| 30000 | 0.521 / 0.479 | 0.604 / 0.583 | 0.667 / 0.562 | **0.542 ± 0.045** | never |
| 100000 | 0.521 / 0.625 | 0.417 / 0.542 | 0.479 / 0.562 | **0.576 ± 0.035** | never |

**Measured, the deep tile.**

*The soft pass.*

*addition 6-bit: soft.autodiff.logit* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.492 / 0.488 / 0.488 | 0.469 / 0.469 / 0.473 | 0.555 / 0.543 / 0.562 | **0.508 ± 0.039** | never |
| 3 | 0.488 / 0.484 / 0.488 | 0.457 / 0.496 / 0.574 | 0.543 / 0.547 / 0.582 | **0.548 ± 0.042** | never |
| 10 | 0.500 / 0.668 / 0.859 | 0.559 / 0.773 / 0.852 | 0.566 / 0.680 / 0.926 | **0.879 ± 0.033** | never |
| 30 | 0.859 / 0.859 / 0.875 | 0.820 / 0.875 / 0.875 | 0.840 / 0.961 / 0.969 | **0.906 ± 0.044** | never |
| 100 | 0.875 / 0.875 / 0.895 | 0.875 / 0.875 / 0.883 | 0.969 / 0.969 / 0.969 | **0.915 ± 0.038** | never |
| 300 | 0.879 / 0.906 / 0.973 | 0.875 / 0.883 / 0.895 | 0.969 / 0.969 / 0.969 | **0.945 ± 0.036** | never |
| 1000 | 0.938 / 1.000 / 1.000 | 0.895 / 0.898 / 0.945 | 0.969 / 0.969 / 0.969 | **0.971 ± 0.022** | never |
| 3000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 750 |
| 10000 | 0.992 / 0.992 / 0.992 | 1.000 / 1.000 / 1.000 | 0.996 / 1.000 / 1.000 | **0.997 ± 0.004** | never |
| 30000 | 0.734 / 0.750 / 0.746 | 0.676 / 0.691 / 0.691 | 0.836 / 0.828 / 0.844 | **0.760 ± 0.063** | never |
| 100000 | 0.523 / 0.523 / 0.523 | 0.641 / 0.766 / 0.777 | 0.520 / 0.520 / 0.570 | **0.624 ± 0.110** | never |

*addition 6-bit: soft.relay.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.488 / 0.496 / 0.613 | 0.473 / 0.535 / 0.762 | 0.562 / 0.559 / 0.621 | **0.665 ± 0.068** | never |
| 3 | 0.574 / 0.836 / 0.867 | 0.645 / 0.812 / 0.875 | 0.586 / 0.738 / 0.949 | **0.897 ± 0.037** | never |
| 10 | 0.867 / 0.914 / 0.914 | 0.875 / 0.875 / 0.883 | 0.941 / 0.980 / 0.980 | **0.926 ± 0.041** | never |
| 30 | 0.914 / 0.914 / 0.914 | 0.887 / 0.883 / 0.898 | 0.980 / 0.980 / 0.992 | **0.935 ± 0.041** | never |
| 100 | 0.914 / 0.914 / 0.949 | 0.898 / 0.898 / 0.961 | 0.984 / 0.996 / 1.000 | **0.970 ± 0.022** | never |
| 300 | 0.984 / 0.984 / 0.984 | 0.953 / 0.953 / 0.953 | 0.969 / 0.969 / 0.969 | **0.969 ± 0.013** | never |
| 1000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.992 / 0.992 / 0.992 | **0.997 ± 0.004** | never |
| 3000 | 0.523 / 0.539 / 0.500 | 0.516 / 0.676 / 0.441 | 0.523 / 0.523 / 0.652 | **0.531 ± 0.089** | never |
| 10000 | 0.484 / 0.496 / 0.516 | 0.492 / 0.465 / 0.516 | 0.523 / 0.562 / 0.566 | **0.533 ± 0.024** | never |
| 30000 | 0.543 / 0.504 / 0.496 | 0.559 / 0.523 / 0.516 | 0.500 / 0.508 / 0.520 | **0.510 ± 0.010** | never |
| 100000 | 0.531 / 0.535 / 0.527 | 0.484 / 0.457 / 0.516 | 0.516 / 0.512 / 0.527 | **0.523 ± 0.006** | never |

*addition 6-bit: soft.uniform.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.891 / 0.934 / 0.938 | 0.895 / 0.926 / 0.922 | 0.859 / 0.855 / 0.879 | **0.913 ± 0.025** | never |
| 3 | 0.914 / 0.938 / 0.953 | 0.918 / 0.957 / 0.945 | 0.828 / 0.879 / 0.875 | **0.924 ± 0.035** | never |
| 10 | 0.961 / 0.969 / 0.922 | 0.977 / 0.910 / 0.949 | 0.887 / 0.863 / 0.828 | **0.900 ± 0.052** | never |
| 30 | 0.953 / 0.965 / 0.914 | 0.984 / 0.949 / 1.000 | 0.852 / 0.891 / 0.891 | **0.935 ± 0.047** | never |
| 100 | 0.957 / 0.926 / 0.934 | 0.914 / 0.938 / 0.965 | 0.910 / 0.930 / 0.922 | **0.940 ± 0.018** | never |
| 300 | 0.965 / 0.945 / 0.961 | 0.973 / 0.980 / 1.000 | 0.887 / 0.938 / 0.902 | **0.954 ± 0.040** | never |
| 1000 | 0.945 / 0.875 / 0.801 | 0.898 / 0.961 / 0.957 | 0.859 / 0.887 / 0.859 | **0.872 ± 0.064** | never |
| 3000 | 0.945 / 0.840 / 0.879 | 0.891 / 0.898 / 1.000 | 0.844 / 0.855 / 0.801 | **0.893 ± 0.082** | never |
| 10000 | 0.938 / 0.973 / 0.957 | 0.879 / 0.926 / 0.914 | 1.000 / 1.000 / 1.000 | **0.957 ± 0.035** | never |
| 30000 | 0.953 / 0.953 / 0.906 | 0.852 / 0.914 / 0.984 | 0.863 / 0.805 / 0.859 | **0.917 ± 0.052** | never |
| 100000 | 0.828 / 0.883 / 0.887 | 0.957 / 1.000 / 1.000 | 0.883 / 0.859 / 0.762 | **0.883 ± 0.097** | never |

*addition 6-bit: soft.direct.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.742 / 0.812 / 0.836 | 0.797 / 0.797 / 0.777 | 0.695 / 0.699 / 0.742 | **0.785 ± 0.039** | never |
| 3 | 0.828 / 0.824 / 0.801 | 0.805 / 0.820 / 0.773 | 0.742 / 0.707 / 0.793 | **0.789 ± 0.011** | never |
| 10 | 0.836 / 0.805 / 0.816 | 0.805 / 0.785 / 0.812 | 0.754 / 0.727 / 0.766 | **0.798 ± 0.023** | never |
| 30 | 0.820 / 0.836 / 0.801 | 0.797 / 0.773 / 0.801 | 0.711 / 0.762 / 0.766 | **0.789 ± 0.017** | never |
| 100 | 0.789 / 0.727 / 0.758 | 0.797 / 0.738 / 0.801 | 0.723 / 0.754 / 0.746 | **0.768 ± 0.024** | never |
| 300 | 0.730 / 0.762 / 0.680 | 0.770 / 0.758 / 0.793 | 0.688 / 0.668 / 0.707 | **0.727 ± 0.048** | never |
| 1000 | 0.746 / 0.633 / 0.625 | 0.625 / 0.559 / 0.703 | 0.680 / 0.734 / 0.656 | **0.661 ± 0.032** | never |
| 3000 | 0.641 / 0.703 / 0.766 | 0.762 / 0.641 / 0.672 | 0.613 / 0.684 / 0.684 | **0.707 ± 0.042** | never |
| 10000 | 0.719 / 0.668 / 0.672 | 0.727 / 0.711 / 0.660 | 0.562 / 0.645 / 0.617 | **0.650 ± 0.024** | never |
| 30000 | 0.684 / 0.758 / 0.652 | 0.707 / 0.672 / 0.672 | 0.656 / 0.645 / 0.559 | **0.628 ± 0.049** | never |
| 100000 | 0.664 / 0.793 / 0.676 | 0.645 / 0.754 / 0.539 | 0.656 / 0.617 / 0.664 | **0.626 ± 0.062** | never |

*addition 6-bit: soft.reachable.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.871 / 0.898 / 0.957 | 0.832 / 0.918 / 0.945 | 0.840 / 0.883 / 0.867 | **0.923 ± 0.040** | never |
| 3 | 0.945 / 0.965 / 0.938 | 0.906 / 0.969 / 0.957 | 0.867 / 0.891 / 0.898 | **0.931 ± 0.024** | never |
| 10 | 0.953 / 0.941 / 0.941 | 0.953 / 0.977 / 0.945 | 0.910 / 0.887 / 0.914 | **0.934 ± 0.014** | never |
| 30 | 0.961 / 0.957 / 0.938 | 0.977 / 0.914 / 0.980 | 0.906 / 0.902 / 0.895 | **0.938 ± 0.035** | never |
| 100 | 0.953 / 0.938 / 0.945 | 0.980 / 0.922 / 0.906 | 0.898 / 0.898 / 0.902 | **0.918 ± 0.019** | never |
| 300 | 0.984 / 0.965 / 0.938 | 0.949 / 0.922 / 0.902 | 0.922 / 0.934 / 0.898 | **0.913 ± 0.018** | never |
| 1000 | 0.926 / 0.922 / 0.934 | 0.953 / 0.977 / 0.891 | 0.879 / 0.824 / 0.848 | **0.891 ± 0.035** | never |
| 3000 | 0.926 / 0.898 / 0.934 | 0.945 / 0.902 / 0.961 | 0.777 / 0.961 / 0.906 | **0.934 ± 0.022** | never |
| 10000 | 0.930 / 0.922 / 0.945 | 1.000 / 1.000 / 1.000 | 0.828 / 0.844 / 0.848 | **0.931 ± 0.063** | never |
| 30000 | 0.844 / 0.922 / 0.953 | 0.961 / 0.992 / 0.977 | 0.953 / 0.812 / 0.875 | **0.935 ± 0.043** | never |
| 100000 | 0.941 / 0.926 / 0.797 | 0.965 / 0.902 / 0.934 | 0.961 / 0.820 / 0.945 | **0.892 ± 0.067** | never |

*2-junta 6→4: soft.autodiff.logit* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.504 / 0.512 / 0.594 | 0.477 / 0.492 / 0.676 | 0.418 / 0.539 / 0.625 | **0.632 ± 0.034** | never |
| 3 | 0.523 / 0.617 / 0.895 | 0.578 / 0.770 / 0.770 | 0.578 / 0.797 / 0.879 | **0.848 ± 0.056** | never |
| 10 | 0.820 / 1.000 / 1.000 | 0.770 / 0.898 / 1.000 | 0.883 / 1.000 / 1.000 | **1.000 ± 0.000** | 3500 |
| 30 | 1.000 / 1.000 / 1.000 | 0.938 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 1250 |
| 100 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 500 |
| 300–10000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 250 |
| 30000 | 1.000 / 1.000 / 1.000 | 0.875 / 0.875 / 0.875 | 1.000 / 1.000 / 1.000 | **0.958 ± 0.059** | never |
| 100000 | 0.875 / 0.875 / 0.875 | 0.688 / 0.688 / 0.688 | 0.750 / 0.750 / 0.750 | **0.771 ± 0.078** | never |

*2-junta 6→4: soft.relay.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.594 / 0.797 / 1.000 | 0.711 / 0.770 / 0.895 | 0.641 / 0.879 / 1.000 | **0.965 ± 0.050** | never |
| 3 | 0.902 / 1.000 / 1.000 | 0.785 / 0.938 / 1.000 | 0.883 / 1.000 / 1.000 | **1.000 ± 0.000** | 2250 |
| 10 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 750 |
| 30–1000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 250 |
| 3000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 1000 |
| 10000 | 1.000 / 1.000 / 1.000 | 0.473 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 2000 |
| 30000 | 0.891 / 1.000 / 1.000 | 0.469 / 0.531 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 2750 |
| 100000 | 0.754 / 0.867 / 1.000 | 0.605 / 0.875 / 1.000 | 0.777 / 1.000 / 1.000 | **1.000 ± 0.000** | 3000 |

*2-junta 6→4: soft.uniform.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 500 |
| 3–100000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 250 |

*2-junta 6→4: soft.direct.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.859 / 0.938 / 1.000 | 0.719 / 0.734 / 0.734 | 0.922 / 1.000 / 1.000 | **0.911 ± 0.125** | never |
| 3 | 1.000 / 1.000 / 1.000 | 0.656 / 0.750 / 0.781 | 1.000 / 0.984 / 1.000 | **0.927 ± 0.103** | never |
| 10 | 1.000 / 1.000 / 1.000 | 0.781 / 0.703 / 0.750 | 0.992 / 1.000 / 1.000 | **0.917 ± 0.118** | never |
| 30 | 1.000 / 1.000 / 1.000 | 0.625 / 0.797 / 0.812 | 1.000 / 1.000 / 1.000 | **0.938 ± 0.088** | never |
| 100 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 500 |
| 300 | 1.000 / 1.000 / 1.000 | 0.781 / 1.000 / 1.000 | 1.000 / 1.000 / 0.875 | **0.958 ± 0.059** | 1500 |
| 1000 | 1.000 / 1.000 / 1.000 | 0.828 / 0.625 / 0.656 | 1.000 / 1.000 / 1.000 | **0.885 ± 0.162** | never |
| 3000 | 1.000 / 1.000 / 1.000 | 0.766 / 0.719 / 0.688 | 1.000 / 1.000 / 1.000 | **0.896 ± 0.147** | never |
| 10000 | 1.000 / 1.000 / 1.000 | 0.719 / 0.875 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 2750 |
| 30000 | 1.000 / 1.000 / 1.000 | 0.641 / 0.719 / 0.844 | 1.000 / 1.000 / 1.000 | **0.948 ± 0.074** | never |
| 100000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 250 |

*2-junta 6→4: soft.reachable.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 750 |
| 3–100000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 250 |

*The bits.*

*addition 6-bit: hard.autodiff.logit* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.531 / 0.523 / 0.473 | 0.480 / 0.531 / 0.469 | 0.535 / 0.473 / 0.520 | **0.487 ± 0.023** | never |
| 3 | 0.523 / 0.551 / 0.508 | 0.512 / 0.578 / 0.484 | 0.430 / 0.555 / 0.496 | **0.496 ± 0.010** | never |
| 10 | 0.508 / 0.562 / 0.473 | 0.441 / 0.496 / 0.547 | 0.562 / 0.492 / 0.500 | **0.507 ± 0.031** | never |
| 30 | 0.504 / 0.523 / 0.473 | 0.500 / 0.473 / 0.496 | 0.539 / 0.555 / 0.508 | **0.492 ± 0.015** | never |
| 100 | 0.484 / 0.539 / 0.559 | 0.535 / 0.527 / 0.527 | 0.559 / 0.492 / 0.539 | **0.542 ± 0.013** | never |
| 300 | 0.457 / 0.527 / 0.449 | 0.508 / 0.480 / 0.617 | 0.426 / 0.539 / 0.613 | **0.560 ± 0.078** | never |
| 1000 | 0.863 / 0.867 / 0.926 | 0.953 / 0.953 / 0.953 | 0.754 / 0.934 / 0.938 | **0.939 ± 0.011** | never |
| 3000 | 0.672 / 0.730 / 0.746 | 0.805 / 0.816 / 0.785 | 0.812 / 0.863 / 0.867 | **0.799 ± 0.050** | never |
| 10000 | 0.691 / 0.660 / 0.676 | 0.688 / 0.695 / 0.695 | 0.609 / 0.617 / 0.613 | **0.661 ± 0.035** | never |
| 30000 | 0.527 / 0.527 / 0.520 | 0.535 / 0.543 / 0.543 | 0.531 / 0.539 / 0.539 | **0.534 ± 0.010** | never |
| 100000 | 0.516 / 0.516 / 0.516 | 0.527 / 0.527 / 0.527 | 0.520 / 0.520 / 0.520 | **0.521 ± 0.005** | never |

*addition 6-bit: hard.relay.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.516 / 0.488 / 0.504 | 0.465 / 0.559 / 0.422 | 0.469 / 0.523 / 0.453 | **0.460 ± 0.034** | never |
| 3 | 0.461 / 0.512 / 0.535 | 0.520 / 0.492 / 0.484 | 0.508 / 0.562 / 0.504 | **0.508 ± 0.021** | never |
| 10 | 0.547 / 0.527 / 0.492 | 0.465 / 0.531 / 0.496 | 0.504 / 0.438 / 0.500 | **0.496 ± 0.003** | never |
| 30 | 0.512 / 0.480 / 0.465 | 0.543 / 0.473 / 0.520 | 0.457 / 0.500 / 0.430 | **0.471 ± 0.037** | never |
| 100 | 0.383 / 0.508 / 0.500 | 0.516 / 0.469 / 0.445 | 0.508 / 0.465 / 0.527 | **0.491 ± 0.034** | never |
| 300 | 0.574 / 0.488 / 0.461 | 0.504 / 0.449 / 0.496 | 0.531 / 0.535 / 0.516 | **0.491 ± 0.023** | never |
| 1000–30000 | 0.520 / 0.523 / 0.512 | 0.480 / 0.500 / 0.512 | 0.527 / 0.516 / 0.512 | **0.512 ± 0.000** | never |
| 100000 | 0.520 / 0.523 / 0.512 | 0.422 / 0.484 / 0.508 | 0.527 / 0.516 / 0.512 | **0.510 ± 0.002** | never |

*addition 6-bit: hard.uniform.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.902 / 0.902 / 0.945 | 0.902 / 0.883 / 0.926 | 0.867 / 0.918 / 0.969 | **0.947 ± 0.018** | never |
| 3 | 0.855 / 0.926 / 0.938 | 0.840 / 0.883 / 0.844 | 0.836 / 0.832 / 0.801 | **0.861 ± 0.057** | never |
| 10 | 0.887 / 0.914 / 0.918 | 0.926 / 0.914 / 0.969 | 1.000 / 1.000 / 1.000 | **0.962 ± 0.034** | never |
| 30 | 0.863 / 0.887 / 0.914 | 0.816 / 0.879 / 0.934 | 0.938 / 0.871 / 0.758 | **0.868 ± 0.079** | never |
| 100 | 0.832 / 0.867 / 0.770 | 0.910 / 0.852 / 0.922 | 0.832 / 0.910 / 0.883 | **0.858 ± 0.065** | never |
| 300 | 0.898 / 0.949 / 0.926 | 0.871 / 0.824 / 0.977 | 0.828 / 0.871 / 0.969 | **0.957 ± 0.022** | never |
| 1000–100000 | 0.836 / 0.871 / 0.910 | 0.918 / 0.875 / 0.961 | 0.828 / 0.922 / 0.926 | **0.932 ± 0.021** | never |

*addition 6-bit: hard.direct.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.703 / 0.695 / 0.664 | 0.719 / 0.742 / 0.711 | 0.676 / 0.707 / 0.684 | **0.686 ± 0.019** | never |
| 3 | 0.828 / 0.734 / 0.617 | 0.684 / 0.656 / 0.602 | 0.613 / 0.676 / 0.625 | **0.615 ± 0.010** | never |
| 10 | 0.602 / 0.684 / 0.730 | 0.668 / 0.699 / 0.707 | 0.660 / 0.633 / 0.715 | **0.717 ± 0.010** | never |
| 30 | 0.695 / 0.707 / 0.727 | 0.652 / 0.637 / 0.598 | 0.684 / 0.727 / 0.754 | **0.693 ± 0.068** | never |
| 100 | 0.676 / 0.770 / 0.570 | 0.645 / 0.742 / 0.645 | 0.695 / 0.695 / 0.637 | **0.617 ± 0.033** | never |
| 300 | 0.727 / 0.641 / 0.711 | 0.691 / 0.668 / 0.680 | 0.637 / 0.586 / 0.723 | **0.704 ± 0.018** | never |
| 1000–10000 | 0.723 / 0.719 / 0.684 | 0.559 / 0.602 / 0.664 | 0.633 / 0.648 / 0.637 | **0.661 ± 0.019** | never |
| 30000–100000 | 0.723 / 0.719 / 0.684 | 0.559 / 0.602 / 0.664 | 0.633 / 0.676 / 0.613 | **0.654 ± 0.030** | never |

*addition 6-bit: hard.reachable.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.844 / 0.891 / 0.945 | 0.953 / 0.957 / 0.926 | 0.820 / 0.887 / 0.875 | **0.915 ± 0.030** | never |
| 3 | 0.875 / 0.934 / 0.922 | 0.863 / 0.887 / 0.938 | 0.789 / 0.812 / 0.781 | **0.880 ± 0.070** | never |
| 10 | 0.895 / 0.918 / 0.941 | 0.969 / 0.910 / 0.832 | 0.898 / 0.977 / 0.863 | **0.879 ± 0.046** | never |
| 30 | 0.941 / 0.914 / 0.941 | 0.902 / 0.836 / 0.953 | 0.879 / 0.875 / 0.941 | **0.945 ± 0.006** | never |
| 100 | 1.000 / 1.000 / 1.000 | 0.918 / 0.875 / 0.957 | 0.953 / 1.000 / 1.000 | **0.986 ± 0.020** | never |
| 300 | 0.965 / 0.855 / 0.891 | 0.953 / 0.961 / 0.871 | 0.871 / 0.887 / 0.898 | **0.887 ± 0.011** | never |
| 1000–100000 | 0.930 / 0.926 / 0.934 | 1.000 / 1.000 / 1.000 | 0.953 / 0.867 / 0.906 | **0.947 ± 0.039** | never |

*addition 6-bit: hard.flip.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.820 / 0.852 / 0.883 | 0.777 / 0.797 / 0.805 | 0.867 / 0.871 / 0.871 | **0.853 ± 0.034** | never |
| 3 | 0.863 / 0.910 / 0.914 | 0.848 / 0.859 / 0.871 | 0.871 / 0.883 / 0.883 | **0.889 ± 0.018** | never |
| 10 | 0.875 / 0.875 / 0.875 | 0.859 / 0.875 / 0.875 | 0.855 / 0.855 / 0.855 | **0.868 ± 0.009** | never |
| 30 | 0.875 / 0.875 / 0.875 | 0.891 / 0.891 / 0.891 | 0.867 / 0.867 / 0.867 | **0.878 ± 0.010** | never |
| 100 | 0.918 / 0.918 / 0.918 | 0.812 / 0.812 / 0.812 | 0.898 / 0.898 / 0.898 | **0.876 ± 0.046** | never |
| 300 | 0.922 / 0.922 / 0.922 | 0.832 / 0.832 / 0.832 | 0.906 / 0.906 / 0.906 | **0.887 ± 0.039** | never |
| 1000 | 0.891 / 0.891 / 0.891 | 0.891 / 0.891 / 0.891 | 0.906 / 0.906 / 0.906 | **0.896 ± 0.007** | never |
| 3000–100000 | 0.887 / 0.887 / 0.887 | 0.902 / 0.902 / 0.902 | 0.941 / 0.941 / 0.941 | **0.910 ± 0.023** | never |

*2-junta 6→4: hard.autodiff.logit* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.480 / 0.461 / 0.777 | 0.531 / 0.438 / 1.000 | 0.523 / 0.648 / 1.000 | **0.926 ± 0.105** | never |
| 3 | 0.730 / 1.000 / 1.000 | 0.484 / 0.582 / 1.000 | 0.523 / 0.508 / 0.484 | **0.828 ± 0.243** | never |
| 10 | 0.488 / 0.449 / 1.000 | 0.629 / 1.000 / 1.000 | 0.516 / 0.473 / 0.484 | **0.828 ± 0.243** | never |
| 30 | 0.457 / 0.496 / 1.000 | 0.637 / 1.000 / 1.000 | 0.469 / 0.480 / 0.555 | **0.852 ± 0.210** | never |
| 100 | 0.750 / 0.867 / 0.875 | 0.559 / 0.758 / 1.000 | 0.535 / 0.477 / 0.609 | **0.828 ± 0.163** | never |
| 300–1000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 750 |
| 3000 | 0.941 / 0.961 / 0.969 | 0.801 / 0.840 / 0.848 | 0.906 / 0.969 / 0.992 | **0.936 ± 0.063** | never |
| 10000 | 0.695 / 0.707 / 0.727 | 0.660 / 0.641 / 0.648 | 0.723 / 0.750 / 0.750 | **0.708 ± 0.043** | never |
| 30000 | 0.523 / 0.520 / 0.520 | 0.535 / 0.547 / 0.547 | 0.664 / 0.648 / 0.648 | **0.572 ± 0.055** | never |
| 100000 | 0.512 / 0.512 / 0.512 | 0.535 / 0.535 / 0.535 | 0.645 / 0.645 / 0.645 | **0.564 ± 0.058** | never |

*2-junta 6→4: hard.relay.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.480 / 0.484 / 1.000 | 0.496 / 0.582 / 1.000 | 0.355 / 0.480 / 0.500 | **0.833 ± 0.236** | never |
| 3 | 0.500 / 1.000 / 1.000 | 0.637 / 0.805 / 1.000 | 0.445 / 0.551 / 0.473 | **0.824 ± 0.249** | never |
| 10 | 0.637 / 0.875 / 1.000 | 0.656 / 1.000 / 1.000 | 0.547 / 0.535 / 0.562 | **0.854 ± 0.206** | never |
| 30 | 0.570 / 0.641 / 1.000 | 1.000 / 1.000 / 1.000 | 0.504 / 0.531 / 0.438 | **0.812 ± 0.265** | never |
| 100 | 0.555 / 0.527 / 1.000 | 0.496 / 0.621 / 1.000 | 0.539 / 0.531 / 0.402 | **0.801 ± 0.282** | never |
| 300 | 0.551 / 0.660 / 1.000 | 0.430 / 0.516 / 1.000 | 0.551 / 0.457 / 0.504 | **0.835 ± 0.234** | never |
| 1000–30000 | 1.000 / 1.000 / 1.000 | 0.484 / 0.523 / 0.875 | 0.422 / 0.457 / 0.500 | **0.792 ± 0.212** | never |
| 100000 | 1.000 / 1.000 / 1.000 | 0.691 / 0.637 / 1.000 | 0.422 / 0.457 / 0.500 | **0.833 ± 0.236** | never |

*2-junta 6→4: hard.uniform.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 500 |
| 3–100000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 250 |

*2-junta 6→4: hard.direct.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 1.000 / 1.000 / 1.000 | 0.629 / 0.570 / 0.859 | 1.000 / 1.000 / 1.000 | **0.953 ± 0.066** | never |
| 3 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 1000 |
| 10 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 750 |
| 30 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.789 / 1.000 / 1.000 | **1.000 ± 0.000** | 1250 |
| 100 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 750 |
| 300 | 1.000 / 1.000 / 1.000 | 0.695 / 0.750 / 0.719 | 0.719 / 0.688 / 0.812 | **0.844 ± 0.117** | never |
| 1000–100000 | 0.797 / 0.727 / 0.766 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **0.922 ± 0.110** | never |

*2-junta 6→4: hard.reachable.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 500 |
| 3–100000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 250 |

*2-junta 6→4: hard.flip.entry* (hard accuracy at 1000 / 2000 / 4000 steps)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1 | 0.930 / 0.949 / 0.969 | 0.973 / 0.973 / 0.973 | 0.961 / 0.969 / 0.973 | **0.971 ± 0.002** | never |
| 3 | 0.930 / 0.980 / 0.980 | 0.992 / 0.996 / 1.000 | 0.965 / 0.980 / 0.980 | **0.987 ± 0.009** | never |
| 10 | 0.961 / 0.961 / 0.961 | 1.000 / 1.000 / 1.000 | 0.965 / 0.965 / 0.965 | **0.975 ± 0.018** | never |
| 30 | 0.957 / 0.957 / 0.957 | 0.965 / 0.965 / 0.965 | 0.984 / 0.984 / 0.984 | **0.969 ± 0.011** | never |
| 100 | 0.992 / 0.992 / 0.992 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **0.997 ± 0.004** | never |
| 300 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 250 |
| 1000 | 0.984 / 0.984 / 0.984 | 0.977 / 0.977 / 0.977 | 1.000 / 1.000 / 1.000 | **0.987 ± 0.010** | never |
| 3000–100000 | 1.000 / 1.000 / 1.000 | 0.938 / 0.938 / 0.938 | 1.000 / 1.000 / 1.000 | **0.979 ± 0.029** | never |

**The plateaus refined** (`probes/2026-09-08-plain-descent-rate-each-signal-wants.py`).

*deep addition, the reference, ×1.5 around 3000* (hard accuracy at a quarter / half / the full budget)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 1000 | 0.938 / 1.000 / 1.000 | 0.895 / 0.898 / 0.945 | 0.969 / 0.969 / 0.969 | **0.971 ± 0.022** | never |
| 1500 | 0.977 / 0.984 / 0.984 | 0.969 / 0.969 / 0.996 | 0.996 / 0.996 / 1.000 | **0.993 ± 0.007** | never |
| 2000 | 1.000 / 1.000 / 1.000 | 0.980 / 0.980 / 0.980 | 0.992 / 0.992 / 0.992 | **0.991 ± 0.008** | never |
| 3000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 750 |
| 4500 | 1.000 / 1.000 / 1.000 | 0.988 / 0.988 / 0.996 | 1.000 / 1.000 / 1.000 | **0.999 ± 0.002** | never |
| 7000 | 0.996 / 0.996 / 0.996 | 0.992 / 0.992 / 1.000 | 1.000 / 1.000 / 1.000 | **0.999 ± 0.002** | never |
| 10000 | 0.992 / 0.992 / 0.992 | 1.000 / 1.000 / 1.000 | 0.996 / 1.000 / 1.000 | **0.997 ± 0.004** | never |

*deep addition, the reference, the low edge at 3× the budget* (hard accuracy at a quarter / half / the full budget)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 100 | 0.879 / 0.910 / 0.973 | 0.875 / 0.883 / 0.891 | 0.969 / 0.969 / 0.969 | **0.944 ± 0.038** | never |
| 300 | 0.973 / 0.973 / 0.980 | 0.883 / 0.898 / 0.945 | 0.969 / 0.969 / 0.969 | **0.965 ± 0.015** | never |
| 1000 | 1.000 / 1.000 / 1.000 | 0.945 / 0.945 / 0.984 | 0.969 / 0.969 / 0.969 | **0.984 ± 0.013** | never |

*deep addition, soft.relay.entry, ×1.5 around 1000* (hard accuracy at a quarter / half / the full budget)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 300 | 0.984 / 0.984 / 0.984 | 0.953 / 0.953 / 0.953 | 0.969 / 0.969 / 0.969 | **0.969 ± 0.013** | never |
| 500 | 1.000 / 1.000 / 1.000 | 0.984 / 0.984 / 0.984 | 0.984 / 0.988 / 0.992 | **0.992 ± 0.006** | never |
| 700–1000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.992 / 0.992 / 0.992 | **0.997 ± 0.004** | never |
| 1500 | 0.793 / 1.000 / 1.000 | 0.891 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 2000 |
| 2000 | 0.547 / 0.691 / 0.602 | 0.609 / 0.551 / 0.594 | 0.656 / 0.652 / 0.754 | **0.650 ± 0.074** | never |
| 300 | 0.984 / 0.984 / 0.984 | 0.953 / 0.984 / 0.984 | 0.969 / 0.969 / 0.969 | **0.979 ± 0.007** | never |
| 500 | 1.000 / 1.000 / 1.000 | 0.984 / 0.984 / 0.984 | 0.992 / 0.992 / 0.992 | **0.992 ± 0.006** | never |
| 700–1000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.992 / 0.992 / 0.992 | **0.997 ± 0.004** | never |
| 1500 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 2000 |
| 2000 | 0.574 / 0.730 / 0.766 | 0.648 / 0.629 / 0.590 | 0.578 / 0.625 / 0.625 | **0.660 ± 0.076** | never |

*deep addition, straight-through, ×1.5 around 1000* (hard accuracy at a quarter / half / the full budget)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 300 | 0.457 / 0.527 / 0.449 | 0.508 / 0.480 / 0.617 | 0.426 / 0.539 / 0.613 | **0.560 ± 0.078** | never |
| 500 | 0.691 / 0.680 / 0.699 | 0.617 / 0.660 / 0.805 | 0.727 / 0.781 / 0.828 | **0.777 ± 0.056** | never |
| 700 | 0.844 / 0.871 / 0.926 | 0.816 / 0.867 / 0.836 | 0.844 / 0.891 / 0.953 | **0.905 ± 0.050** | never |
| 1000 | 0.863 / 0.867 / 0.926 | 0.953 / 0.953 / 0.953 | 0.754 / 0.934 / 0.938 | **0.939 ± 0.011** | never |
| 1500 | 0.812 / 0.867 / 0.926 | 0.875 / 0.910 / 0.926 | 0.879 / 0.910 / 0.891 | **0.914 ± 0.017** | never |
| 2000 | 0.805 / 0.855 / 0.887 | 0.836 / 0.836 / 0.871 | 0.895 / 0.930 / 0.934 | **0.897 ± 0.027** | never |
| 3000 | 0.672 / 0.730 / 0.746 | 0.805 / 0.816 / 0.785 | 0.812 / 0.863 / 0.867 | **0.799 ± 0.050** | never |

*deep 2-junta, straight-through, ×1.5 around 300–1000* (hard accuracy at a quarter / half / the full budget)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 100 | 0.750 / 0.867 / 0.875 | 0.559 / 0.758 / 1.000 | 0.535 / 0.477 / 0.609 | **0.828 ± 0.163** | never |
| 150 | 0.617 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 1500 |
| 200 | 0.758 / 0.895 / 1.000 | 1.000 / 1.000 / 1.000 | 0.543 / 0.539 / 0.488 | **0.829 ± 0.241** | never |
| 300–450 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 750 |
| 700 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 250 |
| 1000–1500 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 750 |
| 2000 | 0.941 / 0.992 / 1.000 | 0.973 / 0.969 / 0.969 | 0.980 / 0.977 / 1.000 | **0.990 ± 0.015** | never |

*flat addition, straight-through, ×1.5 around 300* (hard accuracy at a quarter / half / the full budget)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 100 | 0.583 / 0.458 / 0.521 | 0.646 / 0.583 / 0.562 | 0.500 / 0.667 / 0.625 | **0.569 ± 0.043** | never |
| 150 | 0.688 / 0.479 / 0.479 | 0.375 / 0.458 / 0.542 | 0.562 / 0.604 / 0.667 | **0.562 ± 0.078** | never |
| 200 | 0.688 / 0.625 / 0.542 | 1.000 / 1.000 / 1.000 | 0.479 / 0.521 / 0.521 | **0.688 ± 0.221** | never |
| 300 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 350 |
| 450 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.979 / 0.979 / 1.000 | **1.000 ± 0.000** | 1450 |
| 700 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 250 |
| 1000 | 0.958 / 0.979 / 0.979 | 0.979 / 1.000 / 1.000 | 0.979 / 0.979 / 1.000 | **0.993 ± 0.010** | never |

*flat 2-junta, straight-through, ×1.5 where addition's plateau is* (hard accuracy at a quarter / half / the full budget)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 200–300 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 100 |
| 450–700 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 50 |
| 1000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 100 |

*flat 2-junta, the reference online, ×1.5 inside 10–1000* (hard accuracy at a quarter / half / the full budget)

| rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|
| 30 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 200 |
| 50 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 150 |
| 70–150 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 100 |
| 200–300 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 50 |

*The deep pins under the recipe seeds* (`probes/2026-09-08-plain-descent-recipe-seeds-at-depth.py`;
hard accuracy at 1000 / 2000 / 4000 steps):

| signal | rate | seed 0 | seed 1 | seed 2 | **mean ± sd** | every seed at 1.000 by |
|---|---|---|---|---|---|---|
| soft.autodiff.logit | 2000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 1000 |
| soft.autodiff.logit | 3000 | 0.992 / 1.000 / 1.000 | 0.992 / 0.992 / 0.992 | 0.988 / 0.988 / 0.988 | **0.993 ± 0.005** | never |
| soft.autodiff.logit | 4500 | 1.000 / 1.000 / 1.000 | 0.992 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 1250 |
| soft.relay.entry | 700 | 1.000 / 1.000 / 1.000 | 0.992 / 0.996 / 0.996 | 0.996 / 0.996 / 1.000 | **0.999 ± 0.002** | never |
| soft.relay.entry | 1000 | 1.000 / 1.000 / 1.000 | 0.996 / 0.996 / 0.996 | 1.000 / 1.000 / 1.000 | **0.999 ± 0.002** | never |
| soft.relay.entry | 1500 | 0.652 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | **1.000 ± 0.000** | 1750 |
| soft.relay.entry | 2000 | 0.809 / 0.684 / 0.609 | 0.766 / 0.773 / 0.836 | 0.652 / 0.691 / 0.727 | **0.724 ± 0.093** | never |

*Where the reference stops below its band* (`probes/2026-09-08-plain-descent-deploy-gap-at-depth.py`;
deep addition, probe seeds, 4000 steps; the gradient's norm against its norm at initialisation):

| signal | rate | seed | soft acc | hard acc | loss | entries with \|z\| < 2 | \|g\| / \|g₀\| |
|---|---|---|---|---|---|---|---|
| soft.autodiff.logit | 300 | 0 · 1 · 2 | 0.980 · 0.902 · 0.965 | 0.973 · 0.895 · 0.969 | 0.0070 · 0.0272 · 0.0079 | 0.53 · 0.67 · 0.56 | 0.0042 · 0.0264 · 0.0022 |
| soft.autodiff.logit | 1000 | 0 · 1 · 2 | 1.000 · 0.969 · 0.965 | 1.000 · 0.945 · 0.969 | 0.0000 · 0.0135 · 0.0078 | 0.49 · 0.59 · 0.55 | 0.0007 · 0.0036 · 0.0006 |
| soft.autodiff.logit | 3000 | 0 · 1 · 2 | 1.000 · 1.000 · 1.000 | 1.000 · 1.000 · 1.000 | 0.0000 · 0.0000 · 0.0000 | 0.42 · 0.41 · 0.41 | 0.0002 · 0.0003 · 0.0002 |
| soft.relay.entry | 700 | 0 · 1 · 2 | 1.000 · 1.000 · 1.000 | 1.000 · 1.000 · 0.992 | 0.0000 · 0.0000 · 0.0015 | 0.46 · 0.44 · 0.53 | 0.0000 · 0.0000 · 0.0000 |
| soft.relay.entry | 1500 | 0 · 1 · 2 | 1.000 · 1.000 · 1.000 | 1.000 · 1.000 · 1.000 | 0.0000 · 0.0000 · 0.0000 | 0.29 · 0.26 · 0.28 | 0.0000 · 0.0000 · 0.0000 |

