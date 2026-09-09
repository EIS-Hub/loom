# 2026-09-08 — The floors under plain descent: a rate is a step per vote; at depth the plain reference is a point, and a readout at the cell makes it a band

*From the step-1 record chunk (loom PR #5), built by an encapsulated subagent under a brief and
reviewed on 2026-09-08 and 09; the probes of those dates. The path is recorded, including the
first note's failure to be readable, the convention it forced, and where the smell of rates in the
thousands led.*

**Question.** Every statement of the four earlier notes about which signal descent can follow was
made under Adam. Which of them hold when descent is the plain update, Δ = −lr·s with nothing
normalised, and what rate does each signal want?

**Why plain descent, and what a rate is.** Decided 2026-09-08 (Gabriel): the floor is the plain
update, because an adaptive optimiser had been saying three things about signals that were not true
of them: it normalises every entry, so a signal's size and the σ′ factor never mattered; it carries
two moments per entry, a memory the cost table excludes; on a window of one case it amplifies the
entries a case rarely addresses. Step 2's rule is −η·s with η learned, so the plain update is its
only honest baseline. The **convention** the review forced (2026-09-08, "rates in the thousands are
a smell"): the seed is one vote per case, the residual over the number of cases in the window,
summed over the outputs a gate reaches (`docs/signals.md`, item 2). A rate is then in logit units
per unit average vote, the same online and batched, and every rate below is shown with the step it
implies per layer. The first version of this note carried eight hundred lines of grids and no
intuition; the grids are the probes' printed output (seven minutes on the CPU), not this note.

**Conditions.** The flat tile, hidden `(16, 8)` at arity 4, built to the task's width, on 2-juntas
(4 → 2) and 2-bit addition (4 → 3), batched and online; the deep tile `(6, 32, 32, 16, 4)` at arity 3
on 6-bit addition and 2-juntas (6 → 4). Rates ×3 from 1 to 100000 (old units; per vote, divide by
the task's outputs), hard accuracy at 500 and 2000 steps on the flat tile, 1000, 2000 and 4000 on
the deep one, three probe seeds; the pins re-verified under the recipe seeds. Probes:
`2026-09-08-plain-descent-{flat,deep,rate-each-signal-wants,recipe-seeds-at-depth,deploy-gap-at-depth}.py`,
`2026-09-09-descent-with-a-readout-at-depth.py`. Device: CPU. A *band* is the range of rates at
which every probe seed reaches 1.000 within the budget; a *point* is a band one grid step wide.

**The flat tile: what each signal buys.** Steps are at the pinned rate, output layer → input layer.

| signal | reaches | band (per vote) | pinned | step out → in | what it says |
|---|---|---|---|---|---|
| soft reference (to the logit) | both tasks, batched and online | junta [1.5, 15000], addition [3, 1000]; online [5, 500] | 50 (`SOFT_FLOOR`, `ONLINE_FLOOR`) | 0.024 → 0.004 | gentle descent; online and batched share the band |
| soft relay to the entry | both | addition [1, 333]; junta [0.5, 50000] (above ~1500 every table saturates at once and it is the bits' relay) | — | — | a decade below the reference on addition: no σ′, a larger vote |
| straight-through (hard, to the logit) | both | addition [100, 233]; junta [0.5, 500] | 150 (`HARD_FLOOR`) | 1.0 → 0.8 | one vote moves an entry by a logit; "the bits want a smaller step" was a narrower band |
| hard relay to the entry | junta only | junta every rate; addition none (0.47–0.60) | — | — | with one hidden layer the σ′-free relay solves a junta and not a carry |

**The deep tile: what each signal buys** (6-bit addition; the 2-junta in the last column).

| signal | reaches | band or best | pinned | step out → in | on the junta |
|---|---|---|---|---|---|
| soft reference | **a point**: 750 on the probe seeds, 500 and 1125 on the recipe seeds, never on all six tiles | within two bits on every seed (≥ 0.98) | 750 (`DEEP_FLOOR`) | 1.0 → 0.02 | band [2.5, 2500] |
| soft relay to the entry | the point 375, on all six tiles; ≥ 500 collapses | 1.000 | 375 (`DEEP_RELAY`) | 2.5 → 0.04 | [0.75, 25000] |
| soft uniform · direct · reachable | short at every rate | 0.95 · 0.80 · 0.94 | — | — | uniform and reachable every rate; direct 2 of 3 seeds |
| straight-through | short, well clear of chance | 0.94 ± 0.01 at 250; 0.90–0.94 on [175, 500]; chance ≤ 75 and ≥ 7500 | 250 (`DEEP_HARD_FLOOR`) | 3.2 → 1.3 | [37, 375] |
| hard relay to the entry | never | chance at every rate | — | 15 → 6 | 2 seeds of 3 at best |
| hard uniform · reachable · flip | short, rate-free above 250 (identical digits to 25000) | 0.86–0.96 · 0.88–0.99 · 0.85–0.91 | 250 | 18 → 15 · 13 → 15 · 3.3 → 2.3 | every seed at 1.000 (flip: 0.97–1.000) |
| hard direct | short | 0.62–0.72 | — | — | [0.75, 25] |

**What the steps say.** On the flat tile the reference descends gently, a fortieth of a logit per
step at the output. Straight-through moves an entry by a logit per vote: a bit fabric's natural
step, and the reason its band is narrow. On the deep tile the reference's point sits where the
output layer moves a whole logit per step while the input layer moves a fiftieth: the vote shrinks
tenfold per layer on the soft pass (1.3e−3 at the output, 2.4e−5 at the input), so one global rate
is a compromise between an input layer that barely moves and an output layer that overshoots. Below
the point the run parks at a soft solution short of the target (soft read 0.90–0.98, gradient a few
percent of its initial size); at the point it reaches the target from the edge of saturation (41 %
of entries unsaturated against 53–67 % a decade below). On the bits every vote is a whole bit and
the wiring-shaped feedback counts paths, so at the pinned rate one vote moves an entry by six to
eighteen logits, past any scale a table has: **every bits number here was measured with a counter
one vote deep, last vote wins**, which is why those cells are rate-free above a threshold. What a
deeper counter does is the first question of the second meta-learning chunk.

**What held, what moved.** Held: one tile computes, batched and online, with no deploy gap; the
relay is autodiff; the blind split stalls short on addition on the soft pass; on the bits the fixed
feedbacks on the reachability and the flip credit train, the random bus worst, the masked bus like
the wiring-shaped split; the flip credit comes to rest; the per-gate signal does not leave chance.
Moved: (1) **the deep floor**: under the plain update the reference reaches the target on most tiles
and never on all, so `test_addition_under_the_deep_floor` (1.000 on every seed under Adam) is
retracted and re-earned as "within two bits on every seed, and the relay to the entry at its own
rate reaches". (2) **Straight-through trains the bits at four layers** (never under Adam), while the
relay to the entry does not; the only difference between the two cells is σ′, which the plain
update keeps as a weight and Adam normalised away, so "the exact relay fails at depth" is now "the
relay without σ′ fails" (a reading, unmeasured: σ′ gives a saturated entry inertia, a "stay"). (3)
"The bits want a smaller step" is a narrower band. (4) The online floor needs 500 steps, not 3000:
the vote is the same size online and batched. (5) The chattering on the bits is the piecewise-
constant gradient's, not Adam's.

**The path.** (1) The brief's grid stopped at 3000 and the flat reference had no upper edge, so
every sweep was extended two decades. (2) At depth the reference's band was a point; the first
reading, a deploy gap the plain push never closes, was probed and fell (the soft read is wrong too);
the reading that the large step saturates before the landscape flattens is unmeasured. (3) The
recipe seeds inverted the point (2000 and 4500 reach, 3000 does not, old units). (4) The review
(2026-09-08) found the note unreadable and the rates a smell; the convention above followed, and the
step per layer was measured: the smell was a tenfold attenuation per layer on the soft pass, which
Adam's per-entry normalisation had hidden. (5) Whether the transport could carry a non-vanishing
magnitude was tried on the wire (a sign carry, a conserving carry, the error per path: the sign carry
gives a band, kept out of the repo) and then at the cell
(`probes/2026-09-09-descent-with-a-readout-at-depth.py`): on the deep tile with the relay to the
entry, plain descent is the point 300 per vote; the **sign of the vote, no state, reaches on every
seed from 0.01 to 0.3 logit per step**; RMSprop (one accumulator) from 0.003 to 0.3; Adam (two)
0.01 to 1; Lion 0.03 to 0.3 with a 0.999; momentum alone never. The attenuation is a magnitude
problem and any magnitude-erasing readout at the cell removes it, so the adjoint frame stays as it
is and normalisation is the rule's business (a hand-engineered family to benchmark and beat, the
next chunk). The plain floor keeps its point at depth, recorded, not repaired.

**Measured and read.** Measured: everything in the two tables and the steps; the claims below on the
recipe seeds. Read: why the large step avoids the soft plateau; σ′ as inertia; that the sign readout's
band survives on other tasks and shapes (one tile, one task, three seeds).

**Claims left behind.** Recipes (per vote): `SOFT_FLOOR` 50/400 · `HARD_FLOOR` 150/3000 ·
`ONLINE_FLOOR` 50/500 · `DEEP_FLOOR` 750/4000 · `DEEP_HARD_FLOOR` 250/2000 · `DEEP_RELAY` 375/4000
(new). Unchanged apart from the recipe they name: the three step-0 claims, the five straight-through
claims, the four direct-feedback claims, two of the transports claims. Re-stated: the relay reaches
where the blind split stalls, under `DEEP_RELAY`. Retracted and re-earned: the deep addition floor,
as above. New (`claims/test_2026_09_08_the_floors_under_plain_descent.py`): the reference within two
bits and the relay to the entry at 1.000; the same signal short under the reference's rate and at the
target under its own; straight-through above 0.8 at depth where the relay to the entry is at chance;
2-bit addition under the hard floor. Nineteen claims, green; 24 s at `-n 8` on this CPU.

**What would change it.** The hand-engineered family at the cell, as named rules with their state
count, replacing the point with a band under a declared cost. More tiles at depth. The σ′-as-inertia
reading measured (flips per step). The mechanism probes of the two 2026-09-07 notes re-run under the
plain update. The deep tile online.
