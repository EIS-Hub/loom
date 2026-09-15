# 2026-09-09 — A readout at the cell makes the deep floor a band: the hand-engineered family, and what the learned rule must beat

*From the step-2 family chunk and the probe of the same date; the path runs through the previous
evening's wire-side attempt, kept out of the repo. Re-run on 2026-09-15 after round 1 of the
review: the first table came from a probe that re-implemented the six rules and seeded its own
tiles, so its numbers belonged to nothing that runs again; the tables below are the same probe run
through `recipes.trace` on the recipe seeds, and the claim's six recipes are six cells of them.*

**Question.** Under plain descent the deep floor is a point in the rate: the vote shrinks tenfold
per layer on the soft pass, and one rate cannot serve the output layer and the input layer at once
(`findings/2026-09-08-the-floors-under-plain-descent/note.md`). Where is that fixed, on the wire or
in the cell, at what cost, and what does the family of hand-engineered rules look like once it is
written as transformations of the signal?

**Floor and shape.** The deep tile `(6, 32, 32, 16, 4)` at arity 3 on 6-bit addition, the soft
relay to the entry (the signal a soft fabric could carry, no σ′); `DEEP_RELAY`, the plain rule at
375 per vote, reaches the target on every recipe seed and is the point.

**Conditions.** Six rules on the same signal, each a few lines (`rule.py`): plain and the sign of
the vote (no state); momentum, RMSprop and Lion (one accumulator per entry); Adam (two). Every
cell is `DEEP_RELAY._replace(rule, lr)` run by `recipes.trace` on the recipe seeds 0, 1, 2, hard
accuracy at the recipe's budget of 4000 steps (`2026-09-09-descent-with-a-readout-at-depth.py`,
twenty seconds on twelve CPU workers). Two rate grids of six points: per vote for the rules that
keep the vote's size (plain, momentum), the recipe's 375 times 1/30, 1/10, 1/3, 1, 3 and 10; per
step for the rules that erase it, 0.003 to 1 by factors of about three. The claim is six of these
cells, two named rates a decade apart per rule.

**Measured** (hard accuracy at 4000 steps, recipe seeds 0, 1, 2, bold the mean; ★ every seed at
1.000; the per-seed cells and the accuracy at 1000 steps are in the appendix):

| rate (per vote) | plain (0) | momentum (1) | | rate (per step) | sign (0) | RMSprop (1) | Lion (1) | Adam (2) |
|---|---|---|---|---|---|---|---|---|
| 12.5 | **0.970** | **0.991** | | 0.003 | **0.990** | **0.979** | **0.980** | **0.993** |
| 37.5 | **0.971** | **0.996** | | 0.01 | ★ **1.000** | ★ **1.000** | **0.999** | ★ **1.000** |
| 125 | **0.996** | **0.895** | | 0.03 | ★ **1.000** | ★ **1.000** | **0.995** | **0.997** |
| 375 | ★ **1.000** | **0.482** | | 0.1 | ★ **1.000** | ★ **1.000** | **0.999** | ★ **1.000** |
| 1125 | **0.533** | **0.477** | | 0.3 | ★ **1.000** | ★ **1.000** | ★ **1.000** | ★ **1.000** |
| 3750 | **0.520** | **0.520** | | 1 | **0.977** | **0.997** | **0.996** | **0.996** |

Plain is the point 375 (a step of about a logit at the output layer and a fiftieth at the input):
a third of it leaves one seed at 0.988, three times it collapses to chance. Momentum, linear in the
magnitude, never reaches on every seed and collapses a decade before plain does. Every rule that
erases or normalises the magnitude has a band: the sign of the vote, with no state, from 0.01 to
0.3 logit per step; RMSprop the same band; Adam at 0.01, 0.1 and 0.3 with one chattering cell at
0.03 (a seed at 0.992); Lion on every seed at 0.3 only, and between 0.995 and 0.999 elsewhere. The
attenuation is a magnitude problem, and a readout at the cell removes it at zero state.

**The path.**

1. *On the wire first.* The evening before, the same fix was tried inside the adjoint: a gate
   passing its error back with the sign of its sensitivity, or with its sensitivities normalised to
   unit mass, or the error divided by the reach. The sign carry gave a band (every seed at 1.000
   from 25 to 75 per vote, in these units) and the votes came out flat across layers; the
   conserving carry flattened the votes and fell just short; the error per path was untested at
   low rates. That would have been a fourth coordinate of the signal, the carry's scale.
2. *Then at the cell.* Gabriel asked whether a family of transformations at the cell made the
   wire-side family unnecessary. The probe above says yes: the stateless sign readout at the entry
   buys what the sign carry bought, with no change to the frame. The wire-side scales are not
   built; step 1's frame stays as landed; normalisation is the rule's business.
3. *What each costs.* The table in `docs/rule.md`: state per entry, arithmetic beyond multiply-add.
   The zero-state and one-accumulator members are multiply-add and compare; only the second moment
   needs a root and a division. Adafactor's sharing (one accumulator per gate and one per address
   instead of one per entry) is the cheap normaliser to try when the ledger bites.
4. *The floor stays plain.* A fixed step is the condition under which the smallest learnable rule,
   one η, can be diagnosed; the family enters claims only as the baselines a learned rule is
   measured against, at equal memory. Adam is one of them: hand-engineered, cheaper than any learned
   function of a few thousand parameters, to be beaten too.
5. *Where the numbers live (round 1 of the review, 2026-09-15).* The first table was the printed
   output of a probe with six inline rules and its own seeds, on a grid of 10 to 3000 per vote: the
   plain point at 300, RMSprop's band from 0.003, Adam's to 1, Lion's at 0.03 and 0.3. Re-run
   through the recipes on the recipe seeds nothing qualitative moved except Lion, whose band is one
   point here; those readings are replaced above, and the claim's cells are now cells of this table.

**Measured and read.** Measured: the six rules on one tile, one task, one signal, the three
recipe seeds, through the recipes. Read: that the band survives on other shapes, tasks and
signals; that a learned rule can find these points (step 3's question); that the one-vote-deep
counter of the bits numbers interacts with a readout (the second meta-learning chunk).

**The claim** (`test_claim.py`, re-run on every push): six cells of the table above, named as
recipes. The plain rule reaches on every seed at its rate and not a decade below: `DEEP_RELAY`
(375, ★) and `DEEP_RELAY_TENTH` (37.5, a seed at 0.934). The sign of the vote reaches at two rates a
decade apart: `DEEP_SIGN` (0.1, ★) and `DEEP_SIGN_TENTH` (0.01, ★). RMSprop likewise: `DEEP_RMSPROP`
(0.1, ★) and `DEEP_RMSPROP_TENTH` (0.01, ★). A band is claimed by two points, never by a sweep
inside a claim; the sweep is this note's.

**What would change it.** The family on the bits signals, where every vote is already a whole bit
and the sign readout is a no-op. Adafactor's sharing measured against the per-entry state. The
learned η of meta-learning I on a rule other than plain.

## Appendix: the cells

Every cell of the table, per seed, with the accuracy at 1000 steps beside it (the probe's second
table, verbatim):

| rule | rate | seed 0 · 1 · 2 at the budget | mean ± sd | at 1000 |
|---|---|---|---|---|
| plain | 12.5 | 0.988 · 0.988 · 0.934 | **0.970 ± 0.026** | 0.99 0.95 0.87 |
| plain | 37.5 | 0.992 · 0.988 · 0.934 | **0.971 ± 0.027** | 0.99 0.99 0.93 |
| plain | 125 | 1.000 · 1.000 · 0.988 | **0.996 ± 0.006** | 1.00 1.00 0.99 |
| plain | 375 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 0.65 1.00 1.00 |
| plain | 1125 | 0.477 · 0.562 · 0.559 | **0.533 ± 0.040** | 0.55 0.54 0.56 |
| plain | 3750 | 0.496 · 0.523 · 0.539 | **0.520 ± 0.018** | 0.53 0.55 0.50 |
| sign | 0.003 | 1.000 · 1.000 · 0.969 | **0.990 ± 0.015** | 0.93 0.96 0.86 |
| sign | 0.01 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 0.95 |
| sign | 0.03 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 1.00 |
| sign | 0.1 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 1.00 |
| sign | 0.3 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 1.00 |
| sign | 1 | 0.938 · 0.992 · 1.000 | **0.977 ± 0.028** | 0.94 0.99 1.00 |
| momentum | 12.5 | 0.996 · 0.996 · 0.980 | **0.991 ± 0.007** | 1.00 1.00 0.96 |
| momentum | 37.5 | 1.000 · 0.996 · 0.992 | **0.996 ± 0.003** | 1.00 1.00 0.99 |
| momentum | 125 | 1.000 · 0.996 · 0.688 | **0.895 ± 0.146** | 1.00 1.00 0.66 |
| momentum | 375 | 0.461 · 0.516 · 0.469 | **0.482 ± 0.024** | 0.48 0.52 0.54 |
| momentum | 1125 | 0.461 · 0.453 · 0.516 | **0.477 ± 0.028** | 0.48 0.50 0.47 |
| momentum | 3750 | 0.527 · 0.496 · 0.535 | **0.520 ± 0.017** | 0.53 0.47 0.49 |
| rmsprop | 0.003 | 1.000 · 1.000 · 0.938 | **0.979 ± 0.029** | 0.95 0.93 0.86 |
| rmsprop | 0.01 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 0.98 |
| rmsprop | 0.03 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 1.00 |
| rmsprop | 0.1 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 1.00 |
| rmsprop | 0.3 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 1.00 |
| rmsprop | 1 | 1.000 · 0.992 · 1.000 | **0.997 ± 0.004** | 1.00 0.99 1.00 |
| lion | 0.003 | 1.000 · 0.996 · 0.945 | **0.980 ± 0.025** | 0.92 0.96 0.84 |
| lion | 0.01 | 1.000 · 1.000 · 0.996 | **0.999 ± 0.002** | 1.00 0.98 0.94 |
| lion | 0.03 | 1.000 · 0.996 · 0.988 | **0.995 ± 0.005** | 1.00 0.99 0.96 |
| lion | 0.1 | 1.000 · 1.000 · 0.996 | **0.999 ± 0.002** | 1.00 1.00 1.00 |
| lion | 0.3 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 1.00 |
| lion | 1 | 1.000 · 1.000 · 0.988 | **0.996 ± 0.006** | 1.00 1.00 0.99 |
| adam | 0.003 | 1.000 · 1.000 · 0.980 | **0.993 ± 0.009** | 1.00 0.96 0.93 |
| adam | 0.01 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 0.98 |
| adam | 0.03 | 1.000 · 1.000 · 0.992 | **0.997 ± 0.004** | 1.00 1.00 0.99 |
| adam | 0.1 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 1.00 |
| adam | 0.3 | 1.000 · 1.000 · 1.000 | **1.000 ± 0.000** | 1.00 1.00 1.00 |
| adam | 1 | 1.000 · 0.996 · 0.992 | **0.996 ± 0.003** | 1.00 1.00 0.99 |
