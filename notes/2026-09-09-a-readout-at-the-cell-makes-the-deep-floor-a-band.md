# 2026-09-09 — A readout at the cell makes the deep floor a band: the hand-engineered family, and what the learned rule must beat

*From the step-2 family chunk and the probe of the same date; the path runs through the previous
evening's wire-side attempt, kept out of the repo.*

**Question.** Under plain descent the deep floor is a point in the rate: the vote shrinks tenfold
per layer on the soft pass, and one rate cannot serve the output layer and the input layer at once
(`notes/2026-09-08-the-floors-under-plain-descent.md`). Where is that fixed, on the wire or in the
cell, at what cost, and what does the family of hand-engineered rules look like once it is written
as transformations of the signal?

**Floor and shape.** The deep tile `(6, 32, 32, 16, 4)` at arity 3 on 6-bit addition, the soft
relay to the entry (the signal a soft fabric could carry, no σ′); `DEEP_RELAY`, the plain rule at
375 per vote, reaches the target on every recipe seed and is the point.

**Conditions.** Six rules on the same signal, each a few lines (`rule.py`): plain and the sign of
the vote (no state); momentum, RMSprop and Lion (one accumulator per entry); Adam (two). Rates ×3
on two grids, per vote for the rules that keep the vote's size (plain, momentum) and per step for
those that erase it; hard accuracy at 1000, 2000 and 4000 steps; three probe seeds
(`probes/2026-09-09-descent-with-a-readout-at-depth.py`). The claims under the recipe seeds, at
two named rates a decade apart per rule.

**Measured** (hard accuracy at 4000 steps, three probe seeds, bold the mean; ★ every seed at 1.000):

| rate | plain (0) | momentum (1) | | rate | sign (0) | RMSprop (1) | Lion (1) | Adam (2) |
|---|---|---|---|---|---|---|---|---|
| 10 | **0.936** | **0.973** | | 0.003 | **0.996** | ★ **1.000** | **0.975** | **0.993** |
| 30 | **0.970** | **0.947** | | 0.01 | ★ **1.000** | ★ **1.000** | **0.977** | ★ **1.000** |
| 100 | **0.992** | **0.965** | | 0.03 | ★ **1.000** | ★ **1.000** | ★ **1.000** | **0.997** |
| 300 | ★ **1.000** | **0.964** | | 0.1 | ★ **1.000** | ★ **1.000** | **0.999** | ★ **1.000** |
| 1000 | **0.458** | **0.979** | | 0.3 | ★ **1.000** | ★ **1.000** | ★ **1.000** | ★ **1.000** |
| 3000 | **0.501** | **0.492** | | 1 | **0.975** | **0.999** | **0.995** | ★ **1.000** |

Plain is the point 300 (a step of about a logit at the output layer and a fiftieth at the input);
momentum, linear in the magnitude, never reaches on every seed. Every rule that erases or
normalises the magnitude has a band: the sign of the vote, with no state, from 0.01 to 0.3 logit
per step; RMSprop from 0.003 to 0.3; Adam from 0.01 to 1 with one chattering cell; Lion 0.03 and
0.3 with 0.999 between. The attenuation is a magnitude problem, and a readout at the cell removes
it at zero state.

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

**Measured and read.** Measured: the six rules on one tile, one task, one signal, three seeds; the
claims on the recipe seeds. Read: that the band survives on other shapes, tasks and signals; that a
learned rule can find these points (step 3's question); that the one-vote-deep counter of the bits
numbers interacts with a readout (the second meta-learning chunk).

**Claims left behind.** `claims/test_2026_09_09_a_readout_at_the_cell_makes_the_deep_floor_a_band.py`:
the plain rule reaches at `DEEP_RELAY` and not at a tenth of its rate; the sign of the vote and
RMSprop reach at two named rates a decade apart.

**What would change it.** The family on the bits signals, where every vote is already a whole bit
and the sign readout is a no-op. Adafactor's sharing measured against the per-entry state. The
learned η of meta-learning I on a rule other than plain.
