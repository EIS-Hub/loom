# 2026-09-07 — Straight-through is a cell of the signal taxonomy, not its own thing

*From the step-1 chunk (loom PR #3) and its review; the two probes of the same date.*

**Question.** Is the gradient computed on the deployed bits (autodiff through straight-through
tables) the same signal as the reference gradient on the soft pass? Does descent on it reach the
target, and with what deploy gap?

**Floor.** The reference, `SOFT_FLOOR`: hard accuracy 1.000 on 2-juntas for three seeds (the
previous note).

**Conditions.** `SOFT_FLOOR`; `BITS_FLOOR` (the same autodiff on the bits, Adam at 0.02 for 2000
steps); and, in the sweep, unnamed variants of both over rate × budget.

**Measured, at initialisation** (`probes/2026-09-07-soft-vs-bits-gradient.py`, random tiles,
shape `(4, 16, 8, 2)`, 2-juntas): the two gradients are nearly orthogonal and the one on the bits
touches a fraction of the logits; where both are nonzero they agree in sign a little more than half
the time.

| tile | nonzero, soft | nonzero, bits | cosine | sign agreement where both nonzero |
|---|---|---|---|---|
| 0 | 0.67 | 0.14 | +0.13 | 0.70 |
| 1 | 0.68 | 0.26 | +0.00 | 0.59 |
| 2 | 0.69 | 0.25 | +0.17 | 0.56 |

**Measured, in training** (`probes/2026-09-07-straight-through-sweep.py`): on the soft pass every
condition reaches 1.000. On the bits at the soft rate (0.1), hard accuracy after 500 steps is
0.594, 0.719 and 1.000 for the three seeds, and after 2000 steps 0.688, 1.000 and 1.000: two of
three seeds miss at the short budget, one at the long, and the first seed *worsens* with more
steps. At a fifth of the rate (0.02) for 2000 steps all three reach 1.000. The deploy gap
(`recipes.trace`): identically zero at every recorded step on the bits; on the soft pass it opens
during training and closes at the end.

**Why.** Straight-through replaces only the derivative of the rounding step; the residual, the
address distribution and the paths back through other gates are evaluated on the bits that flowed
(`docs/signals.md`). Factored, the bits gradient is the error relayed through the rounded tables to
a gate's output, times the one-hot of the selected address, times σ′(logit): blastema's
"relay × basis × surrogate" with the relay on the deployed pass. It is therefore the cell
(`hard`, `autodiff`, surrogate kept) of a taxonomy whose coordinates are the pass, the transport and
the surrogate factor, and not a signal of its own kind. Being piecewise constant, it changes sign
when a table entry crosses one half; with Adam's normalised steps at the soft rate a logit marches
to the threshold, flips, and marches back, which is the chattering above.

**Also found.** `hard + soft − stop_gradient(soft)` is not exactly `hard` in float32; the zero must
be parenthesised. A mechanistic test now asserts the equality.

**Claims left behind.** `claims/test_2026_09_07_straight_through.py`: the junta under the bits
floor; no deploy gap at any step on the bits; the gap opens then closes on the soft pass; every
matrix cell reaches its target under its recipe.

**What would change it.** A shape where depth is forced (does the chattering grow with depth?);
the relay on the bits, which should be sign-identical to this gradient at every logit (a theorem to
check numerically, next chunk); plain −η·signal instead of Adam (step 2), which separates the
signal's coarseness from Adam's normalisation.
