# 2026-09-07 — Direct feedback and the flip credit: the "stay" votes rescue the exact relay; alignment wants a feedback the wiring can realise

*From round 3 of the step-1 chunk (loom PR #4), under the adjoint frame; the probes of the same
date. The path is recorded, including the bug found on the way.*

**Question.** Chunk 3 read the bits' failure of the exact relay as "flip, never stay", and the
blind split's success as feedback alignment. Two tests of that reading: give the relay its "stay"
votes (the exact credit of one flip, outside the adjoint frame), and give the circuit the cheapest
feedback there is (direct feedback alignment, a random ±1 coefficient per gate and output, no
reverse wiring). Same shape as chunk 3, `(6, 32, 32, 16, 4)` at arity 3 on 6-bit addition.

**Conditions.** `DEEP_HARD_FLOOR` (Adam 0.02, 2000 steps) and `DEEP_FLOOR` with the signal
swapped; the probes also at 0.1.

**A bug on the way.** The first direct-feedback run did not move at all: accuracy identical at 500
and 2000 steps on both passes (0.484–0.516). The feedback matrix was being applied to the output
layer too, whereas in direct feedback alignment an output gate reads its own residual. Fixed
(`feedback` returns the identity for the output layer; a test asserts direct feedback equals the
relay at the output layer). The numbers below are after the fix.

**Measured at initialisation** (`probes/2026-09-07-signal-ladder-by-layer.py`, sign agreement
with the reference per layer, input → output): direct feedback 0.48 / 0.46 / 0.55 / 1.00 on the
soft pass and 0.49 / 0.48 / 0.47 / 0.54 on the bits; the flip credit 0.47 / 0.48 / 0.46 / 0.44 on
the soft pass and 0.55 / 0.54 / 0.57 / 0.52 on the bits. Neither resembles the reference anywhere
but the output layer, like every blind transport.

**Measured in training** (`probes/2026-09-07-descent-by-transport.py`; hard accuracy at 500 /
2000 steps, seeds 0–2):

| signal | rate | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| soft.uniform.entry | 0.1 | 0.922 / 0.934 | 0.938 / 0.941 | 0.863 / 0.895 |
| soft.direct.entry | 0.1 | 0.855 / 0.828 | 0.773 / 0.742 | 0.742 / 0.770 |
| soft.direct.entry | 0.02 | 0.820 / 0.746 | 0.805 / 0.770 | 0.699 / 0.773 |
| soft.flip.entry | 0.1 | 0.582 / 0.590 | 0.617 / 0.621 | 0.609 / 0.617 |
| hard.relay.entry | 0.02 | 0.484 / 0.500 | 0.492 / 0.512 | 0.535 / 0.539 |
| hard.uniform.entry | 0.02 | 0.867 / 0.938 | 0.820 / 0.926 | 0.812 / 0.844 |
| hard.direct.entry | 0.1 | 0.715 / 0.629 | 0.660 / 0.730 | 0.551 / 0.680 |
| hard.direct.entry | 0.02 | 0.656 / 0.633 | 0.707 / 0.688 | 0.500 / 0.652 |
| hard.flip.entry | 0.1 | 0.941 / 0.941 | 0.859 / 0.859 | 0.895 / 0.895 |
| hard.flip.entry | 0.02 | 0.883 / 0.883 | 0.902 / 0.902 | 0.922 / 0.922 |

**The flip credit trains the bits.** Against expectation. Chunk 3's note had read the relay's
failure as the absence of "stay" votes and expected that restoring them would not be enough,
because the credit still changes with every bit around it. It is enough: the relay plus the cost
of the outputs a flip would break reaches 0.86–0.94, level with the blind split. And it stops:
accuracy is identical at 500 and 2000 steps. `probes/2026-09-07-flip-fixed-point.py`: after
training, zero entries have a credit pointing toward a flip and zero bits flip per step over the
last 200, on every probe seed (under the recipe's seeds, zero on two and three entries out of 672
on the third, still settling at 2000 steps); under the blind split 60–135 entries are still pushed
and 2–5 bits flip per step. The flip credit is greedy coordinate descent on the hard loss, one entry at a time, and
it lands on a local optimum of single flips at 0.88–0.92, where no single flip helps. On the soft
pass it is meaningless (0.55–0.62): the cost of a flip is a bits quantity.

**Direct feedback trains, worst of the blind transports.** 0.63–0.73 on the bits against the
wiring-shaped split's 0.84–0.94 and the relay's 0.50; 0.74–0.83 on the soft pass against
0.86–0.94. `probes/2026-09-07-alignment-to-any-feedback.py` measures the alignment of the bits'
actual Jacobian ∂r_o/∂r_g with the feedback given, per hidden layer: under the wiring-shaped split
it rises from 0.5 to 0.81 / 0.89 / 0.96 at 1000 steps and holds; under random direct feedback it
rises to 0.71 / 0.69 / 0.84 at 1000 steps and falls back to 0.57–0.61 by 2000 with the accuracy;
under the relay it never leaves 0.5. The reading: alignment needs a feedback the forward circuit
can realise. The wiring-shaped feedback asks every gate for the same thing, be monotone in your
inputs, which each gate can do on its own and which composes across layers; random signs per gate
and output ask a gate to affect two outputs through one shared path with opposite signs, and to
respond to outputs it cannot reach, which no table can do. The circuit aligns as far as it can
and no further.

**Measured and read.** Measured: the flip credit trains the bits to a fixed point of single flips;
the blind transports order as wiring-shaped, random, none on every seed; the alignment to the
feedback given rises fully under the wiring-shaped feedback, partly and unstably under random
signs, never under the relay. Read, not yet tested: that realisability by the forward circuit is
what separates the two feedbacks. A feedback that is random in sign but restricted to the outputs
each gate can reach, and consistent along shared paths, would test it.

**Claims left behind.** `claims/test_2026_09_07_direct_feedback_and_the_flip_credit.py`: the flip
credit comes to rest where the blind split keeps pushing (ten times fewer entries still pushed),
above 0.8; on the bits the blind
transports order as wiring-shaped, random, none, with the relay at chance.

**What would change it.** Escaping the flip credit's local optimum: a temperature, or pairs of
flips. A reachability-restricted random feedback. The second substrate, where the soft pass is
what the fabric computes and the whole bits problem may not arise. And the rule of step 3, which
receives λ and could learn to use a blind feedback better than descent does.

**For the map.** On a bits fabric there are now two signals a chip could run that train at depth:
the wiring-shaped feedback (a reverse channel per wire, no table read, aligns) and the flip credit
(a reverse channel per wire carrying two numbers, a table read per input, exact for one flip,
stops at a local optimum). The cheapest wiring, a bus, trains worst. The exact relay, the one
thing autodiff would suggest, is the one that does not train at all.
