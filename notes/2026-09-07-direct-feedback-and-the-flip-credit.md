# 2026-09-07 — Direct feedback and the flip credit: the "stay" votes rescue the exact relay; a fixed feedback trains if it lives on the wiring's reachability

*From round 3 of the step-1 chunk (loom PR #4), under the adjoint frame; the probes of the same
date; audited adversarially the same evening and corrected on 2026-09-08 (the audit's experiments
are the `probes/2026-09-08-audit-*.py` files). The path is recorded, including a bug and two
readings that fell.*

**Question.** Chunk 3 read the bits' failure of the exact relay as "flip, never stay", and the
blind split's success as feedback alignment. Two tests of that reading: give the relay its "stay"
votes (the exact credit of one flip, outside the adjoint frame), and give the circuit the cheapest
feedback there is (direct feedback alignment, a random ±1 coefficient per gate and output, no
reverse wiring). Same shape as chunk 3, `(6, 32, 32, 16, 4)` at arity 3 on 6-bit addition.

**Conditions.** `DEEP_HARD_FLOOR` (Adam 0.02, 2000 steps) and `DEEP_FLOOR` with the signal
swapped; the probes also at 0.1. The tables of this note and the previous one were produced on a
GPU before the probes were pinned to the CPU; the audit reran them on the CPU and every cell
reproduces except the chattering ones, which move at the third decimal (the exact relay at 0.1,
and `soft.uniform.entry` seed 2, recorded as 0.859 in the first note, 0.895 here, 0.910 on the CPU).

**A bug on the way.** The first direct-feedback run did not move at all: accuracy identical at 500
and 2000 steps on both passes (0.484–0.516). The feedback matrix was being applied to the output
layer too, whereas in direct feedback alignment an output gate reads its own residual. Fixed
(`random_signs`, then called `feedback`, returns the identity for the output layer; a test asserts direct feedback equals the
relay at the output layer). The numbers below are after the fix.

**Measured at initialisation** (`probes/2026-09-07-signal-ladder-by-layer.py`, sign agreement
with the reference per layer, input → output): direct feedback 0.48 / 0.46 / 0.55 / 1.00 on the
soft pass and 0.49 / 0.48 / 0.47 / 0.54 on the bits; the flip credit 0.55 / 0.54 / 0.57 / 0.52 on
the bits. Neither resembles the reference anywhere but the output layer, like every blind
transport.

**Measured in training** (`probes/2026-09-07-descent-by-transport.py`; hard accuracy at 500 /
2000 steps, seeds 0–2):

| signal | rate | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| soft.uniform.entry | 0.1 | 0.922 / 0.934 | 0.938 / 0.941 | 0.863 / 0.895 |
| soft.direct.entry | 0.1 | 0.855 / 0.828 | 0.773 / 0.742 | 0.742 / 0.770 |
| soft.direct.entry | 0.02 | 0.820 / 0.746 | 0.805 / 0.770 | 0.699 / 0.773 |
| hard.relay.entry | 0.02 | 0.484 / 0.500 | 0.492 / 0.512 | 0.535 / 0.539 |
| hard.uniform.entry | 0.02 | 0.867 / 0.938 | 0.820 / 0.926 | 0.812 / 0.844 |
| hard.direct.entry | 0.1 | 0.715 / 0.629 | 0.660 / 0.730 | 0.551 / 0.680 |
| hard.direct.entry | 0.02 | 0.656 / 0.633 | 0.707 / 0.688 | 0.500 / 0.652 |
| hard.flip.entry | 0.1 | 0.941 / 0.941 | 0.859 / 0.859 | 0.895 / 0.895 |
| hard.flip.entry | 0.02 | 0.883 / 0.883 | 0.902 / 0.902 | 0.922 / 0.922 |

The flip credit on the soft pass is not a signal (0.55–0.62): the cost of a flip is a bits
quantity, and the soft cells were removed from `CELLS`.

**The flip credit trains the bits.** Against expectation. Chunk 3's note had read the relay's
failure as the absence of "stay" votes and expected that restoring them would not be enough,
because the credit still changes with every bit around it. It is enough: the relay plus the cost
of the outputs a flip would break reaches 0.86–0.94, level with the blind split. And it stops:
accuracy is identical at 500 and 2000 steps. `probes/2026-09-07-flip-fixed-point.py`: after
training, zero entries have a credit pointing toward a flip and zero bits flip per step over the
last 200, on every probe seed (under the recipe's seeds, zero on two and three entries out of 672
on the third, still settling at 2000 steps); under the blind split 60–135 entries are still pushed
and 2–5 bits flip per step. It rests at 0.88–0.92.

*Corrected by the audit.* Two things this note first said about it were wrong. (1) The reach is
not "the number of outputs a flip changes" but the number of live paths to them
(`probes/2026-09-08-audit-reach-and-exactness.py`): equal to the outputs changed for 90–93 % of
(case, line) pairs at the input layer and 96–100 % above it, larger where paths reconverge, up to
5 with four outputs. So the credit is exact for one flip in the upper layers (88–100 % of entries,
sign 95–100 %) and inexact at the input layer (67–78 %, sign 88–98 %). At the end state an
exhaustive scan of all 672 single flips finds 0 / 1 / 0 that would still improve the loss, and an
oracle continuation gains 0.004: the fixed point holds in substance. The corpus audit of the
same day (Codex, read 2026-09-08) supplied the mechanism in the other direction: where a line
branches into two paths that reconverge, the credit can *under*-estimate as well, since each path
alone can have zero sensitivity while flipping the line moves both; its counterexample runs
against the code in `probes/2026-09-08-flip-credit-counterexample.py` (predicted 0, actual −½;
and with both paths live, predicted −1, actual −½). (2) It is not "greedy
coordinate descent one entry at a time": the audit's true greedy single-flip descent on the exact
loss change stops lower (0.852 / 0.875) than the credit under Adam on the same tiles
(0.883 / 0.902), and greedy descent on the credit itself cycles. Adam's many simultaneous early
flips are part of why it lands higher.

*The cost term is what does it* (`probes/2026-09-08-audit-flip-credit-controls.py`, recipe seeds):
the relay alone 0.48–0.51; the relay plus a constant "stay" bias of 0.25 to 2 per case 0.47–0.73;
plus the blind reach (path counts, no sensitivity) 0.64–0.72; plus the local case count
0.71–0.80; the uniform split plus the reach 0.68–0.81; the flip credit 0.867–0.887. No
regulariser-shaped substitute matches it.

**Direct feedback trains, worst of the blind transports, and the reason is reachability alone.**
0.63–0.77 on the bits over three random draws against the wiring-shaped split's 0.84–0.94 and the
relay's 0.50; 0.74–0.83 on the soft pass against 0.86–0.94. This note first read the failure as
random signs asking a gate to affect two outputs through one shared path with opposite signs, and
credited the wiring-shaped split with asking every gate to "be monotone". The audit refuted both
(`probes/2026-09-08-audit-feedback-support.py`, bits, three seeds):

| fixed feedback | seeds 0 · 1 · 2 |
|---|---|
| uniform, the path counts | 0.938 · 0.926 · 0.844 |
| direct, random ±1, three draws | 0.63–0.77, every draw below uniform |
| direct, random ±1, **masked to the outputs each gate can reach**, two draws | 0.953 · 0.863 · 0.879 and 0.945 · 0.816 · 0.852 |
| reachability alone, all +1 | 0.914 · 0.898 · 0.859 |
| path counts with a random sign per gate and output | 0.914 · 0.977 · 0.879 |
| the layered adjoint with a carry of −1 on every edge | 0.902 · 0.840 · 0.836 |
| the layered adjoint with a random fixed sign per edge, two draws | 0.840 · 0.949 · 0.816 and 0.961 · 0.883 · 0.812 |

The mask alone closes the gap; random signs on the path counts train as well as the counts; a
negative or a random carry trains as well as +1, and the gates drift to whatever sign they are
given (positive fraction of sensitivities toward the given sign 0.68–1.0 by step 2000). Over
eighteen recipe-seed runs the audit measured plain direct feedback at 0.686 on average, masked at
0.873, the uniform split at 0.861. The correct reading: **alignment needs a fixed feedback whose
support is the wiring's reachability; its signs can be anything fixed.** Random signs per gate
and output are fine; feedback from outputs a gate cannot reach is noise it must cancel, and it
cannot. The masked bus is now the cell `reachable`, costing a gate one bit per output. One masked
run on a probe seed reached 1.000 on the bits, the only bits signal to reach the target at depth
in this whole set; none of eighteen recipe-seed runs did.

**What "alignment" measures.** The alignment probes score the circuit's actual Jacobian against
the feedback given; that quantity rises and holds (0.81 / 0.89 / 0.96 by 1000 steps under the
wiring-shaped split). Feedback alignment's own quantity, the delivered hidden signal against the
true adjoint (`probes/2026-09-08-audit-alignment-cause.py`), rises to 0.59–0.91 mid-training and
falls back to 0.25–0.64 by 2000 steps while accuracy keeps rising: the delivered signal aligns on
the way and de-aligns on the residual errors, which is what a ceiling short of the target looks
like. The same probe settles cause against side effect: with the output layer always exact,
hidden layers frozen give 0.52–0.61, driven by noise 0.56–0.60, by noise on the signal's support
0.54–0.59, by the signal's magnitudes with shuffled signs 0.52–0.61, by the signal 0.84–0.94; a
hidden stack initialised monotone with output-only learning gives 0.54–0.62. Neither random drift
nor monotonicity substitutes for the hidden signal, and "monotone" was an artefact of the carry
being +1.

**Where the relay stops** (`probes/2026-09-08-audit-depth-boundary.py`). The majority class of
6-bit addition is 0.516. With one hidden layer of 32 gates the exact relay is the *better* bits
signal (0.63–0.81 against the blind split's 0.56–0.67); with two hidden layers it has already
failed (0.45–0.56 against 0.72–0.76); at four layers 0.50–0.54 at Adam 0.005 and 0.001 for 4000
steps, sign-SGD for 5000, and init scales 0.3, 3 and 10, while the blind split reaches 0.73–0.97
at every scale. On 2- and 3-junta tasks of the same shape the relay is at 0.47–0.61 and the blind
split reaches 1.000 on every run, the flip credit 0.92–1.00. "A signal for shallow circuits"
means one hidden layer.

**Also found (2026-09-08).** The per-gate signal, the adjoint variable before the readout, is now
exposed (`signals.gate_errors`) and has its own cell, `to="gate"`: the gate's summed error broadcast to
every entry, address-blind. Under descent it does not leave chance, as it cannot: a table whose
entries all move together learns a bias (`probes/2026-09-08-per-gate-descent.py`; soft.relay.gate
0.48–0.53 where soft.relay.entry reaches 1.000, hard.uniform.gate 0.49–0.56 where the entry cell
reaches 0.84–0.94, hard.reachable.gate 0.50–0.52). Its use is downstream: it is what a wire or a
bus carries, and a rule reading it with the gate's own inputs must reconstruct the address, the
step-3 benchmark.

**Measured and read.** Measured, and audited: the flip credit trains the bits to a fixed point
of single flips and no substitute cost does; the blind transports order as wiring-shaped, random,
none on eight seeds; masking the random bus to reachable outputs lifts it to the wiring-shaped
split's level; the signs of a fixed feedback do not matter; the hidden updates carry the
learning; the exact relay never trains past one hidden layer. Read, not tested: what the
wiring-shaped ceiling (0.84–0.98, the target once in twenty-one masked runs) is made of.

**Claims left behind.** `claims/test_2026_09_07_direct_feedback_and_the_flip_credit.py`: the flip
credit comes to rest where the blind split keeps pushing (ten times fewer entries still pushed),
above 0.8; on the bits the blind transports order as wiring-shaped, random, none, with the relay
within 0.1 of chance; restricting the bus to reachable outputs lifts it above the unmasked bus and
above 0.75.

**What would change it.** Escaping the flip credit's local optimum: a temperature, or pairs of
flips. The comparators the corpus audit names and this note lacked: differentiable weightless
networks (DWN), whose LUT gradients are finite differences of table entries like our sensitivity,
BOLD, and the recent depth results on straight-through, against which "one hidden layer" was
measured here on one shape and one task. What reaches 1.000 reliably on the bits, if anything fixed does. The second substrate,
where the soft pass is what the fabric computes and the whole bits problem may not arise. And the
rule of step 3, which receives λ and could learn to use a blind, stable feedback better than
descent does.

**For the map.** On a bits fabric there are now three signals a chip could run that train at
depth: any fixed feedback on the wiring's reachability, delivered by a reverse channel per wire
(the wiring-shaped split) or by a bus with one reachability bit per output (`reachable`); and the
flip credit, a reverse channel per wire carrying two numbers and a table read per input, exact
for one flip where paths do not reconverge, resting at a local optimum. The exact relay, the one
thing autodiff would suggest, trains with one hidden layer and not with two.
