# 2026-09-07 — The relay is autodiff; on the bits at depth the exact relay fails and the blind split learns by feedback alignment

*From the step-1 chunk Signals II (loom PR #4) and the probes of the same date. This note records
the path, not only where it ended: the shape that failed, the hypotheses that fell, the one that held.*

**Question.** Three transports carry the residual to the logits: autodiff through everything, the
relay through each gate's own sensitivity (a local message), the uniform split blind to it. On a
shape where depth is forced, which keeps the reference gradient's sign, which can descent follow,
on each pass, with and without σ′?

**The floor, and how the shape was chosen.** The reference must reach the target before anything
is scored against it. The first deep shape tried, `(6, 24, 12, 4)` at arity 3 on 6-bit addition,
left the reference at 0.93–0.98 hard accuracy: not a floor. `probes/2026-09-07-deep-shape-search.py`
swept widths and arity (hard accuracy at 1000 / 2000 steps, Adam 0.1, three seeds):

| widths | arity 3 | arity 4 |
|---|---|---|
| (6, 24, 12, 4) | 0.984 · 0.930 · 0.969 | 1.000 · 0.992 · 1.000 |
| (6, 32, 16, 4) | 0.988 · 0.961 · 1.000 | 1.000 · 1.000 · 1.000 |
| (6, 48, 24, 4) | 0.973 · 0.977 · 0.969 | 1.000 · 1.000 · 1.000 |
| (6, 32, 32, 16, 4) | 1.000 · 1.000 · 1.000 | 1.000 · 1.000 · 1.000 |
| (6, 48, 48, 24, 4) | 1.000 · 1.000 · 1.000 | 1.000 · 1.000 · 1.000 |

Chosen: `(6, 32, 32, 16, 4)` at arity 3, the deepest that reaches 1.000 on every seed within 1000
steps, four hops for the error to travel; `DEEP_FLOOR`. The transports had been run on the first
shape too and every finding below already held there (on the bits, uniform 0.77–0.91 against relay
0.58–0.76).

**Conditions.** `DEEP_FLOOR` (the reference, Adam 0.1, 4000 steps) and `DEEP_HARD_FLOOR` (the
bits, Adam 0.02, 2000 steps); the claims swap the recipe's signal and nothing else. The recipe draws
its tile from a different key than the probes did, and under its seeds the reference is at 1.000
by 2000 steps (0.996 at 1000 on one seed) and the relay without σ′ by 3250 on its slowest seed
(1.000 by 750 with σ′ on the same seed): the budget is set at 4000 so that the claims are about the
signals, not the clock, and dropping σ′ cost time on the last bit once. The probes also visited
rates 0.1, 0.02 and 0.005, budgets to 5000 steps, plain and sign SGD, and throttled updates.

**Measured at initialisation** (`probes/2026-09-07-signal-ladder-by-layer.py`, three seeds; per
layer, input → output: nonzero fraction / cosine with the reference / sign agreement where both
are nonzero):

| signal | layer 0 | layer 1 | layer 2 | output layer |
|---|---|---|---|---|
| soft·autodiff·σ′ (reference) | 0.76 / +1.00 / 1.00 | 0.85 / +1.00 / 1.00 | 0.75 / +1.00 / 1.00 | 1.00 / +1.00 / 1.00 |
| soft·relay·σ′ | 0.76 / +1.00 / 1.00 | 0.85 / +1.00 / 1.00 | 0.75 / +1.00 / 1.00 | 1.00 / +1.00 / 1.00 |
| soft·relay·1 | 0.76 / +0.98 / 1.00 | 0.85 / +0.98 / 1.00 | 0.75 / +0.98 / 1.00 | 1.00 / +0.97 / 1.00 |
| soft·uniform·σ′ | 0.76 / +0.14 / 0.49 | 0.85 / +0.20 / 0.57 | 0.75 / −0.06 / 0.53 | 1.00 / +1.00 / 1.00 |
| soft·uniform·1 | 0.76 / +0.12 / 0.49 | 0.85 / +0.21 / 0.57 | 0.75 / −0.06 / 0.53 | 1.00 / +0.97 / 1.00 |
| hard·autodiff·σ′ (straight-through) | 0.54 / +0.04 / 0.51 | 0.38 / +0.04 / 0.50 | 0.44 / −0.01 / 0.58 | 0.71 / +0.06 / 0.54 |
| hard·relay·σ′ | 0.54 / +0.04 / 0.51 | 0.38 / +0.04 / 0.50 | 0.44 / −0.01 / 0.58 | 0.71 / +0.06 / 0.54 |
| hard·relay·1 | 0.54 / +0.03 / 0.51 | 0.38 / +0.03 / 0.50 | 0.44 / −0.01 / 0.58 | 0.71 / +0.08 / 0.54 |
| hard·uniform·σ′ | 0.69 / +0.02 / 0.48 | 0.59 / +0.02 / 0.51 | 0.54 / −0.04 / 0.59 | 0.71 / +0.06 / 0.54 |
| hard·uniform·1 | 0.69 / +0.01 / 0.48 | 0.59 / +0.03 / 0.51 | 0.54 / −0.03 / 0.59 | 0.71 / +0.08 / 0.54 |

The relay is the reference on both passes (the theorem; `tests/test_signals.py` asserts it to
1e-7 on two shapes). Dropping σ′ keeps every sign and 0.97–0.98 of the cosine. The uniform split
agrees with the reference at the output layer, where nothing has been transported, and is at
chance in every hidden layer: the sign is lost after one hop. Every hard cell is at chance
everywhere, 0.54 even at the output layer: at this depth the soft activations of a random tile
wash toward one half, so the reference spreads over a gate's entries while the bits address one.
The fraction of back-edges carrying exactly zero (a gate insensitive to that input at the other
inputs' values): 0.00 in every layer on the soft pass, 0.47–0.51 on the bits.

**Measured in training** (`probes/2026-09-07-descent-by-transport.py`; hard accuracy at step 500 /
2000, seeds 0–2):

| signal | rate | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| soft·relay·σ′ | 0.1 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| soft·relay·1 | 0.1 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| soft·uniform·1 | 0.1 | 0.922 / 0.934 | 0.938 / 0.941 | 0.863 / 0.859 |
| soft·uniform·1 | 0.02 | 0.977 / 0.941 | 0.906 / 0.945 | 0.883 / 0.906 |
| hard·relay·σ′ | 0.1 | 0.461 / 0.449 | 0.477 / 0.473 | 0.543 / 0.496 |
| hard·relay·σ′ | 0.02 | 0.555 / 0.520 | 0.543 / 0.512 | 0.512 / 0.461 |
| hard·relay·1 | 0.1 | 0.512 / 0.477 | 0.582 / 0.562 | 0.469 / 0.492 |
| hard·relay·1 | 0.02 | 0.484 / 0.500 | 0.492 / 0.512 | 0.535 / 0.539 |
| hard·uniform·1 | 0.1 | 0.898 / 0.922 | 0.879 / 0.977 | 0.816 / 0.902 |
| hard·uniform·1 | 0.02 | 0.867 / 0.938 | 0.820 / 0.926 | 0.812 / 0.844 |

On the soft pass the relay reaches the target with or without σ′ (Adam does not see a positive
factor) and the blind split stalls short of it. On the bits the exact relay, with σ′ (which is
straight-through) or without (blastema's deployable `delta·basis`), does not leave chance at either
rate, nor at 0.005 for 5000 steps (run by hand: 0.51–0.56; the blind split 0.90–0.93). The blind
split trains to 0.84–0.98 and never to 1.000 in any of its runs.

**The path to the why.**

1. *Dead paths?* Half the back-edges on the bits carry zero, so a gate the circuit does not
   currently listen to gets no signal and cannot be recruited. `probes/2026-09-07-flip-credit.py`:
   gates the relayed error never reaches over the whole batch are 26–29 % in the hidden layers on
   the bits against 15–25 % on the soft pass, and the relay reaches nearly as many entries as the
   blind split (107 against 117 at the input layer). Real, and not the story.
2. *Is either hard signal right about a flip?* Same probe: for every entry with a nonzero signal,
   flip that table bit and compare the sign of the actual change of the hard loss with the sign the
   signal predicts. The relay is right 0.51–0.61 of the time, the blind split 0.48–0.51, at every
   layer including the output one. The relay is the first-order term of a two-term difference: on
   the bits the second term, the cost of breaking outputs that were right, is as large as the first,
   and a signal seeded by the residual cannot see it, since a right output has zero residual.
3. *Thrash?* `probes/2026-09-07-bits-dynamics.py`: under Adam the relay flips 60 table bits per
   step out of 672, under plain or sign SGD 160–270; the blind split 3–20. But
   `probes/2026-09-07-bits-thrash.py`: updating a random 2 % of the logits per step makes the
   relay's signal 0.99 consistent from step to step with 2.8 flips per step, and accuracy stays at
   chance (0.50–0.54). Consistent and wrong, not noisy.
4. *Who learns?* `probes/2026-09-07-bits-who-learns.py`: the two hard signals are identical at the
   output layer, so any difference is what their hidden updates do. Hidden layers frozen, both give
   0.52–0.61. The blind split's hidden updates lift it to 0.76–0.89 at 2 %, 0.81–0.88 at 10 %,
   0.84–0.94 free; the relay's lift it to nothing at any fraction.
5. *Alignment.* `probes/2026-09-07-bits-alignment.py`: the blind split carries every error back as
   if each gate's sensitivity to each input were +1. Under it, the fraction of nonzero hard
   sensitivities that are positive rises from 0.44–0.61 at initialisation to 0.66 / 0.72 / 0.71 /
   0.93 (input → output) by step 2000, with accuracy 0.90; under the relay it stays at 0.43–0.56.
   The circuit makes the blind feedback right.

6. *What the votes say.* `probes/2026-09-07-vote-coherence.py`: for each entry, how much the
   cases that address it agree, |Σ e| / Σ |e| over those cases. On the bits the relay's votes are
   unanimous in every layer (1.00) and the blind split's are split (0.53–0.76 in the hidden
   layers, 1.00 at the output layer). Unanimous because on the bits the relay can only say one
   thing: a wrong output on a live path credits a flip, a right output is silent, so every vote
   an entry receives says "flip" and none says "stay". The blind split's message is a direction
   for the gate's output, "lower" or "higher", the same whatever the gate's bit is; an entry
   already facing that way saturates and stays, and the entry moves on the majority of its cases.

**Why.** On the soft pass the chain rule is the right signal, the relay computes it as local
messages, and the blind split loses the sign after one hop and stalls. On the bits there is no
infinitesimal, and a right output is silent. The relay turns that into a *relative* instruction at
every gate on a live path, "flip", and never "stay": every entry addressed by a wrong reachable
case flips, every flip breaks other cases, which then vote to flip back, and the only configuration
the relay is content with is zero error, which it cannot reach because fixes are never weighed
against breaks. The blind split turns the same residuals into an *absolute* instruction, "your
output should be lower on this case": an entry already facing that way stays, only the gates whose
bit disagrees with the broadcast move, and the gates downstream learn to make the assumed positive
path true. That is why straight-through, which reached the target on
the flat tile (previous note), fails at four layers because there the hidden layers have to
learn. The blind split learns for the reason fixed random feedback trains a network (feedback
alignment): it is stable, and the forward circuit adapts to it, the gates drifting monotone in
their inputs. The previous note's open question, whether the chattering grows with depth, has its
answer: at depth the bits fail before chattering matters. What the blind split's ceiling is made
of (0.84–0.98, never 1.000; possibly monotone hidden features cannot carry the sum bits' parity) is
not measured here.

**Also found.** The two probes on the first shape were kept; every conclusion held there. Arity 3
was chosen so that depth bites (with 6 inputs a gate of arity 4 sees more of them); asked whether
the finding is an artefact of it, `probes/2026-09-07-arity-4-check.py` reran the transports at
arity 4 on the four-layer shape and on `(6, 32, 16, 4)`: on the bits the exact relay stays at
chance (0.47–0.65) and the blind split trains, on four of six runs to 1.000. What arity changes is
the soft pass: at arity 4 the blind split reaches the target on some seeds (1.000, 1.000, 0.977 on
the four-layer shape), so "the blind split stalls" measures how hard depth bites, and arity 3 is
where it bites cleanly. `Recipe.arity` stays 4 by default; only the two deep recipes say 3.

**Claims left behind.** `claims/test_2026_09_07_transports.py`: addition under the deep floor; the
uniform split at chance against the reference from the first hop and exact at the output layer;
descent on the relay without σ′ reaches the target where the blind split stalls; on the bits the
exact relay is at chance where the blind split is well clear of it.

**What would change it.** A flip credit with the second term, carried as a second channel (the
signed error, and the reach |Δ| of a flip): the exact effect of one flip, still local. A leaky
carry, ε plus the sensitivity, between the two transports. The workshop's Δ = −η·signal (step 2)
with these signals, separating the signal from Adam. A substrate that stores probabilities rather
than bits runs the soft relay on the chip: the soft/hard split maps onto analogue/digital, which
matters for the second substrate. And the bits' ceiling: what the blind split cannot build.

**For the map.** blastema's finding that `delta·basis` (hard·relay·1) is the deployable optimiser
held on flat tiles; at four layers it does not. On the bits, the deployable signal so far is the
blind split, and it learns by alignment, not by gradient.
