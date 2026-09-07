# Signals

A signal is what a local update may read: an array the shape of the tile's logits, one entry per
table entry, saying which way and how much each should move. Descent consumes a signal; so will
the rule. The point of naming signals is to say, precisely, how much of the whole circuit's
gradient a chip could actually produce.

## A signal is three coordinates, not a name

| coordinate | values today | meaning |
|---|---|---|
| `on` | `soft`, `hard` | the pass the signal is computed on: the soft forward, or the deployed bits |
| `via` | `autodiff` (the relay and the uniform split come next) | how the error at the outputs reaches each logit |
| `surrogate` | kept (dropping it comes with the relay) | whether the factor σ′(logit) is included |

`REFERENCE = Signal("soft", "autodiff", surrogate=True)` is the true gradient of the loss on the
soft pass: the idealised signal every other one is scored against. `Signal("hard")` is the same
autodiff run on the bits, the signal usually called straight-through. Naming products
(`true_grad_soft`, `relay_hard`, …) would explode; naming coordinates keeps the table small and
makes every combination a cell the matrix can visit.

## The residual, and one loss for every pass

The error at the outputs is the residual, read minus demanded. On the bits it is itself a bit,
−1, 0 or 1: the error a chip can see. The loss is half the squared error, chosen because its
derivative at an output *is* the residual, in every pass. A cross-entropy is infinite when a bit is
wrong and, once clipped, its derivative is enormous and no longer says how wrong; it would tie the
loss to the soft pass.

## Running autodiff on the bits

The hard read has no gradient (rounding is flat), so the straight-through tables are used instead:
`round(soft) + (soft − stop_gradient(soft))`, exactly the bits in value and differentiating like
the probabilities. This replaces *only* the derivative of the rounding step. Everything downstream
is evaluated on the bits that actually flowed, which changes three factors of the gradient:

- **the error** is the residual in bits, not the graded soft error: a gate whose entry sits at
  0.6 with a target of 1 gets no error at all on the bits, and −0.4 on the soft pass;
- **the address**: with bit inputs, one table entry per gate per case receives gradient, where the
  soft pass spreads it over all entries by the product distribution of its inputs;
- **the paths back**: a rounded table can be exactly insensitive to an input, so whole paths carry
  zero until some bit flips; soft tables always let something through.

Only σ′(logit) is shared, and it is positive, so it changes no sign. At initialisation the two
gradients are nearly orthogonal and the one on the bits is several times sparser; they converge as
the soft tables saturate, which is why a soft-trained tile ends with no deploy gap either. Being
piecewise constant, the bits gradient changes sign discontinuously when a table entry crosses one
half, and at the soft learning rate a logit marches to the threshold, flips, and marches back:
descent on the bits wants a smaller step. The numbers are in
`notes/2026-09-07-straight-through-is-a-signal-cell.md`.

## Where straight-through sits in the ladder

Factor the gradient on the bits: the residual, transported back through the rounded tables' local
Jacobians to a gate's output, times the one-hot of the address its bit inputs select, times σ′.
That is blastema's decomposition, error relay × basis × surrogate, with the relay evaluated on the
deployed pass. So straight-through is a cell of the taxonomy, (hard, autodiff, kept), and not a
thing of its own. The next chunk adds the two transports a chip could run, the relay through each
gate's own table and the value-blind uniform split, and with them dropping σ′, which a chip that
stores bits rather than logits cannot see.
