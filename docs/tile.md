# The tile

A tile is the smallest thing that computes: a layered circuit of look-up tables over a fixed
wiring. It is data plus one function, and every later piece of loom (signals, the rule, the
workshop) is written against exactly that.

## The data

Two tuples of arrays, one entry per layer.

- **Tables** (`logits`): for layer *l*, an array `[gates_l, 2**arity]`. Each row is one gate's
  truth table, stored as logits: `sigmoid(logit)` is the probability that the table reads 1 at
  that address. A gate of arity 4 has sixteen entries.
- **Wiring** (`wires`): for layer *l*, an array `[arity, gates_l]` of integer indices into the
  previous layer's lines. Any index array is a valid wiring: a line may feed one gate or many
  (fan-out is unconstrained), and `init` only chooses to spread fan-out as evenly as the fan-in
  allows.

`widths = (n_in, hidden…, n_out)` gives the number of lines leaving each layer. The checks use
`(4, 16, 8, 2)` at arity 4, twenty-six gates: the shape inherited from blastema's first rung, not a
derived minimum. With fan-out free, `(4, 8, 2)` works too, and a single layer of two arity-4 gates
already represents every function of four inputs, which leaves nothing to discover. Depth becomes
necessary when a gate cannot see all the inputs (arity below the input width); that is also where
carry chains live and, later, routing. The signal checks therefore run on a shape where depth is
forced, and this one stays the smoke shape.

The wiring's only constraints: indices in `[0, n_prev)`, shape `[arity, gates]`; a gate may read the
same line twice. Wiring becomes configuration at step 6: each index turns into a categorical over
the previous layer's lines, soft as a mixture (an assignment matmul), hard as an argmax, with the
straight-through trick on the choice, the same construction as the Mosaic router. `read` needs no
change, since it already takes the gathered inputs.

## The read

A gate *reads* its table at the address its inputs form. With bit inputs that is an ordinary
lookup: for XOR, table `[0, 1, 1, 0]`, inputs `a = 1, b = 0` form the address `1 + 2·0 = 1`, and
the table reads 1. The first input is the least significant address bit.

The same read is written so that it also makes sense for inputs in between 0 and 1. Each input
halves the table: an input `x` keeps `(1 − x)` of the even entries and `x` of the odd ones, and
after `arity` halvings one number remains. On bits this is exact. On probabilities it is the
table's expected value under the product distribution of its inputs, and it is differentiable.
That one function, `read`, is the whole substrate; `activations` applies it layer by layer.

## Two ways to read the same tile

| read | tables | what it is for |
|---|---|---|
| `soft` | `sigmoid(logit)` | gradients flow through everything; the training view |
| `hard` | `round(sigmoid(logit))` | the deployed circuit: bits in, bits out, no gradient |

Every check in loom is measured on the `hard` read. The difference between a tile's accuracy on
its training view and on the hard read is its *deploy gap*.

There is a third set of tables, but it belongs to the signals, not to the tile: the *straight-through*
tables, `round(soft) + (soft − stop_gradient(soft))`, are exactly the bits in value and
differentiate like the probabilities. They are how a gradient is made to run on the deployed pass.
Only the rounding's derivative is replaced; everything downstream (the residual at the outputs, the
paths back through other gates) is evaluated on the bits that flowed, so that gradient is sparser
and coarser than the soft one and agrees with it only once the soft tables have saturated. That is
why descent on the bits wants a smaller step (bits chatter at the soft rate), and why it has no
deploy gap at any step: its training view is the deployed circuit.

The sigmoid's slope is an implicit temperature that today lives in the scale of the logits
(`init(scale=…)`); an explicit temperature on the soft read is a two-line change, held until a step
needs it (annealing would).

## What a tile is not, yet

No engram, no message to neighbours, no learnable wiring: a tile knows only its tables, its
wiring and its inputs. Those enter as later steps, each with its own page.
