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

`widths = (n_in, hidden…, n_out)` gives the number of lines leaving each layer; the smallest tile
with a hidden layer, `(4, 16, 8, 2)` at arity 4, has twenty-six gates.

## The read

A gate *reads* its table at the address its inputs form. With bit inputs that is an ordinary
lookup: for XOR, table `[0, 1, 1, 0]`, inputs `a = 1, b = 0` form the address `1 + 2·0 = 1`, and
the table reads 1. The first input is the least significant address bit.

The same read is written so that it also makes sense for inputs in between 0 and 1. Each input
halves the table: an input `x` keeps `(1 − x)` of the even entries and `x` of the odd ones, and
after `arity` halvings one number remains. On bits this is exact. On probabilities it is the
table's expected value under the product distribution of its inputs, and it is differentiable.
That one function, `read`, is the whole substrate; `activations` applies it layer by layer.

## Three ways to read the same tile

| mode | tables | what it is for |
|---|---|---|
| `soft` | `sigmoid(logit)` | gradients flow through everything; the training view |
| `hard` | `round(sigmoid(logit))` | the deployed circuit: bits in, bits out, no gradient |
| `ste` | the hard value carrying the soft gradient | train the deployed circuit directly |

`ste` is *straight-through*: `hard + (soft − stop_gradient(soft))`, which is exactly `hard` in
value and differentiates like `soft`. Only the derivative of the rounding step is replaced;
everything downstream of it (the residual at the outputs, the paths back through other gates) is
evaluated on the bits that actually flowed, so its gradient is sparser and coarser than the soft
one and agrees with it only once the soft tables have saturated. That is why straight-through
descent wants a smaller step: bits chatter at the soft rate.

Every check in loom is measured on the `hard` read. A tile trained soft is done when its hard
read agrees with its soft one on every case; a tile trained straight-through has no gap to close,
because it trained on bits.

## What a tile is not, yet

No engram, no message to neighbours, no learnable wiring: a tile knows only its tables, its
wiring and its inputs. Those enter as later steps, each with its own page.
