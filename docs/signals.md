# Signals

A signal is what a local update may read: an array the shape of the tile's logits, one entry per
table entry, saying which way and how much each should move. Descent consumes a signal; so will
the rule. The point of naming signals is to say, precisely, how much of the whole circuit's
gradient a chip could actually produce.

## A signal is three coordinates, not a name

| coordinate | values today | meaning |
|---|---|---|
| `on` | `soft`, `hard` | the pass the signal is computed on: the soft forward, or the deployed bits |
| `via` | `autodiff`, `relay`, `uniform` | how the error at the outputs reaches each logit |
| `surrogate` | kept, dropped | whether the factor σ′(logit) is included |

`REFERENCE = Signal("soft", "autodiff", surrogate=True)` is the true gradient of the loss on the
soft pass: the idealised signal every other one is scored against. `Signal("hard")` is the same
autodiff run on the bits, the signal usually called straight-through. Naming products
(`true_grad_soft`, `relay_hard`, …) would explode; naming coordinates keeps the table small and
makes every combination a cell the matrix can visit: `signals.CELLS` lists the ten the code
supports (autodiff cannot drop σ′; the relay computes that signal).

## The residual, and one loss for every pass

The error at the outputs is the residual, read minus demanded. On the bits it is itself a bit,
−1, 0 or 1: the error a chip can see. The loss is half the squared error, chosen because its
derivative at an output *is* the residual, in every pass. A cross-entropy is infinite when a bit is
wrong and, once clipped, its derivative is enormous and no longer says how wrong; it would tie the
loss to the soft pass.

## The gradient, factored

For one gate with logits $z \in \mathbb{R}^{2^k}$ over its addresses, the soft table is
$T = \sigma(z)$ and the hard one $H = \mathbb{1}[z > 0]$. Its read at inputs $u \in [0,1]^k$ is

$$r(u) = \sum_a P(a \mid u)\, T[a], \qquad P(a \mid u) = \prod_j u_j^{a_j} (1-u_j)^{1-a_j},$$

In words, for one gate with $k$ inputs:

| symbol | what it is | ranges over |
|---|---|---|
| $T$ | the gate's table, one entry in $[0,1]$ per input pattern; bits when hard, probabilities when soft | $2^k$ entries |
| $a$ | the table's index, an integer from $0$ to $2^k - 1$; its binary digits are the $a_j$, input $j$ being bit $j$ of the index, least significant first (the convention `read` uses) | the sum: all $2^k$ entries |
| $u$ | what is actually at the inputs, $k$ numbers in $[0,1]$; $u_j$ is input $j$ | the product: the $k$ inputs |
| $P(a \mid u)$ | the probability that the pattern present at the inputs is the one index $a$ spells, when input $j$ is a 1 with probability $u_j$, independently | one factor per input: $u_j$ if bit $j$ of $a$ is 1, $1-u_j$ if it is 0 (the exponents are that switch) |
| $T[a]$ | the table's entry at index $a$ | |
| $r(u)$ | the gate's output: every entry weighted by how likely its index is to be the one addressed | |

- On bits, exactly one index has $P = 1$ and the read is the plain lookup $T[a]$.
- On soft inputs, $P$ is a distribution over the $2^k$ indices, summing to 1, and the read is the table's
  expectation under it. One formula covers both passes.

Example, $k = 2$, the XOR table $T = (0, 1, 1, 0)$ over indices $a = (00, 01, 10, 11)$, inputs
$u = (0.9, 0.2)$:

```
P(00 | u) = 0.1 · 0.8 = 0.08
P(01 | u) = 0.1 · 0.2 = 0.02
P(10 | u) = 0.9 · 0.8 = 0.72
P(11 | u) = 0.9 · 0.2 = 0.18        (sum 1)
r(u) = 0.08·0 + 0.02·1 + 0.72·1 + 0.18·0 = 0.74
```

With bits $u = (1, 0)$ only index $10$ survives and $r = T[10] = 1$. In code, `signals.address`
builds $P$ one input at a time, input $i$ being address bit $i$, the convention the halving `read` uses.

On bits $P(\cdot \mid u)$ is thus the one-hot of the selected address. With the squared loss $L$,
the gradient at one table entry is a product of three factors:

$$\frac{\partial L}{\partial z[a]} \;=\; e \cdot P(a \mid u) \cdot \sigma'(z[a]),$$

where $e = \partial L / \partial r$ is the **error at the gate's output**: the residual at the
circuit's outputs, transported back through the downstream gates' local Jacobians,

$$\frac{\partial r}{\partial u_j} = \sum_a P(a \mid u)\,\big(T[a \,|\, u_j{=}1] - T[a \,|\, u_j{=}0]\big),$$

each gate passing error to its input $j$ weighted by how much its own table changes when $j$
flips, at the other inputs' values. That transport *is* the relay through each gate's own table,
and autodiff computes exactly it.

## Running it on the bits

On the soft pass every factor is evaluated at soft $u$ and $T$. On the hard pass the inputs are
bits and the tables are $H$: $P$ is one-hot, the Jacobians are differences of bits in
$\{-1, 0, 1\}$, and the only factor that does not exist there is $\partial H / \partial z$, which is
zero because rounding is flat. The `surrogate` coordinate is the answer to "what do we put there":

- **kept**: $\partial H/\partial z := \sigma'(z)$, i.e. $\mathrm{round}(\sigma(z))$ is differentiated
  as $\sigma(z)$, giving $\hat e \cdot \mathbb{1}[a = a(u)] \cdot \sigma'(z[a])$;
- **dropped**: $\hat e \cdot \mathbb{1}[a = a(u)]$, what a chip that stores bits can compute.

There is one surrogate, not two: the classical straight-through choice (identity through the
rounding) chained with the sigmoid's derivative *is* $\sigma'(z)$. In code the kept case is the
straight-through tables, $\mathrm{round}(T) + (T - \mathrm{stop\_gradient}(T))$: exactly the bits
in value, differentiating like $T$. So changing `on` from `soft` to `hard` means every factor
evaluated on the deployed pass, plus a declared choice for the one derivative the bits do not have.

Three factors change between the passes: the **error** ($e$ graded on the soft pass, $\hat e$ a
transported bit on the hard one; a gate whose entry sits at 0.6 with a target of 1 gets no error on
the bits and $-0.4$ on the soft pass), the **address** ($P$ spread over entries by the product
distribution, or one-hot), and the **paths back** (a rounded table can be exactly insensitive to an
input, so whole paths carry zero until some bit flips). Only $\sigma'$ is shared, and it is
positive, so it changes no sign. At initialisation the two gradients are nearly orthogonal and the
one on the bits is several times sparser; they converge as the soft tables saturate. Being
piecewise constant, the hard-pass gradient changes sign discontinuously when a table entry crosses
one half, and at the soft learning rate a logit marches to the threshold, flips, and marches back:
descent on the hard pass wants a smaller step. Numbers in
`notes/2026-09-07-straight-through-is-a-signal-cell.md`.

## Where straight-through sits in the ladder

Read the factored gradient on the hard pass again: the error relayed through the rounded tables to
a gate's output, times the one-hot of the selected address, times $\sigma'$. That is blastema's
decomposition, *relay × basis × surrogate*, with the relay on the deployed pass. Straight-through
is therefore the cell (`hard`, `autodiff`, kept) of the taxonomy and not a thing of its own.

## The transports

Autodiff is the reference implementation: it runs on any substrate with a soft read and asks
nothing of the hardware. The two others are written as what a gate could compute, from two
quantities local to its own read.

**The address distribution** (`signals.address`), how much each entry is selected by the inputs:

$$P(a \mid u) = \prod_j u_j^{a_j} (1-u_j)^{1-a_j}, \qquad r(u) = \sum_a P(a \mid u)\, T[a].$$

Soft inputs spread it over the entries; bits make it one-hot. A test asserts that the distribution
against the table is the read.

**The sensitivity** (`signals.sensitivity`), how much the read moves with each input. The read is
multilinear, so it is a difference of two reads at the other inputs' current values:

$$\frac{\partial r}{\partial u_j} = r(u \,|\, u_j{=}1) - r(u \,|\, u_j{=}0),$$

on bits: whether flipping that input flips the output, in $\{-1, 0, 1\}$.

**The sweep** (`signals.sweep`). Start with $e = \partial L/\partial r$ at the outputs, the
residual scaled as the mean loss is. At each layer, from the output back: the per-entry signal is
the error at the gate's output against its address distribution, summed over cases, times
$\sigma'(z)$ if the surrogate is kept,

$$s[a] = \sum_b e_b\, P_b(a \mid u_b) \;(\cdot\, \sigma'(z[a])),$$

and the message to input $j$ is $e_b$ times a *carry*, added onto the line it came from, so a
line's error is the sum over the gates it feeds. Two carries make two transports:

- **relay**: the carry is the gate's own sensitivity. The sweep is then the chain rule written as
  local messages, and it reproduces autodiff to float precision on both passes, on a flat and on a
  deep shape (`tests/test_signals.py`). Where the two exist together, `relay` is what a chip would
  run and `autodiff` is how we check it.
- **uniform**: the carry is one. Every gate passes its error unchanged to every input, whatever it
  computes: the adjoint of a network that computes nothing in particular, value-blind. It agrees
  with the relay at the output layer, where nothing has been transported, and nowhere else.

**Dropping σ′.** The kept signal is the derivative with respect to the logit; the dropped one is
the derivative with respect to the stored entry, $T[a]$ or $H[a]$, which is what the substrate
holds (a logit is a training-time coordinate). Since $\sigma' > 0$ nothing changes sign (a test),
and under Adam, which normalises per coordinate, nothing changes in reach on the soft pass either,
though on one seed the last bit took four times longer without it.

## What the transports do at depth

Scored against the reference on a four-layer tile of arity-3 gates on 6-bit addition
(`notes/2026-09-07-the-relay-is-autodiff-and-the-bits-learn-by-alignment.md`):

- On the soft pass, the relay *is* the reference and reaches the target with or without σ′. The
  uniform split loses the reference's sign after one hop (chance-level agreement in every hidden
  layer) and descent on it stalls short of the target. Value-awareness, at the level of descent.
- On the bits, both transports are at chance against the reference in every layer, and the
  surprise is which one descent can follow: the exact relay, with σ′ (straight-through) or without
  (blastema's deployable `delta·basis`), does not leave chance at any rate or budget tried, while the
  blind split trains to 0.84–0.98. Not dead paths (real, but the relay reaches nearly as many
  entries), not thrash (throttled to a few flips a step its signal is 0.99 consistent and still at
  chance). On the bits there is no infinitesimal: the relay is exact about a linearisation that
  does not describe a flip, and a signal seeded by the residual cannot see the cost of breaking the
  outputs that were right. The blind split learns by **feedback alignment**: its carry is a fixed
  +1, and the gates drift monotone to make it right (the positive fraction of their sensitivities
  climbs from one half toward one along training, and stays at one half under the relay).

So straight-through, which reached the target on the flat tile, is a signal for shallow circuits;
at depth the bits need either the soft pass on the chip (a substrate that holds probabilities), a
signal that credits a flip exactly (a second relayed channel, next), or a rule that learns to use
a blind, stable feedback. That is the question step 2 inherits.
