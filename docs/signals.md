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

## The gradient, factored

For one gate with logits $z \in \mathbb{R}^{2^k}$ over its addresses, the soft table is
$T = \sigma(z)$ and the hard one $H = \mathbb{1}[z > 0]$. Its read at inputs $u \in [0,1]^k$ is

$$r(u) = \sum_a P(a \mid u)\, T[a], \qquad P(a \mid u) = \prod_j u_j^{a_j} (1-u_j)^{1-a_j},$$

and on bits $P(\cdot \mid u)$ is the one-hot of the selected address. With the squared loss $L$,
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
is therefore the cell (`hard`, `autodiff`, kept) of the taxonomy and not a thing of its own. The
next chunk adds the two transports a chip could run: the exact relay, written as a local message
(on the same pass it must reproduce autodiff's numbers gate by gate, a theorem to check
numerically, after which `via` may reduce to {exact, uniform} with autodiff as the reference
implementation of exact), and the value-blind uniform split; and with them dropping $\sigma'$,
which a chip that stores bits rather than logits cannot see.
