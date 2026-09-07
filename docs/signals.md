# Signals

A signal is what a local update may read: an array the shape of the tile's logits, one entry per
table entry, saying which way and how much each should move. Descent consumes a signal; so will
the rule. The point of naming signals is to say, precisely, how much of the whole circuit's
gradient a chip could produce, and at what physical cost. The symbols are worked through on one
XOR gate in [`signals-worked.md`](signals-worked.md).

## A signal is three coordinates, not a name

| coordinate | values | meaning |
|---|---|---|
| `on` | `soft`, `hard` | the pass the circuit is linearised on: the soft forward, or the deployed bits |
| `via` | `autodiff`, `relay`, `uniform`, `direct`, `flip` | whose transposed Jacobian carries the error back to each gate: the circuit's own (by autodiff, or locally by the relay), the same wiring with every gate a sum (`uniform`, feedback alignment shaped by the wiring), a fixed random ±1 bus from the outputs (`direct`, direct feedback alignment), or, outside the frame, the exact credit of one bit flip (`flip`) |
| `to` | `logit`, `entry` | the parameter the local partial is taken to: the logit, or the stored entry, which leaves out σ′(logit) |

`REFERENCE = Signal("soft", "autodiff", "logit")` is the true gradient of the loss on the soft
pass: the idealised signal every other one is scored against. `Signal("hard")` is the same
autodiff run on the bits, the signal usually called straight-through. Naming products would
explode; naming coordinates keeps the table small and makes every combination a cell the matrix
can visit: `signals.CELLS` lists the eighteen the code supports (autodiff cannot take the partial
to the entry). A label such as `hard.relay.entry` is the coordinates joined, a display name and a
test id; a gate asserts no code branches on one.

## One frame: the adjoint method

Every signal is an instance of the adjoint method, which says that the gradient at a parameter is
the adjoint variable at the place the parameter acts, times that place's local partial with
respect to the parameter. For a gate $g$ and one of its logits,

$$\frac{\partial L}{\partial z[a]} = \lambda_g \cdot \frac{\partial r_g}{\partial z[a]}
= \underbrace{e}_{\lambda_g} \cdot \underbrace{P(a \mid u) \cdot \sigma'(z[a])}_{\partial r_g / \partial z[a]},$$

the three factors derived below. The adjoint variable $e$ is carried back from the outputs
through a transposed Jacobian; the local partial never leaves the gate. The three coordinates are
three choices inside that formula:

| coordinate | in adjoint terms |
|---|---|
| `on` | where the system is linearised: at the soft state, or at the bits |
| `via` | whose transposed Jacobian carries $\lambda$: the circuit's own, computed by autodiff or locally by the relay; the all-sums network's on the same wiring (uniform); a fixed one-layer network's from the outputs to every gate (direct) |
| `to` | which parameter the local partial is taken to: the logit, or the stored entry |

The frame also says what is *outside* it: a signal carrying a second-order term, the exact credit
of a bit flip, is not an adjoint. A learned transport is the adjoint of a network with learned
Jacobians; e-prop is this formula with $\lambda$ approximated by a broadcast and the local partial
integrated over time into an eligibility trace; the rule of step 3 is a learned readout in place
of the product $\lambda \cdot \partial r / \partial \theta$. The code is laid out in the frame's
order: the seed, the local partials, the adjoints, the readout, then one function per `via`.

## The maths, in order

**1. The read.** One gate has $k$ inputs and a table of $2^k$ entries. With logits
$z \in \mathbb{R}^{2^k}$, the soft table is $T = \sigma(z)$ and the hard one
$H = \mathbb{1}[z > 0]$. At inputs $u \in [0,1]^k$ the read is

$$r(u) = \sum_a P(a \mid u)\, T[a], \qquad P(a \mid u) = \prod_j u_j^{a_j} (1-u_j)^{1-a_j}.$$

$a$ is the table's index, $0$ to $2^k - 1$, its binary digits $a_j$ saying what that index wants
of input $j$ (input $j$ is bit $j$, least significant first, the convention `read` uses).
$P(a \mid u)$, the **address distribution** (`signals.address`), is the probability that the
inputs together spell index $a$ when each $u_j$ is read as the probability that input $j$ is a 1,
independently: one factor per input. On bits it is the one-hot of the index the inputs spell and
the read is a lookup; on soft inputs it is a distribution over the entries and the read is the
table's expectation under it. One formula, both passes.

**2. The loss and its seed.** The loss is half the squared error over cases and output bits,
chosen because its derivative at an output *is* the residual, read minus demanded, in every pass:
on the bits a bit, −1, 0 or 1, the error a chip can see. A cross-entropy is infinite when a bit is
wrong and, clipped, no longer says how wrong. The seed of every adjoint (`signals.seed`) is
$\partial L / \partial r$ at the outputs, the residual over the mean's $N$.

**3. The local partials.** The read is linear in each entry and the entry is the sigmoid of its
logit:

$$\frac{\partial r}{\partial T[a]} = P(a \mid u), \qquad \frac{\partial T[a]}{\partial z[a]} = \sigma'(z[a]) = T[a]\,(1 - T[a]).$$

The first is the address distribution again (entry ← output); the second (`signals.slope`) is
at most $1/4$ and near $0$ once the entry is saturated. `to="entry"` stops at the first.

**4. The gradient at one gate, three factors.** Chain rule:

$$\frac{\partial L}{\partial z[a]} \;=\; e \cdot P(a \mid u) \cdot \sigma'(z[a]),$$

with $e = \partial L / \partial r_g$ the error at *this* gate's output.

- On bits the middle factor is one-hot: only the entry the case addressed gets a gradient. On
  soft inputs every entry gets a share, in proportion to $P$.
- Only the first factor comes from elsewhere. $P$ and $\sigma'$ are local to the gate; $e$ is
  what has to travel, and *how* it travels is the `via` coordinate.
- Over a batch the first two factors are summed over cases and the third is shared:
  $\partial L/\partial z[a] = \sigma'(z[a]) \sum_b e_b\, P_b(a \mid u_b)$. That sum is the
  **readout** (`signals.readout`): the adjoint variables against the address distribution, times
  the slope when `to="logit"`.

**5. The sensitivity.** How much a gate's read moves with each input, the gate's row of the
Jacobian. The read is multilinear, so it is a difference of two reads at the other inputs'
current values, equivalently the table's difference under the address distribution:

$$\frac{\partial r}{\partial u_j} = r(u \,|\, u_j{=}1) - r(u \,|\, u_j{=}0) = \sum_a P(a \mid u)\,\big(T[a \,|\, u_j{=}1] - T[a \,|\, u_j{=}0]\big).$$

On bits: whether flipping that input flips the output, in $\{-1, 0, 1\}$ (`signals.sensitivity`).

**6. The layered adjoint** (`signals.layered`). $\lambda$ at the output gates is the seed. One
layer back, a line's $\lambda$ is the sum, over the gates $h$ it feeds, of $h$'s $\lambda$ times a
*carry*:

$$\lambda_g = \sum_{h \text{ fed by } g} \lambda_h \cdot c_{h,j}, \qquad
c_{h,j} = \frac{\partial r_h}{\partial u_{h,j}} \;\text{(relay)} \quad\text{or}\quad c_{h,j} = 1 \;\text{(uniform)}.$$

The forward gathers each gate's inputs along the wiring, `u[:, w]`; the adjoint scatters and adds
along the same wiring, `zeros.at[:, w].add(...)`, which is the transpose of the gather. So the
recursion applies the transposed Jacobian of the forward pass layer by layer from the outputs,
as EventProp does for a spiking network with time in place of depth. With the sensitivity as the
carry it is the chain rule, and it reproduces autodiff to float precision on both passes (a test):
`relay` is what a chip would run, `autodiff` how we check it. With ones as the carry it is the
exact adjoint of a different network, the same wiring with every gate a sum: value-blind, a fixed
feedback whose weights are the number of wiring paths from each gate to each output. That is
feedback alignment shaped by the wiring (`uniform`).

**7. The direct adjoint** (`signals.direct`). No layers: $\lambda$ at every hidden gate is the
seed through a fixed matrix, $\lambda_g = \sum_o B[g, o]\, e_o$, with $B$ random ±1 drawn once
from the tile's shape and the identity at the output layer, whose gates read their own residual.
Direct feedback alignment (`direct`): what a bus and a coefficient per gate would compute. A test
asserts that the same construction with $B$ = the wiring's path counts is the uniform split.

**8. The readout**, item 4's sum, closes the frame: a signal is a choice of adjoint, then
$s[a] = \sum_b \lambda_{g,b}\, P_b(a \mid u_b)$, times $\sigma'(z[a])$ when `to="logit"`.

**9. Outside the frame: the flip credit** (`flip`). On the bits an entry does not move, it
flips, and the loss change of a flip has two terms per output it reaches, $\Delta_o \in \{-1,0,1\}$:

$$\Delta L = \sum_o \big( e_o\, \Delta_o + \tfrac{1}{2} |\Delta_o| \big).$$

The first is what the relay carries. The second is the cost of changing an output that was right,
invisible to any adjoint because a right output has zero residual, and on bits as large as the
first. It is carried by a second layered adjoint, the **reach**, the same recursion with
$|c|$ as the carry seeded by ones, counting the outputs a flip at the gate changes. The signal is
the relay's plus half the reach in the direction of the flip, $(1 - 2H[a])$: exact for one flip,
not a gradient.

## Running it on the bits

On the soft pass every factor is evaluated at soft $u$ and $T$. On the hard pass the inputs are
bits and the tables are $H$: $P$ is one-hot, the sensitivities are differences of bits in
$\{-1, 0, 1\}$, and the only factor that does not exist there is $\partial H / \partial z$, which is
zero because rounding is flat. `to` is the answer to "what do we put there": to the logit,
$\partial H/\partial z := \sigma'(z)$, differentiating $\mathrm{round}(\sigma(z))$ as $\sigma(z)$;
to the entry, nothing, what a chip that stores bits can compute. There is one surrogate, not two:
the classical straight-through choice (identity through the rounding) chained with the sigmoid's
derivative *is* $\sigma'(z)$. In code the kept case is the straight-through tables,
$\mathrm{round}(T) + (T - \mathrm{stop\_gradient}(T))$: exactly the bits in value, differentiating
like $T$.

Three factors change between the passes: the **error** ($e$ graded on the soft pass, a
transported bit on the hard one; a gate whose entry sits at 0.6 with a target of 1 gets no error
on the bits and $-0.4$ on the soft pass), the **address** (spread over entries, or one-hot), and
the **paths back** (a rounded table can be exactly insensitive to an input: half the back-edges
carry zero on a random deep tile). Only $\sigma'$ is shared, and it is positive, so it changes no
sign. At initialisation the two gradients are nearly orthogonal and the one on the bits is several
times sparser; they converge as the soft tables saturate. Being piecewise constant, the hard-pass
gradient changes sign when a table entry crosses one half, and at the soft learning rate a logit
marches to the threshold, flips, and marches back: descent on the hard pass wants a smaller step.

## The transports, and what they cost

A signal is deployable on a fabric if each gate's update needs only things the fabric has. Three
costs: state per gate beyond its table, reads of its own table per case, and wiring for $\lambda$.

| `via` | state per gate | local reads per case | wiring for the error |
|---|---|---|---|
| `autodiff` | not a fabric signal: the whole graph's activations and a transpose pass | | |
| `relay` | a counter per entry (the logit's stand-in) | its own table at each flipped input: $k$ reads | a reverse channel per forward wire, carrying a 2-bit message ($\lambda$ times a sensitivity in $\{-1,0,1\}$), summed where a line fans out |
| `uniform` | the same counters | none | a reverse channel per wire, but one message per gate broadcast to all its sources, plus the fan-out sum |
| `direct` | the same counters, and a fixed ±1 coefficient per output | none | a bus of $n_{out}$ residual bits, no reverse wiring |
| `flip` | the same counters | $k$ reads | a reverse channel per wire carrying two numbers, the error and the reach |
| any, on the soft pass | probabilities rather than bits, for activations and tables: an analogue or stochastic representation | | as above |

The relay implies the transpose of the wiring, every forward connection with a way back, but no
duplicated tables: the sensitivity is computed downstream from the gate's own table and sent as a
value. On an FPGA that is a second routing network and an adder per line. The bus is the cheapest
learning channel there is, and it has the shape of a neuromorphic three-factor rule: a global
error, a local eligibility (the addressed entry), and local state.

## What the transports do at depth

Scored against the reference and run under descent on a four-layer tile of arity-3 gates on
6-bit addition (`notes/2026-09-07-the-relay-is-autodiff-and-the-bits-learn-by-alignment.md`,
`notes/2026-09-07-direct-feedback-and-the-flip-credit.md`):

- **On the soft pass** the relay *is* the reference and reaches the target with the partial to
  the logit or to the entry. Every blind transport loses the reference's sign after one hop
  (chance-level agreement in every hidden layer) and stalls short of the target, the wiring-shaped
  one highest, the random bus lower. Value-awareness, at the level of descent.
- **On the bits** every transport is at chance against the reference in every layer, and the
  question is which one descent can follow. The exact relay, to the logit (straight-through) or to
  the entry, does not leave chance at any rate, budget or optimiser tried. A right output is
  silent on the bits, so the relay can only ever say "flip" to a gate on a live path and never
  "stay": every vote an entry receives is unanimous, fixes are never weighed against breaks, and the
  only state the relay is content with is zero error. Not dead paths (real, but the relay reaches
  nearly as many entries as the blind split), not thrash (throttled to a few flips a step its
  signal is 0.99 consistent and still at chance).
- **Give the relay its "stay" votes** and it trains: the flip credit reaches 0.86–0.94 on the bits
  and then stops exactly, at a state where no single flip helps, a local optimum of single flips.
- **The wiring-shaped feedback** trains to 0.84–0.98 and keeps moving. Its message is a direction
  for the gate's output rather than a flip, so an entry already facing that way stays and the
  entry moves on the majority of its cases; and it learns by **feedback alignment**: its carry is
  a fixed +1 and the gates drift monotone to make it right (the agreement between the circuit's
  actual Jacobian and the feedback climbs from one half to 0.8–0.96 along training, and stays at
  one half under the relay).
- **The random bus** trains worst of the blind transports (0.63–0.73 on the bits): the circuit
  aligns to it partly, then loses it. Random signs per gate and output ask a gate to affect two
  outputs through one shared path with opposite signs, and to answer outputs it cannot reach;
  the wiring-shaped feedback asks every gate for one thing it can do alone, be monotone. Alignment
  wants a feedback the forward circuit can realise (a reading, to be tested).

So straight-through, which reached the target on the flat tile, is a signal for shallow circuits.
At depth a bits fabric has two signals that train, the wiring-shaped feedback and the flip credit,
and the one autodiff would suggest is the one that does not. The soft pass on the chip, a
substrate that holds probabilities, keeps the exact signal; that is the question the second
substrate inherits, and the question step 2 inherits is which of these signals the workshop is
trained on.

## Where straight-through sits in the ladder

Read the factored gradient on the hard pass again: the error relayed through the rounded tables
to a gate's output, times the one-hot of the selected address, times $\sigma'$. That is the
decomposition *relay × address × slope* with the relay on the deployed pass, and straight-through
is the cell (`hard`, `autodiff`, `logit`) of the taxonomy, not a thing of its own.
