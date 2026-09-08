# Signals

A signal is what a local update may read: an array the shape of the tile's logits, one entry per
table entry, saying which way and how much each should move. Descent consumes a signal; so will
the rule. The point of naming signals is to say, precisely, how much of the whole circuit's
gradient a chip could produce, and at what physical cost. The symbols are worked through on one
XOR gate in [`signals-worked.md`](signals-worked.md).

## A signal is three coordinates, not a name

| coordinate | values | meaning | in adjoint terms |
|---|---|---|---|
| `on` | `soft`, `hard` | the pass the derivatives are taken on: the soft forward, or the deployed bits | where the system is linearised (below) |
| `via` | `autodiff`, `relay`, `uniform`, `direct`, `reachable`, `flip` | how the error at the outputs reaches each gate | whose transposed Jacobian carries the error: the circuit's own (by autodiff, or locally by the relay); the same wiring with every gate a sum (`uniform`, feedback alignment shaped by the wiring); a fixed random ±1 bus from the outputs (`direct`, direct feedback alignment), or that bus restricted to the outputs a gate can reach (`reachable`, which is no longer DFA: it knows the wiring's reachability); outside the frame, the exact credit of one bit flip (`flip`, on the bits only) |
| `to` | `logit`, `entry`, `gate` | where the local partial stops: the logit $z[a]$; the stored entry $T[a]$ one step earlier, which leaves out σ′; or the gate's output, before the address, every entry receiving the gate's error | `logit` and `entry` are per entry; `gate` is the adjoint variable itself, broadcast: the per-gate signal a wire or a bus actually carries, exposed as such by `signals.gate_errors` |

`REFERENCE = Signal("soft", "autodiff", "logit")` is the true gradient of the loss on the soft
pass: the idealised signal every other one is scored against. `Signal("hard")` is the same
autodiff run on the bits, the signal usually called straight-through. Naming products would
explode; naming coordinates keeps the table small and makes every combination a cell the matrix
can visit: `signals.CELLS` lists the twenty-eight the code supports (autodiff cannot take the
partial to the entry; the flip credit is a bits quantity, per entry). A label such as `hard.relay.entry` is the
coordinates joined, a display name and a test id; a gate asserts no code branches on one.

**Linearised** means that every derivative below is taken at a state. On the soft pass that state
is the soft tables and the soft activations, and the derivatives are the ordinary ones. On the
bits it is the rounded tables and the bit activations: every derivative of the read becomes a
difference of two reads, exact for the flip of one input, and the one derivative that does not
exist there, the rounding's, is what the `to` coordinate settles.

## One frame: the adjoint method

Every signal is an instance of the adjoint method, which says that the gradient at a parameter is
the adjoint variable at the place the parameter acts, times that place's local partial with
respect to the parameter. Here the place is a gate: the adjoint variable is the error at its
output, carried back from the loss through a transposed Jacobian, and the local partial never
leaves the gate. The three coordinates are the three choices inside that sentence: where the
derivatives are taken (`on`), whose transposed Jacobian carries the error (`via`), which parameter
the partial stops at (`to`). The formula is item 4 below, once its symbols exist.

The frame also says what is *outside* it: a signal carrying a second-order term, the exact credit
of a bit flip, is not an adjoint. A learned transport is the adjoint of a network with learned
Jacobians; the rule of step 3 is a learned last hop in place of the product of adjoint variable
and local partial. And the frame is the *spatial* adjoint of one forward pass, depth being its
only axis. The online axis, a window of cases, an error that arrives later, the pool's episodes,
is the other side of the same duality: forward sensitivities carried along time (RTRL and its
approximations, e-prop's eligibility traces), with their own information and memory costs. A
reverse adjoint over depth does not by itself give an online rule over time; that object enters
at step 2 and gets its own name there. The code is laid out in the frame's order: the seed, the
local partials, the two backward transports, the last hop, the adjoints a `via` can choose, and
`compute`, which is the frame in one line.

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
table's expectation under it. One formula, both passes. Every symbol, and an XOR example carried
through, in [`signals-worked.md`](signals-worked.md).

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

**4. The gradient at one gate, three factors.** Write $e_g = \partial L / \partial r_g$ for the
error at *this* gate's output, whatever brought it there; the adjoint literature writes it
$\lambda_g$, and so does the code (`lams`). The chain rule through item 3 is

$$\frac{\partial L}{\partial z[a]}
= \frac{\partial L}{\partial r_g} \cdot \frac{\partial r_g}{\partial T[a]} \cdot \frac{\partial T[a]}{\partial z[a]}
= e_g \cdot P(a \mid u) \cdot \sigma'(z[a]).$$

This is the adjoint method's formula: $e_g$ is the adjoint variable at the gate, and
$P \cdot \sigma'$ is the gate's local partial to its parameter.

- On bits the middle factor is one-hot: only the entry the case addressed gets a gradient. On
  soft inputs every entry gets a share, in proportion to $P$.
- Only the first factor comes from elsewhere. $P$ and $\sigma'$ are local to the gate; $e_g$ is
  what has to travel, and *how* it travels is the `via` coordinate.
- Over a batch the first two factors are summed over cases and the third is shared. The signal
  that reaches each entry is therefore
  $$s[a] = \sum_b \lambda_{g,b}\, P_b(a \mid u_b) \;\big(\cdot\, \sigma'(z[a])\big),$$
  the **last hop** from the gate's error onto its parameters (`signals.last_hop`), item 7.

The symbols one by one, and the XOR gate's four gradients worked out, in
[`signals-worked.md`](signals-worked.md).

**5. The backward pass** (`signals.backward`): where $e_g$ comes from. At an output gate it is
the seed. One layer back, a line's error is the sum over every *pin* wired to it, each downstream
gate $h$ and the input $j$ of $h$ that reads the line, of $h$'s error times a *carry*, one number
per pin:

$$e_g = \sum_{(h,\,j)\,:\; w_{h,j} = g} e_h \cdot c_{h,j}.$$

$j$ is fixed by the wiring: it is which of $h$'s $k$ inputs carries $g$'s line, and a gate that
reads the same line on two pins contributes twice. The forward gathers each gate's inputs along
the wiring, `u[:, w]`; the backward pass scatters and adds along the same wiring,
`zeros.at[:, w].add(...)`, which is the transpose of the gather. It is one recursion, each
layer's errors from the next layer's, applying the transposed Jacobian of the forward pass from
the outputs down, as EventProp does for a spiking network with time in place of depth. Two
carries make two transports.

*The relay's carry is the gate's sensitivity*, how much $h$'s read moves with its input $j$: its
row of the Jacobian. The read is multilinear, so it is a difference of two reads at the other
inputs' current values, equivalently the table's difference under the address distribution:

$$c_{h,j} = \frac{\partial r_h}{\partial u_{h,j}} = r_h(u \,|\, u_j{=}1) - r_h(u \,|\, u_j{=}0) = \sum_a P(a \mid u)\,\big(T[a \,|\, u_j{=}1] - T[a \,|\, u_j{=}0]\big),$$

on bits whether flipping that input flips the output, in $\{-1, 0, 1\}$ (`signals.sensitivity`).
With it the backward pass is the chain rule, and it reproduces autodiff to float precision on
both passes (a test): `relay` is what a chip would run, `autodiff` how we check it.

*The uniform split's carry is one*, $c_{h,j} = 1$ for every pin, so the sum above loses its
weights and the error at a gate is the plain sum of the errors of the pins it feeds; unrolled to
the outputs, it is the residual at each output times the number of wiring paths to it,

$$e_g = \sum_{(h,\,j)\,:\; w_{h,j} = g} e_h \;=\; \sum_o R[g, o]\, e_o,$$

with $R$ the path counts of item 6. A carry of one is the sensitivity of a gate whose output is
the *sum* of its inputs, not of an OR: an OR's sensitivity to an input is 1 only when the other
inputs are 0, an AND's only when they are 1, and only the adder moves by one for any input at any
state (a table of the four on the worked page). So the *transport* is the exact transpose of a
different network, the same wiring with every gate an adder: value-blind, a fixed feedback. The
signal as a whole is not that network's gradient, since the seed and the address are the LUT
circuit's own; it is a transpose identity followed by an empirical fact about training. That is
feedback alignment shaped by the wiring (`uniform`). One hop of both, worked out on the XOR gate feeding a second one, in
[`signals-worked.md`](signals-worked.md).

**6. The broadcast** (`signals.broadcast`). No layers: the error at every hidden gate is the
seed through a fixed matrix, $e_g = \sum_o B[g, o]\, e_o$, with $B$ random ±1 drawn once from the
tile's shape and the identity at the output layer, whose gates read their own residual. Direct
feedback alignment (`direct`): what a bus and a coefficient per gate would compute.

*Reachability.* A gate can reach an output when a chain of wires leads from the gate to it. The
number of such chains is the wiring's **path count**, $R[g, o]$, and it is itself a layered
backward pass: the all-sums pass (carry one) seeded with one output at a time, so that each
seed's error at a gate is its number of paths to that output (`signals.path_counts`; a test
asserts that the broadcast with $B = R$ is the uniform split). Zero means the gate cannot reach the
output. `reachable` is the random bus with $B$ masked to where $R > 0$: the same coefficients,
silent for outputs a gate has no path to. It costs a gate one bit per output, and it matters
because feedback from an unreachable output is noise the gate cannot cancel, which the audit
found to be the whole difference between the bus and the wiring-shaped split. It is therefore
*not* direct feedback alignment, whose feedback knows nothing of the circuit: it knows the
wiring's reachability, structural knowledge that is static (the tables and activations never
enter it), fixed at configuration, and computable off-line from the netlist or on the fabric by
one backward pass of ones through the wires, once. Its closest relative is a reachability mask
on an approximate gradient (SnAp). The honest statement is that a fixed feedback on a bus needs
the wiring's reachability and nothing else; and when a rule decides the wiring, it decides that
support for free.

**7. The last hop** (`signals.last_hop`), item 4's sum, closes the frame: a signal is a choice of
adjoint, then the hop from the gate's error onto its parameters, $s[a] = \sum_b \lambda_{g,b}\, P_b(a \mid u_b)$,
times $\sigma'(z[a])$ when `to="logit"`. In code, `compute` is exactly that: the gate errors of
the adjoint `via` chooses, then the last hop. The adjoint
variables before the readout, $e_{g,b}$, one number per gate and case, are the **per-gate signal**:
what a wire or a bus carries, the best a gate can know before its own address and slope turn it
into a per-entry move; `signals.gate_errors` returns them for any `via` inside the frame, and
`to="gate"` is that signal broadcast to every entry, address-blind. Descent on it fails, as a
table whose entries all move together can only learn a bias (`probes/2026-09-08-per-gate-descent.py`);
its use is downstream: a rule that reads it with its own inputs must reconstruct the address
(step 3), and its correlation with each input line across cases is the router of step 6.

**8. Outside the frame: the flip credit** (`flip`). On the bits an entry does not move, it
flips, and the loss change of a flip has two terms per output it reaches, $\Delta_o \in \{-1,0,1\}$:

$$\Delta L = \sum_o \big( e_o\, \Delta_o + \tfrac{1}{2} |\Delta_o| \big).$$

The first is what the relay carries. The second is the cost of changing an output that was right,
invisible to any adjoint because a right output has zero residual, and on bits as large as the
first. It is carried by a second layered adjoint, the **reach**, the same recursion with
$|c|$ as the carry seeded by ones: the number of live paths from the gate to the outputs, which
is the number of outputs a flip changes wherever paths do not reconverge (exact in the upper
layers, wrong for a few percent of the pairs at the input layer). Where paths reconverge the
composed credit can err both ways: a line branching into two buffers that meet at an AND has zero
sensitivity along each path alone, so the credit sees nothing though flipping the line fixes the
output, and with both paths live it counts the one output twice
(`probes/2026-09-08-flip-credit-counterexample.py`, the corpus audit's example run against the
code). The signal is the relay's plus half the reach in the direction of the flip, $(1 - 2H[a])$:
exact for one flip where the wiring does not reconverge, not a gradient, and a bits quantity (no
soft cell).

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
| `relay` | a counter per entry (the logit's stand-in) | its own table at each flipped input: $k$ reads | a reverse channel per forward wire, carrying $\lambda$ times a sensitivity in $\{-1,0,1\}$: the sensitivity is two bits, $\lambda$ is not (a sum over fan-out and paths), summed again where a line fans out |
| `uniform` | the same counters | none | a reverse channel per wire, but one message per gate broadcast to all its sources, plus the fan-out sum |
| `direct` | the same counters, and a fixed ±1 coefficient per output | none | a bus of $n_{out}$ residual bits, no reverse wiring |
| `reachable` | the same, and one bit per output: reachable or not, structural knowledge fixed at configuration (from the netlist, or one backward pass of ones through the wires, once) | none | the same bus |
| `flip` | the same counters | $k$ reads | a reverse channel per wire carrying two numbers, the error and the reach |
| any, on the soft pass | probabilities rather than bits, for activations and tables: an analogue or stochastic representation | | as above |

The relay implies the transpose of the wiring, every forward connection with a way back, but no
duplicated tables: the sensitivity is computed downstream from the gate's own table and sent as a
value. On an FPGA that is a second routing network and an adder per line. The bus is the least
wiring, not universally the cheapest channel: it still needs distribution and timing of
$n_{out}$ residual bits and a coefficient per gate *and* output. It has the shape of a
neuromorphic three-factor rule: a global error, a local eligibility (the addressed entry), and
local state. The table costs the *signal*; the optimiser's state is not in it. Under Adam every
logit carries two moments, which a fabric would not; the smallest rule of step 2,
$\Delta = -\eta \cdot$ signal, carries none, and what replaces the floating logit is the workshop's
question.

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
  fixed and the gates drift to make it right (the agreement between the circuit's actual Jacobian
  and the feedback climbs from one half to 0.8–0.96 along training, and stays at one half under
  the relay). The sign of the carry does not matter: −1 everywhere, or a random fixed sign per
  edge, trains as well, and the gates drift to the sign they are given.
- **The random bus** trains worst of the blind transports (0.63–0.77 on the bits), and the audit
  found the one reason: it delivers feedback from outputs a gate cannot reach, noise the gate
  cannot cancel. Masked to the reachable outputs (`reachable`), the same random bus trains like
  the wiring-shaped split (0.82–0.95); the path counts with a random sign per gate and output
  train as well as the counts. **On the shapes and tasks tested, a fixed feedback trains if its
  support is the wiring's reachability; its signs can be anything fixed.** One four-layer shape,
  addition and two junta families, eight seeds: a strong determinant here, not a law.

So straight-through, which reached the target on the flat tile, is a signal for one hidden layer
on these tasks, where the exact relay is in fact the better bits signal; with two it has failed
here, and the recent depth results on straight-through elsewhere are the comparator to read
before saying more. At depth a bits
fabric has three signals that train, any fixed feedback on the reachability delivered by wires or
by a bus, and the flip credit, and the one autodiff would suggest is the one that does not. The soft pass on the chip, a
substrate that holds probabilities, keeps the exact signal; that is the question the second
substrate inherits, and the question step 2 inherits is which of these signals the workshop is
trained on.

## Where straight-through sits in the ladder

Read the factored gradient on the hard pass again: the error relayed through the rounded tables
to a gate's output, times the one-hot of the selected address, times $\sigma'$. That is the
decomposition *relay × address × slope* with the relay on the deployed pass, and straight-through
is the cell (`hard`, `autodiff`, `logit`) of the taxonomy, not a thing of its own.
