# Meta-learning

## The words, each with one meaning

The method has one state, two reads of it, two losses of it, and one objective, and the same word
is not allowed to name two of them. A pass qualifies a signal, never a state or a rule: "the rule
on the hard pass" means the signal it is fed is computed on bits. "Loss" appears only as
$L^{\text{soft}}$, $L^{\text{hard}}$ or $L^{\text{ste}}$ of a named state; "the loss" alone is not
a phrase.

| term | meaning | in code | notation |
|---|---|---|---|
| **state** | the tile's logits, one real per entry; on a chip a bounded counter. A state is neither soft nor hard: the read is. | `Tile.logits` | $z$ |
| **read** | the tables as a function of the state: the probabilities $\sigma(z)$, or the bits $\mathrm{round}(\sigma(z))$. One state, two reads. | `tile.tables(z, mode)` | $T = \sigma(z)$, $H = \mathrm{round}(T)$ |
| **pass** | a forward with one read applied throughout, so its activations are probabilities or bits. A signal's `on` coordinate names the pass and nothing else. | `activations(tile, x, mode)` | soft / hard |
| **residual** | the error at the outputs on a pass, read minus demanded, over the number of cases. It is $\partial\ell/\partial r$ at the read on *both* passes: a real in $(-1, 1)$ on the soft pass, a bit in $\{-1, 0, 1\}$ on the hard pass. The seed of every signal. | `signals.seed` | $e$ |
| **signal to the entry** | $\partial L/\partial T$ on the pass: the gradient with respect to the table entries at that pass's point, carried by a `via` and hopped onto the entries by the address. Exact on both passes, since the read is multilinear (a test in `test_signals`); on the hard pass the first-order credit of a flip, nonzero only at addressed entries. | `Signal(on, via, "entry")` | $s$ |
| **signal to the logit** | the signal to the entry times $\sigma'(z)$, the hop from entry to logit. A gradient on the soft pass; on the hard pass a surrogate, the chip having no $z$: the straight-through cells. | `Signal(on, via, "logit")` | $s \cdot \sigma'$ |
| **inner loop** | the rule at work, $z \leftarrow z - \eta\, s$ on a pass, $K$ steps. It consumes a signal and never evaluates a loss; at deployment it is all there is. | `meta.rollout` | $z_{t+1} = z_t - \eta\, s(z_t)$ |
| **objective** | the one function the outer loop differentiates: the soft loss of the state the inner loop ends on. Never used for anything else. | `rollout(...)[0]`, `meta.objective` | $J = L^{\text{soft}}(z_K)$ |
| **outer loop** | the training of the rule, on the host, once: the objective's gradient through the $K$ steps, then an optimiser. Nothing of it is on the chip or in the cost table, so a surrogate here is free. | `meta.step`, `meta.learn` | $\partial J / \partial \theta$ |
| **first-order** | the outer gradient with the signal treated as data: credit runs through $\eta$'s linear entry in every step and not through $\partial s/\partial z$. Exact for the `entry` cells of the hard pass, where $\partial s/\partial z = 0$; an approximation on the soft pass and on the straight-through cells. Nothing about trajectories. | `first_order=True` | $\langle \nabla L^{\text{soft}}(z_K),\, -\sum_t s_t \rangle$ |
| **deployed loss, deployed accuracy** | $L^{\text{hard}}$ and $A^{\text{hard}}$ of a state: the numbers a claim is worded on, measured on a fresh tile and a held-out task after plain steps of the rule at the $\eta$ found. | `signals.loss(·, "hard")`, `tile.accuracy(·, "hard")`, `recipes.adapt` | $L^{\text{hard}}(z)$, $A^{\text{hard}}(z)$ |
| **deploy gap** | $L^{\text{hard}}(z) - L^{\text{soft}}(z)$ of one state, recorded beside the objective at every outer step and never differentiated. A rule trained on one signal and deployed with another is not a gap but a different condition, and gets no name. | the third of `learn`'s yields | |
| **straight-through (STE)** | a derivative substitution only: hard in value, $\sigma'$ in the backward, and always with its location. In the signal it is the `logit` cells on the hard pass, and costs the chip a stored logit; in the objective, $L^{\text{ste}}(z_K)$, it is host-side and free (the fallback if the deploy gap bites; not the objective today). | `signals.straight_through` | $L^{\text{ste}}$ |
| **cell** | one signal, named by its three coordinates. | `signals.CELLS`, `Signal.label` | `hard.uniform.entry` |
| **recipe** | a training condition as data: the inner cell, the outer objective, the window, $K$, the batch, the outer schedule. Its label is the pair inner → outer. | `MetaRecipe`, `.label` | `hard.uniform.entry -> L^soft` |

## Two loops

The **inner** loop runs the rule for $K$ steps on a tile, from some state, fed the signal a chip
would have: that is the rule at work, and at deployment it is all there is. The **outer** loop
trains the rule: it takes the objective, differentiates it with respect to the rule's parameters
through those $K$ steps, and moves the parameters. It runs on the host, once, before the rule
ships; nothing of it is on the chip, and its cost is not in the ledger. `meta.rollout` is the inner
loop in one scan; `meta.step` is one outer step; `meta.learn` runs outer steps without end, the
caller setting the budget, as `descent.descend` does.

## The objective, declared

$$J(\theta) = L^{\text{soft}}\big(z_K(\theta)\big), \qquad z_{t+1} = z_t - \eta(\theta)\, s^{\text{on},\text{via},\text{to}}(z_t, \text{cases}_t).$$

Every one of the $K$ updates is credited equally, through the trajectory, for where the tile
ended. The pass rides on the signal, never on the state: a rule fed a hard-pass signal descends
the same soft objective, because the objective is the host's and the signal is the chip's. Two
alternatives were considered and are not the objective. The deployed loss cannot be: rounding is
flat, so it has no derivative to anything; it is *recorded* beside the objective at every outer
step, so the deploy gap is a column, and its straight-through version is the fallback if that
column ever opens. The mean of the losses along the way (the loss a deployed tile experiences,
step by step) is returned by the rollout as a record, but it is not the objective, because it
weights the steps: an early update is credited through every later loss and the last through
none, which favours whatever descends fastest at the start.

## The derivative path, per cell

The outer gradient has two parts. $\eta$ enters every update linearly, and that part is always
there: the first-order term, the soft gradient at the end against the sum of the steps taken,
$\langle \nabla L^{\text{soft}}(z_K),\, -\sum_t s_t \rangle$. The signal $s_t$ also depends on the
state it was computed at, and whether the derivative runs through it depends on the cell:

- on the **soft pass** it does, and the term is a Hessian: measured at a tenth of the total at
  $\eta = 3$ and half at $\eta = 30$ on the flat tile, sign-changing past the optimum;
- on the **`entry` cells of the hard pass** it does not: the signal is computed from rounded tables
  and bit activations, every derivative through it is zero by construction, and the outer gradient
  is exactly the first-order term (a test; the full and the first-order gradient agree to the last
  digit on `hard.uniform.entry`, `hard.relay.entry` and `hard.flip.entry`);
- on the **`logit` cells of the hard pass** it does, through $\sigma'(z)$ of the stored logit and,
  for `hard.autodiff.logit`, the straight-through tables: a surrogate of a surrogate, a factor of
  nine off the first-order term at $\eta = 30$ on the flat tile, with nothing to check it against.

`first_order=True` treats the signal as data on any cell. It is the **default on the hard pass**:
a no-op on the `entry` cells and the honest choice on the `logit` cells. On the soft pass the
default is the full gradient, and first order is the fair comparison. The rollout is truncated at
$K$: whatever state it starts from, credit runs through these $K$ steps and no further. Once the
rule reads its own stored value (step 3), $\partial\,\text{rule}/\partial z$ is nonzero on every
cell and a trajectory term returns on the hard pass too; "first-order" keeps meaning the signal as
data, and nothing more.

## What a rollout starts from

Fresh tiles on fresh tasks, each state carrying its own task (`meta.members`), a batch of them per
outer step. This trains the rule on the beginning of every life and never on its middle or its end.
The pool of the next chunk keeps states across outer steps, at every age; an old state is then an
*input* to the outer step, never a longer horizon. The window is the rollout's, as in
`docs/descent.md`: every case per step, or a few drawn as a deployed tile meets them. The window
decides whether there is anything to find: with every case per step the exact soft signal cannot
overshoot what a sixteen-entry table absorbs, the objective is flat in $\eta$ over decades, and a
found $\eta$ says little; online (one case per step) the objective has an interior optimum, and
finding it is the claim (the finding).

## The controls

A rule that trains under the true signal says little until three controls have failed to. The
**sign-flipped** signal, $-s$: since $\eta$ cannot go negative (`docs/rule.md`), the outer loop must
fail to recover a rule; this tests the *direction* of the signal. The **shuffled** signal, the same
numbers permuted within each layer at every step: an entry then receives a magnitude that has
nothing to do with it, and the outer loop may find any $\eta$ it likes, but nothing should train;
this tests the *information* in the signal, and its claim is worded on the outcome, not on where
$\eta$ lands, because a permutation keeps each layer's mean. The **output-only** signal, the hidden
layers silenced: only the output gates, which read their own residual, move; this is the baseline
any claim about the hidden layers must beat. The controls act on the signal arrays inside the
rollout (`meta.controlled`) and are never signal cells: `signals.CELLS` is what a chip can compute.
Each control has its own landscape in $\eta$, and the loop is expected to find *that* landscape's
minimum: under the flip it is at zero, under the information controls it is wherever nothing moves.

## Why the outer loop keeps Adam, and how fast it climbs

Steps 0 and 1 made plain descent the floor inside the tile, so that no claim about a signal leans
on an optimiser's normalisation. The outer loop is a different place: it is offline, on the host,
and the object it moves is the rule's parameter, not a table. With $\eta = \exp(\text{raw})$ the
gradient on raw is $\eta$ times the gradient on $\eta$, which vanishes exactly at the
non-functional start the check demands; a step whose size does not vanish with it is needed, and
Adam's normalisation gives one, a constant factor on $\eta$ per step. The gradient's global norm is
clipped as well. What is learned is the same whatever moves it; how fast is the host's affair, with
one limit: the climb from the start is geometric, and a factor per step that reaches the optimum
in a few dozen steps carries momentum past it, where the outer gradient is noisy and a seed can run
away (one in ten at 0.1 on the flat tile online). The recipe climbs at 0.03.

## What is measured, and what is read

Measured: whether the outer loop finds the step size the landscape's own optimum names, online,
from a non-functional start under the true signal; whether it walks $\eta$ down under the flipped
one and parks it where nothing moves under the shuffled and the output-only ones; whether the
$\eta$ found adapts a fresh tile on a held-out task; which cells carry a second-order term. Read:
that at first order the check measures the cosine of the accumulated steps with the soft gradient
at the end, which is why a sign-consistent signal is found and a shuffled one is not. What $\eta$
means on the bits, with the stored logit bounded, is the next chunk's question.
