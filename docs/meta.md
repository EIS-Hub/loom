# Meta-learning

Two loops. The **inner** loop runs the rule for $K$ steps on a tile, from some state, fed the
signal a chip would have: that is the rule at work, and at deployment it is all there is. The
**outer** loop trains the rule: it takes the loss of the tile the rule ends on, differentiates it
with respect to the rule's parameters through those $K$ steps, and moves the parameters. The outer
loop runs on the host, once, before the rule ships; nothing of it is on the chip, and its cost is
not in the ledger. `meta.rollout` is the inner loop in one scan; `meta.step` is one outer step;
`meta.learn` runs outer steps without end, the caller setting the budget, as `descent.descend` does.

## The objective, declared

The objective is the soft loss, on every case, of the tile the rule ends on after $K$ steps:

$$J(\theta) = L_{\text{soft}}\big(z_K(\theta)\big), \qquad z_{t+1} = z_t - \eta(\theta)\, s_t(z_t).$$

Every one of the $K$ updates is credited equally, through the trajectory, for where the tile
ended. Two alternatives were considered and are not the objective. The loss on the bits cannot be:
rounding is flat, so it has no derivative to anything. The mean of the losses along the way (the
loss a deployed tile experiences, step by step) is returned by the rollout as a *record*, but it
is not the objective, because it weights the steps: an early update is credited through every
later loss and the last through none, which favours whatever descends fastest at the start.

The derivative path has two parts. $\eta$ enters every update linearly, and that part is always
there. The signal $s_t$ also depends on the tile, and on the soft pass the derivative runs through
it too, a second-order term. On the bits it does not: the signal is computed from rounded tables
and bit activations, every derivative through it is zero by construction, and the meta-gradient is
exactly the first-order term, the soft gradient at the end against the sum of the steps taken,
$\langle \nabla L_{\text{soft}}(z_K),\, -\sum_t s_t \rangle$. `first_order=True` makes that explicit
on any pass by treating the signal as data: on the bits a no-op (a test), on the soft pass the fair
comparison. The rollout is truncated at $K$: whatever state it starts from, credit runs through
these $K$ steps and no further.

## What a rollout starts from

Fresh tiles on fresh tasks, each state carrying its own task (`meta.members`), a batch of them per
outer step. This trains the rule on the beginning of every life and never on its middle or its end.
The pool of the next chunk keeps states across outer steps, at every age; an old state is then an
*input* to the outer step, never a longer horizon. The window is the rollout's, as in
`docs/descent.md`: every case per step, or a few drawn as a deployed tile meets them.

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

## Why the outer loop keeps Adam

Steps 0 and 1 made plain descent the floor inside the tile, so that no claim about a signal leans
on an optimiser's normalisation. The outer loop is a different place: it is offline, on the host,
and the object it moves is the rule's parameter, not a table. With $\eta = \exp(\text{raw})$ the
gradient on raw is $\eta$ times the gradient on $\eta$, which vanishes exactly at the
non-functional start the check demands; a step whose size does not vanish with it is needed, and
Adam's normalisation gives one, a constant factor on $\eta$ per step. The gradient's global norm is
clipped as well. What is learned is the same whatever moves it; how fast is the host's affair.

## What is measured, and what is read

Measured: whether the outer loop finds a step size from a non-functional start under the true
signal, whether it fails to under the flipped one, whether nothing trains under the shuffled and
the output-only ones, and whether the $\eta$ found adapts a fresh tile on a held-out task. Read:
that on the soft pass the $\eta$ found is a step size, with an interior optimum that depends on the
window, and that at first order the check measures the cosine of the accumulated steps with the
soft gradient at the end. What $\eta$ means on the bits, where there may be no optimum to find, is
the next chunk's question.
