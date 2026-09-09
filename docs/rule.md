# The rule

The rule is what one gate does to its own table, given the signal at its entries. It is the
object the programme is about: a chip is to host it beside every gate, so whatever it reads must
be there, and whatever it writes is the gate's own configuration. Everything else in loom exists to
train it (`docs/meta.md`), to feed it (`docs/signals.md`) or to measure it.

## The smallest rule

`docs/signals.md` factors the gradient at an entry into three local pieces, the error at the
gate's output, the address distribution and the sigmoid's slope, multiplied. The smallest rule is
that product scaled by one number:

$$\Delta z[a] = -\eta \cdot s[a],$$

with $s$ the signal a chip delivers to the entry and $\eta$ the step size, the rule's only
parameter (`rule.Params`, one real). Applied at every entry of every gate (`rule.update`), it is
plain descent at a fixed rate, and `descent.descend` calls this same function: one step, shared by
the floor, the rule and any instrument that later wants to run a tile forward. What step 2 adds is
that $\eta$ is *learned*, from a start where it is too small to move anything, by the outer loop of
`docs/meta.md`. Step 3 replaces the product by a small function $g$ of the same three factors,
shared by every gate; later steps give $g$ a carry and a message. The rule grows in place.

## What the rule may read

Only what is an argument: the gate's table and the signal at its entries. Locality is structural,
nothing else can be reached from inside `update`. The signal's own locality is the `via`
coordinate's business, and its physical cost is tabulated there; the rule's cost is its parameters
and the state it keeps, which today is none beyond the table.

## The sign of η

$\eta = \exp(\text{raw})$, so it is never negative. This is not a convenience: with a free sign the
outer loop could turn a sign-flipped signal into a working rule by choosing $\eta < 0$, and the
flipped signal would be no control at all. Kept positive, the rule can only follow the direction it
is given, and the sign-flipped control asks exactly whether that direction is the right one.

## The bound

`update` takes an optional bound: the logit is then held within $\pm c$ after every step. On a
chip an entry is not a real number but a counter of a few bits, and the bound is what makes the
stored logit one: an entry at the bound takes no further credit in that direction, and $\eta$
against $c$ says how much of the counter one step moves. Unbounded, the logit is the floating
stand-in that steps 0 and 1 used. Which of the two the deployed rule needs, and what the counter's
depth costs, is measured on the bits in the second meta-learning chunk.
