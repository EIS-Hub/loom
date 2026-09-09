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
nothing else can be reached from inside `update`. The table being an argument, the rule may read
its own logits. The smallest rule reads an entry's value only to add to it and to hold it at the
bound; the signal already carries one read of it, the slope $\sigma'(z[a])$ that `to = logit`
keeps and `to = entry` drops. Step 3's $g$ may take the entry's value as a fourth factor, at the
cost of one local read: on a bounded counter that is how a rule would know how far an entry sits
from its flip, and whether $g$ needs it is step 3's question. The signal's own locality is the
`via` coordinate's business, and its physical cost is tabulated there; the rule's cost is its
parameters and the state it keeps, which today is none beyond the table.

## The sign of η

$\eta = \exp(\text{raw})$, so it is never negative. This is not a convenience: with a free sign the
outer loop could turn a sign-flipped signal into a working rule by choosing $\eta < 0$, and the
flipped signal would be no control at all. Kept positive, the rule can only follow the direction it
is given, and the sign-flipped control asks exactly whether that direction is the right one.

## The bound

`update` takes an optional bound: the logit is then held within $\pm c$ after every step. On a
chip an entry is not a real number but a counter of a few bits, and the bound is what makes the
stored logit one, $c / \eta$ votes deep from rail to rail. A vote that pushes an entry past a rail
is lost, as a saturated counter drops an increment; a vote back inside moves it again, so a rail
silences credit in one direction only. The clip's derivative is zero on a lost vote: the outer loop
earns nothing from a step that changed nothing, and cannot raise $\eta$ by saturating. Unbounded,
the logit is the floating stand-in that steps 0 and 1 used, a counter of unlimited depth that only
its sign is ever read from on the bits. Which of the two the deployed rule needs, and what the
counter's depth costs, is measured on the bits in the second meta-learning chunk.
