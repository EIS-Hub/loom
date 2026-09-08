# Descent

Direct descent on the tables is the floor: not a rule a chip could host when the signal is the
whole circuit's gradient, but the answer to the first question of any substrate, could it train at
all. The same loop driven by a signal a chip can produce is already the smallest rule, which is
what step 2 meta-learns.

## Three functions, one loop

- `descend(tile, x, y, *, lr, window, key, signal)` yields the tile after every step, without
  end; the caller sets the budget. Each step asks the signal for its per-logit arrays and
  subtracts them, scaled by the rate: Δ = −lr·s, nothing normalised, no state beside the tables.
  The wiring never moves.
- `fit(..., steps)` returns the tile after that many steps.
- `trajectory(..., steps, every, signal)` fits while recording, every so many steps, the accuracy
  on the signal's own pass and on the bits.

## The window

`window` is how many cases a step sees: all of them when `window=None`, the batched default; or a random
window drawn from the stream of cases, as a deployed tile would see them. `window=1` is fully
online: predict on one case, adapt, next case. Batched and online are one function with one
argument, not two code paths, so the window can become an axis of a recipe.

## The deploy gap

A tile's accuracy on its training view minus its accuracy on the bits. Descent on the soft pass
runs ahead of its deployed accuracy until the tables saturate, then the gap closes. Descent on the
bits has no gap at any step, by construction: its training view *is* the deployed circuit. It pays
for that with a touchier landscape (`docs/signals.md`); which step it wants is measured, not
assumed.

## Why plain descent

The floor is the plain update, Δ = −lr·s, for three reasons, each a way the standard optimiser
had been saying something about signals that was not true of them. First, an adaptive optimiser
normalises every coordinate, so a signal's magnitude, and with it the σ′ factor that separates
the partial to the logit from the partial to the entry, never entered any statement about which
signal descent can follow; under the plain update the magnitude is the step, and each signal
has a rate it wants, measured rather than assumed. Second, it carries two moments per logit, a
memory the cost table of `docs/signals.md` explicitly leaves out of a signal's price: a floor
with hidden state is not the floor of a rule without it. Third, on a window of one case it
amplifies the entries a case rarely addresses, whose second moment decays toward zero, and its
momentum carries one case's vote into the cases that follow, so the online floor was not the
online regime. Step 2's rule is exactly −η·s with η learned, so the plain update is the only
honest baseline from here on, and the standard optimiser never enters a signal claim again. The
rates and budgets that make a claim are not here but in named recipes (`docs/recipes.md`); the
numbers behind the change are in `notes/2026-09-08-the-floors-under-plain-descent.md`.
