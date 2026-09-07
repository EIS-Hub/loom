# Descent

Direct descent on the tables is the floor: not a rule a chip could host when the signal is the
whole circuit's gradient, but the answer to the first question of any substrate, could it train at
all. The same loop driven by a signal a chip can produce is already the smallest rule, which is
what step 2 meta-learns.

## Three functions, one loop

- `descend(tile, x, y, *, lr, window, key, signal)` yields the tile after every step, without
  end; the caller sets the budget. Each step asks the signal for its per-logit arrays and hands
  them to Adam as if they were gradients. The wiring never moves.
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
for that with a touchier landscape and wants a smaller step (`docs/signals.md`).

## Why Adam, and what is not decided here

Adam is the floor's optimiser because it is the standard one and needs no tuning to reach a
target on a small tile; the step sizes and budgets that make a claim are not here but in named
recipes (`docs/recipes.md`). Step 2 replaces Adam by the smallest rule, Δ = −η·signal, and
meta-learns η.
