"""Descent on the tables, driven by a signal: the floor every local rule is measured against.

Plain descent, Δ = −lr·s, nothing normalised and no state beside the tables: what step 2's
smallest rule is with η fixed by hand, so the floor is the rule's own baseline. Not a rule a chip
could host when the signal is the whole circuit's gradient; but the same loop driven by a signal a
chip can produce is already that rule, and step 2 meta-learns it.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import jax
import jax.numpy as jnp

from loom.rule import apply, state0
from loom.signals import REFERENCE, Signal, compute
from loom.tile import Tile, accuracy

RATE = 100.0  # the default step: the centre of the reference's plateau on the flat tile, batched


def descend(
    tile: Tile,
    x: jax.Array,
    y: jax.Array,
    *,
    lr: float = RATE,
    window: int | None = None,
    key: jax.Array | None = None,
    signal: Signal = REFERENCE,
    rule: str = "plain",
) -> Iterator[Tile]:
    """Descent on the table logits under a named rule of ``rule.RULES`` at a fixed rate, plain by
    default (``logits − lr·s``): one step, shared by the floor and the rule, with the rule's own
    state carried along; the wiring stays fixed.

    Yields the tile after every step, without end: the caller sets the budget. ``window`` is how
    many cases a step sees: all by default (the batched floor), or ``window`` cases drawn with
    replacement, as a deployed tile meets them in a stream (``window=1`` is fully online). A
    signal's magnitude now matters: the per-entry signal is the average vote per case times the
    address and, to the logit, σ′, so a rate is in logit units per unit vote, and a signal without
    σ′ wants a smaller one; which rate each signal wants is measured, never
    assumed (the default is the centre of the reference's plateau on the flat tile, batched:
    ``notes/2026-09-08-the-floors-under-plain-descent.md``).
    """

    @jax.jit
    def step(logits, state, key):
        idx = jnp.arange(len(x)) if window is None else jax.random.choice(key, len(x), (window,))
        t = Tile(logits, tile.wires)
        t, state = apply(rule, state, t, compute(signal, t, x[idx], y[idx]), lr)  # the rule's step
        return t.logits, state

    rng = jax.random.key(0) if key is None else key
    logits, state = tile.logits, state0(rule, tile)
    while True:
        rng, k = jax.random.split(rng)
        logits, state = step(logits, state, k)
        yield Tile(logits, tile.wires)


def fit(tile: Tile, x: jax.Array, y: jax.Array, steps: int = 500, **kw) -> Tile:
    """Run ``steps`` steps of :func:`descend` and return the last tile (every step is computed)."""
    *_, last = islice(descend(tile, x, y, **kw), steps)  # every step runs; only the last is kept
    return last


def trajectory(
    tile: Tile,
    x: jax.Array,
    y: jax.Array,
    steps: int = 500,
    every: int = 50,
    signal: Signal = REFERENCE,
    **kw,
) -> tuple[Tile, dict[str, jax.Array]]:
    """Fit while recording, every ``every`` steps, accuracy on the signal's pass and on the bits.

    Their difference is the deploy gap: identically zero when the signal is computed on the bits
    (the training view is the deployed circuit), closing only at saturation on the soft pass.
    """
    rec: dict[str, list] = {"step": [], "train": [], "hard": []}
    t = tile
    for i, t in enumerate(islice(descend(tile, x, y, signal=signal, **kw), steps), start=1):
        if i % every == 0 or i == steps:
            rec["step"].append(i)
            rec["train"].append(accuracy(t, x, y, signal.on))
            rec["hard"].append(accuracy(t, x, y, "hard"))
    return t, {k: jnp.asarray(v) for k, v in rec.items()}
