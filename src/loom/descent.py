"""Descent on the tables, driven by a signal: the floor every local rule is measured against.

Not a rule a chip could host when the signal is the whole circuit's gradient; but the same loop
driven by a signal a chip can produce is already the smallest rule, and step 2 meta-learns it.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from loom.signals import gradient
from loom.tile import Tile

Signal = Callable[[Tile, jax.Array, jax.Array, str], tuple[jax.Array, ...]]


def fit(
    tile: Tile,
    x: jax.Array,
    y: jax.Array,
    steps: int = 500,
    lr: float = 0.1,
    window: int | None = None,
    key: jax.Array | None = None,
    signal: Signal = gradient,
    mode: str = "soft",
) -> Tile:
    """Adam on the table logits, fed a ``signal`` in place of the gradient; the wiring stays fixed.

    ``signal(tile, x, y, mode)`` returns per-logit arrays; the default is the true gradient through
    the read ``mode`` (``soft``, or ``ste`` to train the deployed circuit directly). ``window`` is
    how many cases a step sees: all by default (the batched floor), or a random window from the
    stream of cases as a deployed tile would see them (``window=1`` is fully online). The
    straight-through read wants a smaller step than the soft one (bits chatter at the soft rate).
    """
    opt = optax.adam(lr)
    state = opt.init(tile.logits)

    @jax.jit
    def step(logits, state, key):
        idx = jnp.arange(len(x)) if window is None else jax.random.choice(key, len(x), (window,))
        grads = signal(Tile(logits, tile.wires), x[idx], y[idx], mode)
        updates, state = opt.update(grads, state, logits)
        return optax.apply_updates(logits, updates), state

    logits = tile.logits
    for k in jax.random.split(jax.random.key(0) if key is None else key, steps):
        logits, state = step(logits, state, k)
    return Tile(logits, tile.wires)
