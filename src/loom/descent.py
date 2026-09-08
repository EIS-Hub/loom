"""Descent on the tables, driven by a signal: the floor every local rule is measured against.

Not a rule a chip could host when the signal is the whole circuit's gradient; but the same loop
driven by a signal a chip can produce is already the smallest rule, and step 2 meta-learns it.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import jax
import jax.numpy as jnp
import optax

from loom.signals import REFERENCE, Signal, compute
from loom.tile import Tile, accuracy


def descend(
    tile: Tile,
    x: jax.Array,
    y: jax.Array,
    *,
    lr: float = 0.1,
    window: int | None = None,
    key: jax.Array | None = None,
    signal: Signal = REFERENCE,
) -> Iterator[Tile]:
    """Adam on the table logits, fed the signal's per-logit arrays; the wiring stays fixed.

    Yields the tile after every step, without end: the caller sets the budget. ``window`` is how
    many cases a step sees: all by default (the batched floor), or ``window`` cases drawn with
    replacement, as a deployed tile meets them in a stream (``window=1`` is fully online). Descent
    on the bits (``Signal("hard")``) wants a smaller step than on the soft pass: bits chatter at the
    soft rate.
    """
    opt = optax.adam(lr)
    state = opt.init(tile.logits)

    @jax.jit
    def step(logits, state, key):
        idx = jnp.arange(len(x)) if window is None else jax.random.choice(key, len(x), (window,))
        grads = compute(signal, Tile(logits, tile.wires), x[idx], y[idx])
        updates, state = opt.update(grads, state, logits)
        return optax.apply_updates(logits, updates), state

    rng = jax.random.key(0) if key is None else key
    logits = tile.logits
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
