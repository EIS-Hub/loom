"""Direct descent on the tables: the floor every local rule is measured against.

Not a rule a chip could host (it reads the whole circuit's gradient), but the answer to the first
question of any substrate: could it train at all, and could it do so online, one case at a time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from loom.tile import Tile, forward


def bce(tile: Tile, x: jax.Array, y: jax.Array) -> jax.Array:
    """Binary cross-entropy of the soft read against the demanded bits, per output bit, averaged."""
    p = jnp.clip(forward(tile, x), 1e-6, 1 - 1e-6)
    return -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))


def fit(
    tile: Tile,
    x: jax.Array,
    y: jax.Array,
    steps: int = 500,
    lr: float = 0.1,
    window: int | None = None,
    key: jax.Array | None = None,
) -> Tile:
    """Adam on the table logits; the wiring stays fixed. Returns the fitted tile.

    ``window`` is how many cases a step sees: all of them by default (the batched floor), or a
    random window from the stream of cases, as a deployed tile would see them (``window=1`` is
    fully online: predict on one case, adapt, next case).
    """
    opt = optax.adam(lr)
    state = opt.init(tile.logits)

    @jax.jit
    def step(logits, state, key):
        idx = jnp.arange(len(x)) if window is None else jax.random.choice(key, len(x), (window,))
        loss_fn = lambda lg: bce(Tile(lg, tile.wires), x[idx], y[idx])  # noqa: E731
        loss, grads = jax.value_and_grad(loss_fn)(logits)
        updates, state = opt.update(grads, state, logits)
        return optax.apply_updates(logits, updates), state, loss

    logits = tile.logits
    for k in jax.random.split(jax.random.key(0) if key is None else key, steps):
        logits, state, _ = step(logits, state, k)
    return Tile(logits, tile.wires)
