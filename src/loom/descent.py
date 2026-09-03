"""Direct descent on the tables: the floor every local rule is measured against.

Not a rule a chip could host (it reads the whole circuit's gradient), but the answer to the first
question of any substrate: could it train at all.
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


def fit(tile: Tile, x: jax.Array, y: jax.Array, steps: int = 500, lr: float = 0.1) -> Tile:
    """Adam on the table logits; the wiring stays fixed. Returns the fitted tile."""
    opt = optax.adam(lr)
    state = opt.init(tile.logits)

    @jax.jit
    def step(logits, state):
        loss, grads = jax.value_and_grad(lambda lg: bce(Tile(lg, tile.wires), x, y))(logits)
        updates, state = opt.update(grads, state, logits)
        return optax.apply_updates(logits, updates), state, loss

    logits = tile.logits
    for _ in range(steps):
        logits, state, _ = step(logits, state)
    return Tile(logits, tile.wires)
