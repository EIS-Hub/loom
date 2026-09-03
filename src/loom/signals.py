"""Signals: what a local update may read, as functions from (tile, cases) to per-logit arrays.

Every signal has the shape of ``tile.logits``, so descent and the rule consume them alike. The spine
of what a local update may read: nothing (the memoriser, step 3's null), task identity (parked),
error. This module starts the error signals: the residual at the outputs, and the reference
transport, the true gradient through a chosen read mode. The next chunk adds the transports a chip
could run: the relay through each gate's own table, and the value-blind adjoint.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from loom.tile import Tile, forward


def loss(tile: Tile, x: jax.Array, y: jax.Array, mode: str = "soft") -> jax.Array:
    """Half the squared error of the read against the demanded bits, over cases and output bits.

    Its derivative at an output is the residual itself, the error a chip can see; the same loss
    serves every read mode, where a cross-entropy would blow up on bits.
    """
    return 0.5 * jnp.mean((forward(tile, x, mode) - y) ** 2)


def residual(tile: Tile, x: jax.Array, y: jax.Array, mode: str = "hard") -> jax.Array:
    """The error at the outputs, read minus demanded: [B, n_out]; bits in {-1, 0, 1} when hard."""
    return forward(tile, x, mode) - y


def gradient(tile: Tile, x: jax.Array, y: jax.Array, mode: str = "soft") -> tuple[jax.Array, ...]:
    """The reference signal: the loss's gradient at every logit, through the read ``mode``."""
    return jax.grad(lambda lg: loss(Tile(lg, tile.wires), x, y, mode))(tile.logits)
