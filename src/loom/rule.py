"""The rule: what one gate does to its own table, given the signal at its entries.

The smallest rule is the gradient's three factors multiplied and scaled by one number,
Δ = −η · s: the signal ``s`` a chip delivers to each entry (``docs/signals.md``) and the step size
η, the only parameter. Step 2 meta-learns η from a non-functional start; step 3 grows the product
into a small shared function of the factors. The rule is the object the programme is about: a
chip hosts it, so it reads only its own table and the signal at its entries, and locality is
structural, nothing else is an argument. Plain descent at a fixed η is this same step.

Re-derived 2026-09 from blastema/training/rule_step.py (apply_deltas), in the smallest form.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from loom.tile import Tile


class Params(NamedTuple):
    raw: jax.Array  # log η: the step size as a free real, so that η = exp(raw) is never negative


def init(eta: float = 1e-2) -> Params:
    """The rule at a step size: non-functional by default, small enough that nothing moves."""
    return Params(jnp.log(jnp.asarray(eta, dtype=jnp.float32)))


def rate(params: Params) -> jax.Array:
    """η = exp(raw) ≥ 0: a sign-flipped signal can never be undone by a negative step, which is
    what makes the sign-flipped control a control."""
    return jnp.exp(params.raw)


def update(eta: jax.Array | float, tile: Tile, s: tuple[jax.Array, ...], clip: float | None = None):
    """One step of the smallest rule at every logit: z ← z − η · s, held within ±clip when given.

    The bound makes the stored logit a finite counter: an entry at the bound takes no further
    credit in that direction (its derivative there is zero), which is what a chip storing a few
    bits per entry does. Unbounded, the logit is the floating stand-in of steps 0 and 1.
    """
    logits = tuple(z - eta * g for z, g in zip(tile.logits, s, strict=True))
    if clip is not None:
        logits = tuple(jnp.clip(z, -clip, clip) for z in logits)
    return Tile(logits, tile.wires)
