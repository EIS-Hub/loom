"""The rule: what one gate does to its own table, given the signal at its entries.

The smallest rule is the gradient's three factors multiplied and scaled by one number,
Δ = −η · s: the signal ``s`` a chip delivers to each entry (``docs/signals.md``) and the step size
η, the only parameter. Step 2 meta-learns η from a non-functional start; step 3 grows the product
into a small shared function of the factors. The rule is the object the programme is about: a
chip hosts it, so it reads only its own table and the signal at its entries, and locality is
structural, nothing else is an argument. Plain descent at a fixed η is this same step.

Beside it, the hand-engineered family: the few transformations every optimiser in the literature
composes (a scale, a sign, a running first or second moment), written as pure functions with their
state explicit, so that what each costs a cell (accumulators per entry, a root and a division or
not) is read off the code. They are what a learned rule must beat, at equal memory.

Re-derived 2026-09 from blastema/training/rule_step.py (apply_deltas), in the smallest form; the
family checked against optax's implementations once (a test) and owned here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from loom.tile import Tile

Arrays = tuple[jax.Array, ...]  # one array per layer, the logits' shape
Rule = Callable[
    [tuple, Arrays, jax.Array | float], tuple[tuple, Arrays]
]  # (state, s, η) → (state, Δ)


class Params(NamedTuple):
    raw: jax.Array  # log η: the step size as a free real, so that η = exp(raw) is never negative


def init(eta: float = 1e-2) -> Params:
    """The rule at a step size: non-functional by default, small enough that nothing moves."""
    return Params(jnp.log(jnp.asarray(eta, dtype=jnp.float32)))


def rate(params: Params) -> jax.Array:
    """η = exp(raw) ≥ 0: a sign-flipped signal can never be undone by a negative step, which is
    what makes the sign-flipped control a control."""
    return jnp.exp(params.raw)


# The primitives: what a cell can do to the signal before it moves the entry.


def ema(beta: float, state: Arrays, s: Arrays) -> Arrays:
    """A running mean with memory ``beta``: one accumulator per entry."""
    return tuple(beta * a + (1 - beta) * g for a, g in zip(state, s, strict=True))


def trace(beta: float, state: Arrays, s: Arrays) -> Arrays:
    """A running sum with memory ``beta`` (the momentum of the literature): one accumulator."""
    return tuple(beta * a + g for a, g in zip(state, s, strict=True))


# The compositions, each (state, s, η) → (state, Δ): the optimisers of the literature as pipelines.


def plain(state, s, eta):
    """Δ = −η · s: the smallest rule, no state, multiply only."""
    return state, tuple(-eta * g for g in s)


def sign(state, s, eta):
    """Δ = −η · sign(s): the vote's direction only; no state, a comparison; erases every
    magnitude, so the step is the same at every depth."""
    return state, tuple(-eta * jnp.sign(g) for g in s)


def momentum(state, s, eta, beta=0.9):
    """A running sum of the signal, then the plain step: one accumulator, multiply-add."""
    m = trace(beta, state[0], s)
    return (m,), tuple(-eta * a for a in m)


def rmsprop(state, s, eta, beta=0.99, eps=1e-8):
    """The signal over the root of its running square (a memory of a hundred steps): one
    accumulator, a root and a division; a gain per entry that follows the signal's own size."""
    v = ema(beta, state[0], tuple(g * g for g in s))
    return (v,), tuple(-eta * g / jnp.sqrt(a + eps) for a, g in zip(v, s, strict=True))


def lion(state, s, eta, b1=0.9, b2=0.99):
    """The sign of a running mean interpolated with the signal: one accumulator, compare-and-add;
    a leaky vote counter with a sign readout."""
    c = ema(b1, state[0], s)
    return (ema(b2, state[0], s),), tuple(-eta * jnp.sign(a) for a in c)


def adam(state, s, eta, b1=0.9, b2=0.999, eps=1e-8):
    """The running mean over the root of the running square, both corrected for their start: two
    accumulators, a root and a division; the instrument, and the ceiling to beat."""
    m, v, t = state
    t = t + 1
    m, v = ema(b1, m, s), ema(b2, v, tuple(g * g for g in s))
    mh = tuple(a / (1 - b1**t) for a in m)
    vh = tuple(a / (1 - b2**t) for a in v)
    return (m, v, t), tuple(-eta * a / (jnp.sqrt(b) + eps) for a, b in zip(mh, vh, strict=True))


RULES: dict[str, Rule] = {
    "plain": plain,
    "sign": sign,
    "momentum": momentum,
    "rmsprop": rmsprop,
    "lion": lion,
    "adam": adam,
}
STATE = {"plain": 0, "sign": 0, "momentum": 1, "rmsprop": 1, "lion": 1, "adam": 2}  # per entry


def state0(name: str, tile: Tile) -> tuple:
    """A rule's state at rest: its accumulators at zero, and Adam's step count."""
    zeros = tuple(jnp.zeros_like(z) for z in tile.logits)
    if name == "adam":
        return (zeros, zeros, jnp.zeros(()))
    return tuple(zeros for _ in range(STATE[name]))


def apply(name: str, state: tuple, tile: Tile, s: Arrays, eta, clip: float | None = None):
    """One step of a named rule at every logit, held within ±clip when given: (tile, state).

    The bound makes the stored logit a finite counter, clip/η votes deep from rail to rail. A
    vote that pushes an entry past a rail is lost, as a saturated counter drops an increment; a
    vote back inside moves it again. The clip's derivative is zero on a lost vote, so the outer
    loop earns no credit for a step that changed nothing and cannot learn by saturating.
    Unbounded, the logit is the floating stand-in of steps 0 and 1, with unlimited memory.
    """
    state, delta = RULES[name](state, s, eta)
    logits = tuple(z + d for z, d in zip(tile.logits, delta, strict=True))
    if clip is not None:
        logits = tuple(jnp.clip(z, -clip, clip) for z in logits)
    return Tile(logits, tile.wires), state


def update(eta: jax.Array | float, tile: Tile, s: Arrays, clip: float | None = None) -> Tile:
    """One step of the smallest rule: z ← z − η · s, held within ±clip when given."""
    return apply("plain", (), tile, s, eta, clip)[0]
