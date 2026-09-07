"""Signals: what a local update may read, as functions from (tile, cases) to per-logit arrays.

A signal is three coordinates, not a name: on which pass it is computed (the soft forward, or the
deployed bits), via which transport the residual reaches the logits (autodiff through everything,
the relay through each gate's own table, the blind uniform split), and whether the surrogate
factor σ'(logit) is kept. The reference gradient is (soft, autodiff, surrogate); what is often
called the straight-through gradient is (hard, autodiff, surrogate). Every signal has the shape of
``tile.logits``, so descent and the rule consume them alike. This chunk holds the autodiff
transport; the next adds the relay and the uniform split, and with them dropping the surrogate.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from loom.tile import Read, Tile, forward, run, tables

Pass = Literal["soft", "hard"]
Via = Literal["autodiff"]  # the next chunk adds "relay" and "uniform"


class Signal(NamedTuple):
    on: Pass = "soft"  # the pass the signal is computed on
    via: Via = "autodiff"  # the transport from the residual to the logits
    surrogate: bool = True  # keep σ'(logit), the factor a chip storing bits cannot see

    @property
    def label(self) -> str:
        """The coordinates as a name, e.g. ``soft·autodiff·σ'``; nothing is named by hand."""
        return f"{self.on}·{self.via}·{'σ′' if self.surrogate else '1'}"


REFERENCE = (
    Signal()
)  # the true gradient on the soft pass: what every other signal is scored against


def squared_error(out: jax.Array, y: jax.Array) -> jax.Array:
    """Half the squared error over cases and output bits. Its derivative at an output is the
    residual itself, the error a chip can see, in every read; a cross-entropy is infinite when a
    bit is wrong and, clipped, its derivative no longer says how wrong."""
    return 0.5 * jnp.mean((out - y) ** 2)


def loss(tile: Tile, x: jax.Array, y: jax.Array, mode: Read = "soft") -> jax.Array:
    """The squared error of the read against the demanded bits."""
    return squared_error(forward(tile, x, mode), y)


def residual(tile: Tile, x: jax.Array, y: jax.Array, mode: Read = "hard") -> jax.Array:
    """The error at the outputs, read minus demanded: [B, n_out]; bits in {-1, 0, 1} when hard."""
    return forward(tile, x, mode) - y


def straight_through(logits: jax.Array) -> jax.Array:
    """Tables that are exactly the bits in value and differentiate like the probabilities: how
    autodiff is made to run on the deployed pass. Only the rounding's derivative is replaced;
    everything downstream is evaluated on the bits that flowed."""
    soft = tables(logits, "soft")
    return jnp.round(soft) + (soft - jax.lax.stop_gradient(soft))


def autodiff(
    tile: Tile, x: jax.Array, y: jax.Array, on: Pass, surrogate: bool
) -> tuple[jax.Array, ...]:
    """The transport through everything, by autodiff: the gradient of the loss on the pass.

    On the soft pass every factor is evaluated at soft tables and inputs. On the hard pass the
    straight-through tables put the bits in the forward and σ'(z) in place of the rounding's
    (zero) derivative, which is what ``surrogate=True`` means; dropping it is computed directly by
    the relay, next chunk (dividing the gradient by σ' would be 0/0 at saturated logits).
    """
    if not surrogate:
        raise NotImplementedError("dropping σ' arrives with the relay transport")
    make = (lambda lg: tables(lg, "soft")) if on == "soft" else straight_through

    def on_pass(logits):
        return squared_error(run(tuple(make(lg) for lg in logits), tile.wires, x)[-1], y)

    return jax.grad(on_pass)(tile.logits)


TRANSPORTS = {"autodiff": autodiff}  # the next chunk adds "relay" and "uniform"


def compute(signal: Signal, tile: Tile, x: jax.Array, y: jax.Array) -> tuple[jax.Array, ...]:
    """The signal's per-logit arrays: its transport, run on its pass, with or without σ'."""
    return TRANSPORTS[signal.via](tile, x, y, signal.on, signal.surrogate)
