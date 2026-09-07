"""Signals: what a local update may read, as functions from (tile, cases) to per-logit arrays.

A signal is three coordinates, not a name: on which pass it is computed (the soft forward, or the
deployed bits), via which transport the residual reaches the logits (autodiff through everything,
the relay through each gate's own table, the blind uniform split), and whether the surrogate
factor σ'(logit) is kept. The reference gradient is (soft, autodiff, surrogate); what is often
called the straight-through gradient is (hard, autodiff, surrogate). Every signal has the shape of
``tile.logits``, so descent and the rule consume them alike.

The relay is re-derived 2026-09 from blastema/signals/relays.py (input_jac, _basis, the sweeps),
written here as two local quantities of a gate's read and one backward sweep.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from loom.tile import Read, Tile, activations, forward, read, run, tables

Pass = Literal["soft", "hard"]
Via = Literal["autodiff", "relay", "uniform"]


class Signal(NamedTuple):
    on: Pass = "soft"  # the pass the signal is computed on
    via: Via = "autodiff"  # the transport from the residual to the logits
    surrogate: bool = True  # keep σ'(logit), the factor a chip storing bits cannot see

    @property
    def label(self) -> str:
        """The coordinates as a name, e.g. ``soft·autodiff·σ'``; nothing is named by hand."""
        return f"{self.on}·{self.via}·{'σ′' if self.surrogate else '1'}"


REFERENCE = Signal()  # the true gradient on the soft pass: what every signal is scored against

CELLS = tuple(
    Signal(on, via, keep)
    for on in ("soft", "hard")
    for via in ("autodiff", "relay", "uniform")
    for keep in (True, False)
    if keep or via != "autodiff"
)  # every combination the code supports: what the combinatorial test visits


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
    (zero) derivative, which is what ``surrogate=True`` means. Dropping it is the relay's to compute
    (dividing this gradient by σ' would be 0/0 at saturated logits).
    """
    if not surrogate:
        raise NotImplementedError("autodiff cannot drop σ'; the relay computes that signal")
    make = (lambda lg: tables(lg, "soft")) if on == "soft" else straight_through

    def on_pass(logits):
        return squared_error(run(tuple(make(lg) for lg in logits), tile.wires, x)[-1], y)

    return jax.grad(on_pass)(tile.logits)


# What a gate can compute about its own read, locally: the two factors every transport is made of.


def address(inputs: jax.Array) -> jax.Array:
    """How much each table entry is selected by the inputs, P(a | u): [B, gates, 2**arity].

    Soft inputs spread it over the entries (the product distribution); bits make it one-hot. The
    read is this distribution against the table, read(T, u) = Σ_a P(a | u) T[a].
    """
    p = jnp.ones((inputs.shape[0], inputs.shape[2], 1))
    for i in range(inputs.shape[1]):
        x = inputs[:, i, :, None]
        p = jnp.concatenate([(1.0 - x) * p, x * p], axis=-1)  # input i is address bit i
    return p


def sensitivity(luts: jax.Array, inputs: jax.Array) -> jax.Array:
    """How much each gate's read moves with each of its inputs, ∂r/∂u_j: [B, arity, gates].

    The read is multilinear, so this is the read at u_j = 1 minus the read at u_j = 0, the other
    inputs as they are: on bits, whether flipping that input flips the output, in {-1, 0, 1}.
    """

    def at(j, v):
        return read(luts, inputs.at[:, j, :].set(v))

    return jnp.stack([at(j, 1.0) - at(j, 0.0) for j in range(inputs.shape[1])], axis=1)


def sweep(tile: Tile, x: jax.Array, y: jax.Array, on: Pass, surrogate: bool, carry):
    """The residual carried back gate by gate on the pass, as local messages.

    At each gate the error at its output meets the address distribution to become the per-entry
    signal (times σ'(logit) when the surrogate is kept), and is passed on to each input times
    ``carry(luts, inputs)``, summed over fan-out. With the gate's own sensitivity as the carry this
    is the chain rule; with ones it is blind to what the gate computes.
    """
    acts = activations(tile, x, on)
    e = (acts[-1] - y) / y.size  # ∂L/∂r at the outputs: the residual, scaled as the mean loss is
    grads = []
    for lg, w, u in reversed(list(zip(tile.logits, tile.wires, acts[:-1], strict=True))):
        luts, inputs = tables(lg, on), u[:, w]
        g = jnp.einsum("bg,bga->ga", e, address(inputs))
        soft = jax.nn.sigmoid(lg)
        grads.append(g * soft * (1.0 - soft) if surrogate else g)
        e = jnp.zeros_like(u).at[:, w].add(e[:, None, :] * carry(luts, inputs))
    return tuple(reversed(grads))


def relay(tile: Tile, x: jax.Array, y: jax.Array, on: Pass, surrogate: bool):
    """The transport a chip could run: the exact chain rule as local messages, equal to autodiff."""
    return sweep(tile, x, y, on, surrogate, sensitivity)


def uniform(tile: Tile, x: jax.Array, y: jax.Array, on: Pass, surrogate: bool):
    """The value-blind transport: every gate passes its error unchanged to every input."""
    return sweep(tile, x, y, on, surrogate, lambda luts, inputs: jnp.ones_like(inputs))


TRANSPORTS = {"autodiff": autodiff, "relay": relay, "uniform": uniform}


def compute(signal: Signal, tile: Tile, x: jax.Array, y: jax.Array) -> tuple[jax.Array, ...]:
    """The signal's per-logit arrays: its transport, run on its pass, with or without σ'."""
    return TRANSPORTS[signal.via](tile, x, y, signal.on, signal.surrogate)


def score(signal, reference) -> dict[str, jax.Array]:
    """A signal against the reference over the logits given: the fraction of nonzero entries, the
    cosine, and the sign agreement where both are nonzero (descent needs the sign, not the size)."""
    a = jnp.concatenate([g.ravel() for g in signal])
    b = jnp.concatenate([g.ravel() for g in reference])
    both = (a != 0) & (b != 0)
    return {
        "nonzero": jnp.mean(a != 0),
        "cosine": jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12),
        "sign": jnp.mean(jnp.sign(a[both]) == jnp.sign(b[both])),
    }
