"""Signals: what a local update may read, as functions from (tile, cases) to per-logit arrays.

Every signal is an instance of the adjoint method: the gradient at a parameter is the adjoint
variable at the place the parameter acts, times that place's local partial with respect to the
parameter. Here the place is a gate, the adjoint variable λ is the error at its output, and the
local partial is the address distribution (entry ← output) times the sigmoid's slope (logit ←
entry). A signal is therefore three coordinates, not a name: on which pass the circuit is
linearised (``on``), whose transposed Jacobian carries λ back from the outputs (``via``), and to
which parameter the local partial is taken (``to``). ``docs/signals.md`` holds the maths in order;
this file holds it in the same order.

The layered adjoint is re-derived 2026-09 from blastema/signals/relays.py (input_jac, _basis, the
backward sweeps), written here as two local quantities of a gate's read and one recursion.
"""

from __future__ import annotations

from itertools import product
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from loom.tile import Read, Tile, activations, forward, read, run, tables

Pass = Literal["soft", "hard"]
Via = Literal["autodiff", "relay", "uniform", "direct", "reachable", "flip"]
To = Literal["logit", "entry"]


class Signal(NamedTuple):
    on: Pass = "soft"  # the pass the circuit is linearised on: the soft forward, or the bits
    via: Via = "autodiff"  # whose transposed Jacobian carries the adjoint variable back
    to: To = "logit"  # the parameter the local partial is taken to: the logit, or the stored entry

    @property
    def label(self) -> str:
        """The coordinates joined, e.g. ``soft.relay.entry``: a display name and a test id, never
        something code branches on (a gate asserts it)."""
        return ".".join(self)


REFERENCE = Signal()  # the true gradient on the soft pass: what every signal is scored against

CELLS = tuple(
    Signal(on, via, to)
    for on, via, to in product(
        ("soft", "hard"),
        ("autodiff", "relay", "uniform", "direct", "reachable", "flip"),
        ("logit", "entry"),
    )
    if (via != "autodiff" or to == "logit")  # autodiff cannot take the partial to the entry
    and (via != "flip" or on == "hard")  # the cost of a flip is a bits quantity
)  # every combination the code supports: what the combinatorial test visits


# The loss, and its derivative at the outputs: the seed of every adjoint.


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


def seed(acts: list[jax.Array], y: jax.Array) -> jax.Array:
    """∂L/∂r at the outputs, [B, n_out]: the residual, scaled as the mean loss is."""
    return (acts[-1] - y) / y.size


# What a gate can compute about its own read, locally: the local partials of the adjoint method.


def address(inputs: jax.Array) -> jax.Array:
    """P(a | u), how much each entry is selected by the inputs: [B, gates, 2**arity].

    Built one input at a time, input i being address bit i: soft inputs spread it over the entries
    (the product distribution), bits make it one-hot. It is ∂r/∂T[a], the partial of the read to the
    entry, and read(T, u) = Σ_a P(a | u) T[a].
    """
    p = jnp.ones((inputs.shape[0], inputs.shape[2], 1))
    for i in range(inputs.shape[1]):
        x = inputs[:, i, :, None]
        p = jnp.concatenate([(1.0 - x) * p, x * p], axis=-1)
    return p


def slope(logits: jax.Array) -> jax.Array:
    """σ'(z) = T (1 − T), the partial of the entry to its logit; the factor a chip storing entries
    rather than logits does not have (``to="entry"`` leaves it out)."""
    t = jax.nn.sigmoid(logits)
    return t * (1.0 - t)


def sensitivity(luts: jax.Array, inputs: jax.Array) -> jax.Array:
    """∂r/∂u_j, how much each gate's read moves with each of its inputs: [B, arity, gates].

    The read is multilinear, so it is the read at u_j = 1 minus the read at u_j = 0, the other
    inputs as they are: on bits, whether flipping that input flips the output, in {-1, 0, 1}. This
    is the gate's row of the Jacobian, what the relay carries back.
    """

    def at(j, v):
        return read(luts, inputs.at[:, j, :].set(v))

    return jnp.stack([at(j, 1.0) - at(j, 0.0) for j in range(inputs.shape[1])], axis=1)


def ones(luts: jax.Array, inputs: jax.Array) -> jax.Array:
    """The Jacobian of a gate that merely adds its inputs: the carry of the uniform split."""
    return jnp.ones_like(inputs)


# The adjoints: how λ, the error at every gate's output, is carried back from the seed.


def layered(tile: Tile, acts: list[jax.Array], on: Pass, e: jax.Array, carry) -> list[jax.Array]:
    """The adjoint variable λ at every gate, layer by layer from the outputs: [B, gates] per layer.

    The transposed Jacobian of the forward pass, applied to the seed one layer at a time: each gate
    passes its λ to input j times ``carry(luts, inputs)[j]``, and a line's λ is the sum over the
    gates it feeds (the forward gathers along the wiring, ``u[:, w]``; the adjoint scatters and adds
    along the same wiring, its transpose). With ``sensitivity`` as the carry this is the chain rule;
    with ``ones`` it is the adjoint of the same wiring with every gate a sum.
    """
    lams = [e]  # the output layer's λ is the seed itself
    for lg, w, u in reversed(list(zip(tile.logits[1:], tile.wires[1:], acts[1:-1], strict=True))):
        message = lams[0][:, None, :] * carry(tables(lg, on), u[:, w])  # [B, arity, gates]
        lams.insert(0, jnp.zeros_like(u).at[:, w].add(message))  # fan-out sums
    return lams


def feedback(tile: Tile, n_out: int, key: int = 0) -> list[jax.Array]:
    """A fixed random ±1 matrix per hidden layer, [gates, n_out], and the identity at the output
    layer, whose gates read their own residual: the feedback of direct feedback alignment. Drawn
    once from the tile's shape and ``key``, never trained."""
    keys = jax.random.split(jax.random.key(key), len(tile.logits) - 1)
    hidden = [
        jnp.sign(jax.random.normal(k, (lg.shape[0], n_out)))
        for k, lg in zip(keys, tile.logits[:-1], strict=True)
    ]
    return [*hidden, jnp.eye(n_out)]


def reachability(tile: Tile, acts: list[jax.Array], n_out: int) -> list[jax.Array]:
    """The number of wiring paths from every gate to every output, [gates, n_out] per layer: the
    layered adjoint of the all-sums network seeded with one output at a time (the carry ignores
    the values, so any ``n_out`` cases serve). Zero where a gate cannot reach an output."""
    per_output = [a[:n_out] for a in acts]
    return [c.T for c in layered(tile, per_output, "soft", jnp.eye(n_out), ones)]


def direct(tile: Tile, e: jax.Array, matrices: list[jax.Array]) -> list[jax.Array]:
    """The adjoint variable at every gate through a fixed matrix straight from the outputs, no
    layers: λ[b, g] = Σ_o B[g, o] e[b, o]. What a bus and a coefficient per gate would compute."""
    return [e @ b.T for b in matrices]


def readout(lams: list[jax.Array], tile: Tile, acts: list[jax.Array], on: Pass, to: To):
    """The per-logit signal from the adjoint variables: λ times the local partials.

    For every entry a of every gate, Σ_b λ[b, g] P_b(a | u_b), the adjoint variable against the
    address distribution summed over cases (the partial to the entry), times σ'(z[a]) when the
    partial is taken to the logit.
    """
    out = []
    for lam, lg, w, u in zip(lams, tile.logits, tile.wires, acts[:-1], strict=True):
        per_entry = jnp.sum(lam[:, :, None] * address(u[:, w]), axis=0)  # [gates, 2**arity]
        out.append(per_entry * slope(lg) if to == "logit" else per_entry)
    return tuple(out)


# The transports: one function per ``via``, each a choice of adjoint, then the readout.


def straight_through(logits: jax.Array) -> jax.Array:
    """Tables that are exactly the bits in value and differentiate like the probabilities: how
    autodiff is made to run on the deployed pass. Only the rounding's derivative is replaced;
    everything downstream is evaluated on the bits that flowed."""
    soft = tables(logits, "soft")
    return jnp.round(soft) + (soft - jax.lax.stop_gradient(soft))


def autodiff(tile: Tile, x: jax.Array, y: jax.Array, on: Pass, to: To):
    """The reference implementation of the true adjoint: the gradient of the loss on the pass by
    autodiff through everything. On the hard pass the straight-through tables put the bits in the
    forward and σ' in place of the rounding's (zero) derivative. It cannot take the partial to the
    entry (dividing by σ' would be 0/0 at saturated logits); the relay computes that signal."""
    if to != "logit":
        raise NotImplementedError("autodiff takes the partial to the logit only; use the relay")
    make = (lambda lg: tables(lg, "soft")) if on == "soft" else straight_through

    def on_pass(logits):
        return squared_error(run(tuple(make(lg) for lg in logits), tile.wires, x)[-1], y)

    return jax.grad(on_pass)(tile.logits)


def relay(tile: Tile, x: jax.Array, y: jax.Array, on: Pass, to: To):
    """The true adjoint computed locally: each gate carries its own sensitivity. Equal to autodiff
    where both exist (a test), and what a chip with a reverse channel per wire would run."""
    acts = activations(tile, x, on)
    return readout(layered(tile, acts, on, seed(acts, y), sensitivity), tile, acts, on, to)


def uniform(tile: Tile, x: jax.Array, y: jax.Array, on: Pass, to: To):
    """Feedback alignment on the wiring: the adjoint of the same wiring with every gate a sum.
    Value-blind; the fixed feedback is the number of wiring paths from each gate to each output."""
    acts = activations(tile, x, on)
    return readout(layered(tile, acts, on, seed(acts, y), ones), tile, acts, on, to)


def direct_feedback(tile: Tile, x: jax.Array, y: jax.Array, on: Pass, to: To):
    """Direct feedback alignment: the residual on a bus, through a fixed random ±1 coefficient per
    gate and output, no reverse wiring at all, and no regard for which outputs a gate can reach."""
    acts = activations(tile, x, on)
    e = seed(acts, y)
    return readout(direct(tile, e, feedback(tile, e.shape[1])), tile, acts, on, to)


def reachable_feedback(tile: Tile, x: jax.Array, y: jax.Array, on: Pass, to: To):
    """Direct feedback restricted to the outputs a gate can reach: the same random ±1 bus, masked
    by the wiring's reachability, which the audit of 2026-09-07 found to be the whole difference
    between the bus and the wiring-shaped split. A gate needs one bit per output: reachable?"""
    acts = activations(tile, x, on)
    e = seed(acts, y)
    masked = [
        b * (r > 0)
        for b, r in zip(
            feedback(tile, e.shape[1]), reachability(tile, acts, e.shape[1]), strict=True
        )
    ]
    return readout(direct(tile, e, masked), tile, acts, on, to)


def flip_credit(tile: Tile, x: jax.Array, y: jax.Array, on: Pass, to: To):
    """Outside the adjoint frame: the exact first-order credit of flipping one entry on the bits,
    with the cost of the outputs it would break. Two adjoints: the relay's λ, and the *reach*, the
    same recursion with |sensitivity| seeded by ones: the number of live paths from the gate to the
    outputs, which is the number of outputs a flip changes wherever paths do not reconverge (exact
    in the upper layers, an over-count at the input layer, audit of 2026-09-07). A flip costs half a
    unit per right output it reaches, so the signal is the relay's plus half the reach in the
    direction of the flip, (1 − 2H[a]). Defined on the bits; a cell of the hard pass only."""
    acts = activations(tile, x, on)
    e = seed(acts, y)
    lams = layered(tile, acts, on, e, sensitivity)
    reach = layered(
        tile, acts, on, jnp.ones_like(e) / y.size, lambda t, u: jnp.abs(sensitivity(t, u))
    )
    signal = readout(lams, tile, acts, on, to)
    cost = readout(reach, tile, acts, on, to)
    return tuple(
        s + 0.5 * c * (1.0 - 2.0 * tables(lg, "hard"))
        for s, c, lg in zip(signal, cost, tile.logits, strict=True)
    )


TRANSPORTS = {
    "autodiff": autodiff,
    "relay": relay,
    "uniform": uniform,
    "direct": direct_feedback,
    "reachable": reachable_feedback,
    "flip": flip_credit,
}


def compute(signal: Signal, tile: Tile, x: jax.Array, y: jax.Array) -> tuple[jax.Array, ...]:
    """The signal's per-logit arrays: its transport, on its pass, with the partial taken to its
    parameter."""
    return TRANSPORTS[signal.via](tile, x, y, signal.on, signal.to)


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
