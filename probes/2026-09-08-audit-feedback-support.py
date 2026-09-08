"""Audit probe (adversarial audit of 2026-09-07, lifted 2026-09-08): what makes a fixed feedback
train the bits? Random direct feedback against the same bus masked to the outputs each gate can
reach, the path counts with a random sign per gate and output, the layered adjoint with a carry of
−1 on every edge, and with a random fixed sign per edge; the uniform split as the reference. If
the mask alone closes the gap, reachability is the whole story and the signs can be anything."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax
import jax.numpy as jnp
import optax

from loom import signals, tasks, tile
from loom.tile import activations

WIDTHS, ARITY, STEPS, SEEDS = (6, 32, 32, 16, 4), 3, 2000, 3
x, y = tasks.add(6)
n_out = y.shape[1]


def train(signal_fn, t):
    opt = optax.adam(0.02)
    state = opt.init(t.logits)

    @jax.jit
    def step(logits, state):
        updates, state = opt.update(signal_fn(t._replace(logits=logits)), state, logits)
        return optax.apply_updates(logits, updates), state

    logits = t.logits
    for _ in range(STEPS):
        logits, state = step(logits, state)
    return float(tile.accuracy(t._replace(logits=logits), x, y, "hard"))


def through_bus(matrices_of):
    def f(t):
        acts = activations(t, x, "hard")
        e = signals.seed(acts, y)
        return signals.readout(signals.direct(e, matrices_of(t, acts)), t, acts, "hard", "entry")

    return f


def through_layers(carry_of):
    def f(t):
        acts = activations(t, x, "hard")
        lams = signals.layered(t, acts, "hard", signals.seed(acts, y), carry_of(t))
        return signals.readout(lams, t, acts, "hard", "entry")

    return f


def edge_signs(t, key):
    return [
        jnp.sign(jax.random.normal(jax.random.fold_in(jax.random.key(key), i), w.shape))
        for i, w in enumerate(t.wires[1:])
    ]


def per_edge(signs):
    """A carry that returns the layer's fixed sign per edge, matched by the wiring's shape."""

    def carry(luts, inputs):
        w = next(s for s in signs if s.shape == inputs.shape[1:])
        return jnp.broadcast_to(w, inputs.shape)

    return carry


def counts(t, acts):
    return signals.reachability(t, acts, n_out)


rows = {
    "uniform (path counts)": through_layers(lambda t: signals.ones),
    "direct, key 0": through_bus(lambda t, a: signals.feedback(t, n_out, 0)),
    "direct, key 1": through_bus(lambda t, a: signals.feedback(t, n_out, 1)),
    "direct, key 2": through_bus(lambda t, a: signals.feedback(t, n_out, 2)),
    "direct key 0, masked to reachable": through_bus(
        lambda t, a: [
            b * (c > 0) for b, c in zip(signals.feedback(t, n_out, 0), counts(t, a), strict=True)
        ]
    ),
    "direct key 1, masked to reachable": through_bus(
        lambda t, a: [
            b * (c > 0) for b, c in zip(signals.feedback(t, n_out, 1), counts(t, a), strict=True)
        ]
    ),
    "reachability, all +1": through_bus(lambda t, a: [(c > 0) * 1.0 for c in counts(t, a)]),
    "path counts x random sign, key 0": through_bus(
        lambda t, a: [
            c * b for b, c in zip(signals.feedback(t, n_out, 0), counts(t, a), strict=True)
        ]
    ),
    "carry -1 on every edge": through_layers(lambda t: lambda luts, u: -jnp.ones_like(u)),
    "random sign per edge, key 0": through_layers(lambda t: per_edge(edge_signs(t, 0))),
    "random sign per edge, key 1": through_layers(lambda t: per_edge(edge_signs(t, 1))),
}
print(
    f"shape {WIDTHS} arity {ARITY}, addition, bits, Adam 0.02, {STEPS} steps: hard acc, seeds 0-2"
)
for name, f in rows.items():
    accs = [train(f, tile.init(jax.random.key(s), WIDTHS, ARITY)) for s in range(SEEDS)]
    print(f"{name:36s}" + "  ".join(f"{a:.3f}" for a in accs), flush=True)
