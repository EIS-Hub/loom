"""Audit probe (adversarial audit of 2026-09-07, lifted 2026-09-08): is the flip credit's success
the exact cost term, or would any "stay" bias do? The relay plus a constant stay bias of several
sizes, plus the blind reach (path counts, no sensitivity), plus the local case count (the output
layer's exact cost applied everywhere without transport), and the uniform split plus the reach;
against the flip credit as implemented and the relay alone. Recipe seeds, Adam 0.02, 2000 steps."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax
import jax.numpy as jnp
import optax

from loom import recipes, signals, tasks, tile
from loom.recipes import DEEP_HARD_FLOOR
from loom.signals import Signal
from loom.tile import activations, tables

ADD, STEPS, SEEDS = tasks.addition(6), 2000, 3


def train(signal_fn, t, x, y):
    opt = optax.adam(0.02)
    state = opt.init(t.logits)

    @jax.jit
    def step(logits, state):
        updates, state = opt.update(signal_fn(t._replace(logits=logits), x, y), state, logits)
        return optax.apply_updates(logits, updates), state

    logits = t.logits
    for _ in range(STEPS):
        logits, state = step(logits, state)
    return float(tile.accuracy(t._replace(logits=logits), x, y, "hard"))


def parts(t, x, y):
    acts = activations(t, x, "hard")
    e = signals.seed(acts, y)
    relay = signals.last_hop(
        signals.backward(t, acts, "hard", e, signals.sensitivity), t, acts, "hard", "entry"
    )
    return acts, e, relay


def direction(t):
    return [1 - 2 * tables(lg, "hard") for lg in t.logits]


def relay_plus_constant(kappa):
    def f(t, x, y):
        _, _, relay = parts(t, x, y)
        return tuple(a + kappa / y.size * d for a, d in zip(relay, direction(t), strict=True))

    return f


def relay_plus_blind_reach(t, x, y):
    acts, e, relay = parts(t, x, y)
    reach = signals.backward(t, acts, "hard", jnp.ones_like(e) / y.size, signals.ones)
    cost = signals.last_hop(reach, t, acts, "hard", "entry")
    return tuple(a + 0.5 * c * d for a, c, d in zip(relay, cost, direction(t), strict=True))


def relay_plus_local_count(t, x, y):
    acts, _, relay = parts(t, x, y)
    counts = [
        jnp.sum(signals.address(u[:, w]), 0) / y.size
        for w, u in zip(t.wires, acts[:-1], strict=True)
    ]
    return tuple(a + 0.5 * c * d for a, c, d in zip(relay, counts, direction(t), strict=True))


def uniform_plus_reach(t, x, y):
    acts, e, _ = parts(t, x, y)
    blind = signals.last_hop(
        signals.backward(t, acts, "hard", e, signals.ones), t, acts, "hard", "entry"
    )
    reach = signals.backward(
        t,
        acts,
        "hard",
        jnp.ones_like(e) / y.size,
        lambda tb, u: jnp.abs(signals.sensitivity(tb, u)),
    )
    cost = signals.last_hop(reach, t, acts, "hard", "entry")
    return tuple(a + 0.5 * c * d for a, c, d in zip(blind, cost, direction(t), strict=True))


def as_is(signal):
    return lambda t, x, y: signals.compute(signal, t, x, y)


rows = {
    "flip credit, as implemented": as_is(Signal("hard", "flip", "entry")),
    "relay alone": as_is(Signal("hard", "relay", "entry")),
    "relay + 0.25/N stay": relay_plus_constant(0.25),
    "relay + 0.5/N stay": relay_plus_constant(0.5),
    "relay + 1/N stay": relay_plus_constant(1.0),
    "relay + 2/N stay": relay_plus_constant(2.0),
    "relay + 0.5 blind reach": relay_plus_blind_reach,
    "relay + 0.5 local count": relay_plus_local_count,
    "uniform + 0.5 reach": uniform_plus_reach,
}
print("bits, recipe seeds 0-2 of DEEP_HARD_FLOOR, Adam 0.02, 2000 steps: hard accuracy")
for name, f in rows.items():
    accs = []
    for seed in range(SEEDS):
        t, x, y, _ = recipes.setup(DEEP_HARD_FLOOR, ADD, seed)
        accs.append(train(f, t, x, y))
    print(f"{name:30s}" + "  ".join(f"{a:.3f}" for a in accs), flush=True)
