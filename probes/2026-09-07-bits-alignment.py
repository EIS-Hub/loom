"""Probe: does the circuit make the blind feedback right? The uniform split carries every error
back unchanged, as if each gate's sensitivity to each input were +1. If the bits train the way
fixed random feedback trains a network (feedback alignment), the gates should drift toward that
assumption: the fraction of nonzero hard sensitivities that are positive should rise from one half.
Measured per layer along training under each hard signal (Adam 0.02)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax
import jax.numpy as jnp
import optax

from loom import signals, tasks, tile
from loom.signals import Signal

WIDTHS, ARITY, SEEDS = (6, 32, 32, 16, 4), 3, 3
x, y = tasks.add(6)
layers = range(len(WIDTHS) - 1)


def positive_fraction(t):
    acts = tile.activations(t, x, "hard")
    out = []
    for lg, w, u in zip(t.logits, t.wires, acts[:-1], strict=True):
        s = signals.sensitivity(tile.tables(lg, "hard"), u[:, w])
        out.append(float(jnp.sum(s > 0) / jnp.maximum(jnp.sum(s != 0), 1)))
    return out


def train(sig, t, steps):
    opt = optax.adam(0.02)
    state = opt.init(t.logits)

    @jax.jit
    def step(logits, state):
        s = signals.compute(sig, t._replace(logits=logits), x, y)
        updates, state = opt.update(s, state, logits)
        return optax.apply_updates(logits, updates), state

    logits = t.logits
    for _ in range(steps):
        logits, state = step(logits, state)
    return t._replace(logits=logits)


print(f"shape {WIDTHS} arity {ARITY}, addition: positive fraction of nonzero hard sensitivities")
print(f"{'signal':16s} {'step':>5s}   layers input → output       hard acc")
for sig in (Signal("hard", "uniform", "entry"), Signal("hard", "relay", "entry")):
    for steps in (0, 200, 1000, 2000):
        fr, acc = [], []
        for seed in range(SEEDS):
            t = train(sig, tile.init(jax.random.key(seed), WIDTHS, ARITY), steps)
            fr.append(positive_fraction(t)), acc.append(float(tile.accuracy(t, x, y, "hard")))
        means = [sum(f[i] for f in fr) / SEEDS for i in layers]
        print(
            f"{sig.label:16s} {steps:5d}   "
            + " ".join(f"{m:.2f}" for m in means)
            + f"        {sum(acc) / SEEDS:.3f}",
            flush=True,
        )
