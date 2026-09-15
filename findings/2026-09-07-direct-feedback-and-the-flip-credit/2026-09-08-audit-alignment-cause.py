"""Audit probe (adversarial audit of 2026-09-07, lifted 2026-09-08): is the hidden drift under the
uniform split a cause or a side effect, and is "alignment" the right word? With the output layer
always exact, the hidden layers frozen, driven by noise, by noise on the signal's support, by the
signal's magnitudes with shuffled signs, or by the signal; a monotone-initialised hidden stack with
output-only learning; and alignment as feedback alignment defines it, the delivered hidden signal
against the true adjoint, along training."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax
import jax.numpy as jnp
import optax

from loom import signals, tasks, tile
from loom.signals import Signal
from loom.tile import activations, tables

WIDTHS, ARITY, STEPS, SEEDS = (6, 32, 32, 16, 4), 3, 2000, 3
x, y = tasks.add(6)
UNIFORM, RELAY = Signal("hard", "uniform", "entry"), Signal("hard", "relay", "entry")


def hidden_as(kind, key):
    def f(t):
        s = list(signals.compute(UNIFORM, t, x, y))
        state_key = jax.random.fold_in(
            jax.random.key(key), (jnp.sum(jnp.abs(t.logits[0])) * 1e3).astype(jnp.int32) % 100000
        )
        for i in range(len(s) - 1):
            k = jax.random.fold_in(state_key, i)
            if kind == "frozen":
                s[i] = jnp.zeros_like(s[i])
            elif kind == "noise":
                s[i] = jax.random.normal(k, s[i].shape) * jnp.mean(jnp.abs(s[i]) + 1e-12)
            elif kind == "noise on support":
                s[i] = jax.random.normal(k, s[i].shape) * jnp.abs(s[i])
            elif kind == "shuffled signs":
                s[i] = jnp.abs(s[i]) * jnp.sign(jax.random.normal(k, s[i].shape))
        return tuple(s)

    return f


def train(signal_fn, t, steps=STEPS):
    opt = optax.adam(0.02)
    state = opt.init(t.logits)

    @jax.jit
    def step(logits, state):
        updates, state = opt.update(signal_fn(t._replace(logits=logits)), state, logits)
        return optax.apply_updates(logits, updates), state

    logits = t.logits
    for _ in range(steps):
        logits, state = step(logits, state)
    return t._replace(logits=logits)


def positive_fraction(t):
    acts = activations(t, x, "hard")
    out = []
    for lg, w, u in zip(t.logits, t.wires, acts[:-1], strict=True):
        s = signals.sensitivity(tables(lg, "hard"), u[:, w])
        out.append(float(jnp.sum(s > 0) / jnp.maximum(jnp.sum(s != 0), 1)))
    return out


def monotone(t, seed):
    """Hidden tables as threshold gates, increasing in every input."""
    new = []
    for i, lg in enumerate(t.logits[:-1]):
        a = jnp.arange(lg.shape[1])
        popcount = sum((a >> j) & 1 for j in range(ARITY))
        thr = jax.random.randint(
            jax.random.fold_in(jax.random.key(seed), i), (lg.shape[0], 1), 1, ARITY + 1
        )
        new.append(3.0 * (popcount[None, :] - thr + 0.5))
    return t._replace(logits=(*new, t.logits[-1]))


print(f"shape {WIDTHS} arity {ARITY}, addition, bits, Adam 0.02, {STEPS} steps")
print("the output layer is always given the exact signal")
print("hidden layers driven by: hard accuracy per seed, and the positive fraction of sensitivities")
for kind in ("frozen", "noise", "noise on support", "shuffled signs", "the uniform signal"):
    cells = []
    for seed in range(SEEDS):
        f = (
            (lambda t: signals.compute(UNIFORM, t, x, y))
            if kind == "the uniform signal"
            else hidden_as(kind, seed)
        )
        done = train(f, tile.init(jax.random.key(seed), WIDTHS, ARITY))
        pos = "/".join(f"{p:.2f}" for p in positive_fraction(done))
        cells.append(f"{float(tile.accuracy(done, x, y, 'hard')):.3f} ({pos})")
    print(f"{kind:20s}" + "  ".join(cells), flush=True)
print("\nmonotone hidden stack: output-only learning, and the uniform signal from the same start")
for seed in range(SEEDS):
    t = monotone(tile.init(jax.random.key(seed), WIDTHS, ARITY), seed)
    only = train(hidden_as("frozen", seed), t)
    full = train(lambda t: signals.compute(UNIFORM, t, x, y), t)
    a, b = float(tile.accuracy(only, x, y, "hard")), float(tile.accuracy(full, x, y, "hard"))
    print(f"seed {seed}: output-only {a:.3f}   uniform {b:.3f}")
print("\nalignment as feedback alignment defines it:")
print("sign agreement of the delivered hidden signal with the true adjoint, per hidden layer")
for seed in range(2):
    line = []
    for steps in (0, 200, 1000, 2000):
        done = train(
            lambda t: signals.compute(UNIFORM, t, x, y),
            tile.init(jax.random.key(seed), WIDTHS, ARITY),
            steps,
        )
        s, r = signals.compute(UNIFORM, done, x, y), signals.compute(RELAY, done, x, y)
        agree = "/".join(
            f"{float(signals.score((s[i],), (r[i],))['sign']):.2f}" for i in range(len(s) - 1)
        )
        line.append(f"step {steps}: {agree} acc {float(tile.accuracy(done, x, y, 'hard')):.3f}")
    print(f"seed {seed}: " + "   ".join(line), flush=True)
