"""Probe: is the exact relay's failure on the bits a matter of stationarity? Its signal at a gate
depends on the current bits of every gate on the path to the outputs, so every flip changes the
signal of many others; the blind split's depends on the output residuals and the wiring alone.
Two tests along training under Adam: the step-to-step sign consistency of each signal on its own
trajectory, and throttling (each step updates a random fraction p of the logits, the rest frozen).
If simultaneous flips are what thrashes the circuit, throttling should rescue the relay."""

import jax
import jax.numpy as jnp
import optax

from loom import signals, tasks, tile
from loom.signals import Signal

WIDTHS, ARITY, STEPS, SEEDS = (6, 32, 32, 16, 4), 3, 2000, 2
x, y = tasks.add(6)


def run(sig, p, t, key):
    opt = optax.adam(0.02)
    state = opt.init(t.logits)

    @jax.jit
    def step(logits, state, prev, key):
        s = signals.compute(sig, t._replace(logits=logits), x, y)
        keys = jax.random.split(key, len(logits))
        masked = tuple(
            a * jax.random.bernoulli(k, p, a.shape) for a, k in zip(s, keys, strict=True)
        )
        updates, state = opt.update(masked, state, logits)
        new = optax.apply_updates(logits, updates)
        flat, was = (jnp.concatenate([a.ravel() for a in v]) for v in (s, prev))
        both = (flat != 0) & (was != 0)
        same = jnp.sum((jnp.sign(flat) == jnp.sign(was)) & both) / jnp.maximum(jnp.sum(both), 1)
        flips = sum(jnp.sum(jnp.sign(a) != jnp.sign(b)) for a, b in zip(logits, new, strict=True))
        return new, state, s, same, flips

    logits, prev, same, flips = t.logits, signals.compute(sig, t, x, y), [], []
    for _ in range(STEPS):
        key, k = jax.random.split(key)
        logits, state, prev, c, f = step(logits, state, prev, k)
        same.append(c), flips.append(f)
    acc = tile.accuracy(t._replace(logits=logits), x, y, "hard")
    return float(acc), float(jnp.mean(jnp.array(same))), float(jnp.mean(jnp.array(flips)))


print(f"shape {WIDTHS} arity {ARITY}, addition, Adam 0.02, {STEPS} steps")
print(
    "per seed: hard accuracy / sign consistency of the signal step to step / bits flipping per step"
)
for sig in (Signal("hard", "relay", False), Signal("hard", "uniform", False)):
    for p in (1.0, 0.1, 0.02):
        cells = [
            run(sig, p, tile.init(jax.random.key(s), WIDTHS, ARITY), jax.random.key(s))
            for s in range(SEEDS)
        ]
        print(
            f"{sig.label:16s} p={p:<5}"
            + "".join(f"   {a:.3f} / {c:.2f} / {f:5.1f}" for a, c, f in cells),
            flush=True,
        )
