"""Probe: why does the blind split train on the bits where the exact relay does not, when at
initialisation neither predicts a single flip's effect (probes/2026-09-07-flip-credit.py)? Both
hard signals under Adam, plain SGD (rate normalised so the mean nonzero step is 0.05 at
initialisation, and a fifth and five times that) and sign-SGD; hard accuracy, the fraction of
logits with a nonzero signal and the number of table bits flipping per step, along training."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax
import jax.numpy as jnp
import optax

from loom import signals, tasks, tile
from loom.signals import Signal

WIDTHS, ARITY, STEPS, SEEDS = (6, 32, 32, 16, 4), 3, 2000, 2
x, y = tasks.add(6)


def run(sig, opt, t):
    state = opt.init(t.logits)

    @jax.jit
    def step(logits, state):
        s = signals.compute(sig, t._replace(logits=logits), x, y)
        updates, state = opt.update(s, state, logits)
        new = optax.apply_updates(logits, updates)
        flat = jnp.concatenate([a.ravel() for a in s])
        flips = sum(jnp.sum(jnp.sign(a) != jnp.sign(b)) for a, b in zip(logits, new, strict=True))
        return new, state, jnp.mean(flat != 0), flips

    logits, dens, flips = t.logits, [], []
    for _ in range(STEPS):
        logits, state, d, f = step(logits, state)
        dens.append(d), flips.append(f)
    acc = tile.accuracy(t._replace(logits=logits), x, y, "hard")
    return float(acc), float(jnp.mean(jnp.array(dens))), float(jnp.mean(jnp.array(flips)))


print(f"shape {WIDTHS} arity {ARITY}, addition, {STEPS} steps")
print("per seed: hard accuracy / logits with nonzero signal / table bits flipping per step")
for sig in (Signal("hard", "relay", "entry"), Signal("hard", "uniform", "entry")):
    t0 = tile.init(jax.random.key(0), WIDTHS, ARITY)
    s0 = jnp.concatenate([a.ravel() for a in signals.compute(sig, t0, x, y)])
    unit = 0.05 / float(jnp.mean(jnp.abs(s0[s0 != 0])))  # SGD rate for a mean nonzero step of 0.05
    opts = {
        "adam 0.02": optax.adam(0.02),
        "sign-sgd 0.02": optax.chain(optax.scale_by_sign(), optax.sgd(0.02)),
    }
    opts.update({f"sgd ×{k}": optax.sgd(unit * k) for k in (0.2, 1.0, 5.0)})
    for name, opt in opts.items():
        cells = [
            run(sig, opt, tile.init(jax.random.key(seed), WIDTHS, ARITY)) for seed in range(SEEDS)
        ]
        print(
            f"{sig.label:16s} {name:14s}"
            + "".join(f"   {a:.3f} / {d:.2f} / {f:5.1f}" for a, d, f in cells),
            flush=True,
        )
