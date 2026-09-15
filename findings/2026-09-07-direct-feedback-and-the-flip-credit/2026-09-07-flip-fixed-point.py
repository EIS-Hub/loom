"""Probe: the flip credit on the bits stops early (hard accuracy identical at 500 and 2000 steps).
Is that a fixed point, a state where no entry wants to flip? After training under it: the number
of entries whose credit still points toward a flip, the bits flipping per step at the end, and the
accuracy, against the same under the uniform split."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax
import jax.numpy as jnp
import optax

from loom import signals, tasks, tile
from loom.signals import Signal

WIDTHS, ARITY, STEPS, SEEDS = (6, 32, 32, 16, 4), 3, 2000, 3
x, y = tasks.add(6)


def run(sig, t):
    opt = optax.adam(0.02)
    state = opt.init(t.logits)

    @jax.jit
    def step(logits, state):
        s = signals.compute(sig, t._replace(logits=logits), x, y)
        updates, state = opt.update(s, state, logits)
        new = optax.apply_updates(logits, updates)
        flips = sum(jnp.sum(jnp.sign(a) != jnp.sign(b)) for a, b in zip(logits, new, strict=True))
        return new, state, flips

    logits, flips = t.logits, []
    for _ in range(STEPS):
        logits, state, f = step(logits, state)
        flips.append(f)
    done = t._replace(logits=logits)
    s = signals.compute(sig, done, x, y)
    wants = sum(
        int(jnp.sum(a * (1.0 - 2.0 * tile.tables(lg, "hard")) < 0))
        for a, lg in zip(s, done.logits, strict=True)
    )
    return float(tile.accuracy(done, x, y, "hard")), wants, float(jnp.mean(jnp.array(flips[-200:])))


print(f"shape {WIDTHS} arity {ARITY}, addition, Adam 0.02, {STEPS} steps")
print(
    "per seed: hard acc / entries still pointing toward a flip / flips per step over the last 200"
)
for sig in (Signal("hard", "flip", "entry"), Signal("hard", "uniform", "entry")):
    cells = [run(sig, tile.init(jax.random.key(s), WIDTHS, ARITY)) for s in range(SEEDS)]
    print(
        f"{sig.label:18s}" + "".join(f"   {a:.3f} / {w:3d} / {f:5.1f}" for a, w, f in cells),
        flush=True,
    )
