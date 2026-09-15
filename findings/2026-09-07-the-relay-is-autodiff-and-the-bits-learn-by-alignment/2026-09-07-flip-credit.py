"""Probe: is a hard-pass signal right about the bits? For every addressed entry with a nonzero
signal, flip that table bit and measure the actual change of the hard loss; compare its sign with
the sign the signal predicts, s[a]·(1 - 2H[a]). The exact relay should be right wherever a single
path carries the credit; the blind split has no reason to be. Also: how many gates the relayed
error never reaches (dead gates), per layer, on each pass."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax
import jax.numpy as jnp

from loom import signals, tasks, tile
from loom.signals import Signal

WIDTHS, ARITY, SEEDS = (6, 32, 32, 16, 4), 3, 3
x, y = tasks.add(6)
layers = range(len(WIDTHS) - 1)


def flip_credit(sig, t):
    s = signals.compute(sig, t, x, y)
    base = signals.loss(t, x, y, "hard")
    right = []
    for i in layers:
        h = tile.tables(t.logits[i], "hard")
        gs, As = jnp.nonzero(s[i])
        predicted = jnp.sign(s[i][gs, As] * (1.0 - 2.0 * h[gs, As]))

        def flipped_loss(g, a, i=i):
            lg = t.logits[i].at[g, a].multiply(-1.0)
            return signals.loss(
                t._replace(logits=t.logits[:i] + (lg,) + t.logits[i + 1 :]), x, y, "hard"
            )

        actual = jnp.sign(jax.vmap(flipped_loss)(gs, As) - base)
        both = (actual != 0) & (predicted != 0)
        right.append((float(jnp.mean(actual[both] == predicted[both])), int(both.sum())))
    return right


print(f"shape {WIDTHS} arity {ARITY}, addition, hard pass, {SEEDS} seeds: layers input → output")
print("sign of the predicted loss change vs the actual one, over flips where both are nonzero")
for sig in (Signal("hard", "relay", "entry"), Signal("hard", "uniform", "entry")):
    rows = [
        flip_credit(sig, tile.init(jax.random.key(seed), WIDTHS, ARITY)) for seed in range(SEEDS)
    ]
    cells = [
        (sum(r[i][0] for r in rows) / SEEDS, sum(r[i][1] for r in rows) // SEEDS) for i in layers
    ]
    print(f"{sig.label:18s}" + "".join(f"  {c[0]:.2f} (n={c[1]:3d})" for c in cells))

print("\nfraction of gates the relayed error never reaches over the whole batch (dead gates)")
for on in ("soft", "hard"):
    rows = []
    for seed in range(SEEDS):
        t = tile.init(jax.random.key(seed), WIDTHS, ARITY)
        g = signals.compute(Signal(on, "relay", "entry"), t, x, y)
        rows.append([float(jnp.mean(jnp.all(gi == 0, axis=1))) for gi in g])
    print(f"{on:6s}" + "".join(f"{sum(r[i] for r in rows) / SEEDS:8.2f}" for i in layers))
