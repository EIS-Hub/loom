"""Probe: every supported signal cell scored against the reference, layer by layer, at
initialisation on a shape where depth is forced (6-input addition, arity 3). Does the uniform split
lose the reference's sign as the error travels deeper, where the relay keeps it?"""

import jax
import jax.numpy as jnp

from loom import signals, tasks, tile
from loom.signals import CELLS, REFERENCE

WIDTHS, ARITY, SEEDS = (6, 32, 32, 16, 4), 3, 3
x, y = tasks.add(6)
layers = range(len(WIDTHS) - 1)
print(f"shape {WIDTHS} arity {ARITY}, addition; mean over {SEEDS} seeds; layers input → output")
print(f"{'signal':20s}" + "".join(f"{'nonzero':>9s}{'cosine':>8s}{'sign':>6s}  " for _ in layers))
for sig in CELLS:
    rows = []
    for seed in range(SEEDS):
        t = tile.init(jax.random.key(seed), WIDTHS, ARITY)
        g, r = signals.compute(sig, t, x, y), signals.compute(REFERENCE, t, x, y)
        rows.append([signals.score((g[i],), (r[i],)) for i in layers])
    cells = [
        {k: float(jnp.mean(jnp.stack([rows[s][i][k] for s in range(SEEDS)]))) for k in rows[0][i]}
        for i in layers
    ]
    print(
        f"{sig.label:20s}"
        + "".join(f"{c['nonzero']:9.2f}{c['cosine']:+8.2f}{c['sign']:6.2f}  " for c in cells)
    )

print("\nfraction of back-edges carrying exactly zero (gate insensitive to that input), per layer")
for on in ("soft", "hard"):
    dead = []
    for seed in range(SEEDS):
        t = tile.init(jax.random.key(seed), WIDTHS, ARITY)
        acts = tile.activations(t, x, on)
        dead.append(
            [
                float(jnp.mean(signals.sensitivity(tile.tables(lg, on), u[:, w]) == 0))
                for lg, w, u in zip(t.logits, t.wires, acts[:-1], strict=True)
            ]
        )
    means = [float(jnp.mean(jnp.array([d[i] for d in dead]))) for i in layers]
    print(f"{on:6s}" + "".join(f"{m:8.2f}" for m in means))
