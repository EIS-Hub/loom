"""The meta-objective's landscape in eta, per window, on the flat tile: J(eta) = the soft loss of
the tile the rule ends on after K steps, against the mean of the online losses (the alternative
objective), for soft.relay.entry; where does each put its optimum, and does the window move it?"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from loom import meta, tasks, tile  # noqa: E402
from loom.signals import Signal  # noqa: E402

SIG = Signal("soft", "relay", "entry")
ETAS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
WINDOWS = {"W=1": (1, 64), "W=4": (4, 32), "W=all": (None, 16)}
SEEDS = range(3)


def setup(seed):
    k_task, k_tile, k_run = jax.random.split(jax.random.key(seed), 3)
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    return tile.init(k_tile, (4, 16, 8, 2)), x, y, k_run


for name, (window, steps) in WINDOWS.items():
    print(f"\n{name}, K={steps}: rows = eta; cells = J_final / mean-online, per seed; bold = mean")
    for eta in ETAS:
        finals, means = [], []
        for seed in SEEDS:
            t, x, y, k = setup(seed)
            J, before, _ = meta.rollout(eta, t, x, y, k, signal=SIG, steps=steps, window=window)
            finals.append(float(J))
            means.append(float(jnp.mean(before)))
        cells = " · ".join(f"{f:.4f}/{m:.4f}" for f, m in zip(finals, means, strict=True))
        print(f"  eta={eta:>7.1f}  {cells}   **{sum(finals) / 3:.4f} / {sum(means) / 3:.4f}**")
