"""Probe: is the bits finding an artefact of arity 3? The same transports at arity 4 on the two deep
shapes (four layers, and the three-layer shape the reference also solves), 6-bit addition, three
seeds. Asked by Gabriel in review of PR #4: "why change to arity 3 all of the sudden?" """

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax

from loom import tasks, tile
from loom.descent import trajectory
from loom.signals import Signal

x, y = tasks.add(6)
for widths in [(6, 32, 32, 16, 4), (6, 32, 16, 4)]:
    print(f"shape {widths} arity 4, addition: hard acc at 500 / 2000, seeds 0-2")
    for sig, lr in (
        (Signal("soft", "relay", "entry"), 0.1),
        (Signal("soft", "uniform", "entry"), 0.1),
        (Signal("hard", "relay", "entry"), 0.02),
        (Signal("hard", "uniform", "entry"), 0.02),
    ):
        cols = []
        for seed in range(3):
            t = tile.init(jax.random.key(seed), widths, 4)
            _, rec = trajectory(t, x, y, steps=2000, every=500, signal=sig, lr=lr)
            cols.append(f"{float(rec['hard'][0]):.3f}/{float(rec['hard'][-1]):.3f}")
        print(f"  {sig.label:16s} lr {lr:<5}  " + "  ".join(cols), flush=True)
