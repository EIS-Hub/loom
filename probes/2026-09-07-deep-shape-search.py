"""Probe: which shapes does the reference solve on 6-input addition? The first deep shape tried,
(6, 24, 12, 4) at arity 3, left the reference at 0.93-0.98 hard accuracy: not a floor. Widths and
arity swept to find one it solves on every seed."""

import jax

from loom import tasks, tile
from loom.descent import trajectory

x, y = tasks.add(6)
print("reference on 6-input addition, hard acc at 1000 / 2000 (lr 0.1), 3 seeds")
for widths in [
    (6, 24, 12, 4),
    (6, 32, 16, 4),
    (6, 48, 24, 4),
    (6, 32, 32, 16, 4),
    (6, 48, 48, 24, 4),
]:
    for arity in (3, 4):
        cols = []
        for seed in range(3):
            t = tile.init(jax.random.key(seed), widths, arity)
            _, rec = trajectory(t, x, y, steps=2000, every=1000, lr=0.1)
            cols.append(f"{float(rec['hard'][0]):.3f}/{float(rec['hard'][-1]):.3f}")
        print(f"{str(widths):24s} arity {arity}   " + "  ".join(cols), flush=True)
