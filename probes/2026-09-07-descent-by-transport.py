"""Probe: Adam driven by each transport on the deep shape (6-input addition, arity 3): hard
accuracy at 500 and 2000 steps, two rates, three seeds. Which signals can descent follow down a
carry chain, and does the value-blind split fail where the relay does not?"""

import jax

from loom import tasks, tile
from loom.descent import trajectory
from loom.signals import Signal

WIDTHS, ARITY = (6, 32, 32, 16, 4), 3
x, y = tasks.add(6)
CELLS = [
    Signal(on, via, to)
    for on in ("soft", "hard")
    for via, to in (
        ("relay", "logit"),
        ("relay", "entry"),
        ("uniform", "entry"),
        ("direct", "entry"),
        ("flip", "entry"),
    )
]
print(f"shape {WIDTHS} arity {ARITY}, addition: hard accuracy at step 500 / 2000, seeds 0-2")
print(f"{'signal':18s}{'lr':>6s}   {'seed 0':>13s}  {'seed 1':>13s}  {'seed 2':>13s}")
for sig in CELLS:
    for lr in (0.1, 0.02):
        cols = []
        for seed in range(3):
            t = tile.init(jax.random.key(seed), WIDTHS, ARITY)
            _, rec = trajectory(t, x, y, steps=2000, every=500, signal=sig, lr=lr)
            cols.append(f"{float(rec['hard'][0]):.3f} / {float(rec['hard'][-1]):.3f}")
        print(f"{sig.label:18s}{lr:6.2f}   " + "  ".join(cols))
