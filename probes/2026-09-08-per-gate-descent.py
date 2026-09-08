"""Probe: descent on the per-gate signal, ``to="gate"``: every entry of a gate moves by the
gate's summed error, with no address factor. A table whose entries all move together can only
learn a bias, so this should fail everywhere; the record keeps the failure rather than the
argument. The deep shape, Adam, 2000 steps, three seeds, against the per-entry cells."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax

from loom import tasks, tile
from loom.descent import fit
from loom.signals import Signal

WIDTHS, ARITY, SEEDS = (6, 32, 32, 16, 4), 3, 3
x, y = tasks.add(6)
CELLS = (
    (Signal("soft", "relay", "gate"), 0.1),
    (Signal("soft", "relay", "entry"), 0.1),
    (Signal("hard", "uniform", "gate"), 0.02),
    (Signal("hard", "uniform", "entry"), 0.02),
    (Signal("hard", "reachable", "gate"), 0.02),
)
print(f"shape {WIDTHS} arity {ARITY}, addition, 2000 steps: hard accuracy, seeds 0-2")
for sig, lr in CELLS:
    accs = []
    for seed in range(SEEDS):
        t = fit(tile.init(jax.random.key(seed), WIDTHS, ARITY), x, y, 2000, lr=lr, signal=sig)
        accs.append(float(tile.accuracy(t, x, y, "hard")))
    print(f"{sig.label:24s}" + "  ".join(f"{a:.3f}" for a in accs), flush=True)
