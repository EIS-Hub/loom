"""Probe: descent on the bits against the soft floor, over rate × budget × seed on 2-juntas.
Unnamed recipe variants are built here on purpose: exploring conditions is a probe's job."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

from loom import recipes, tasks, tile
from loom.recipes import HARD_FLOOR, SOFT_FLOOR

task = tasks.junta(4, 2, k=2)
print("seed  signal  steps  rate   hard accuracy")
for seed in range(3):
    for base in (SOFT_FLOOR, HARD_FLOOR):
        for steps, lr in ((500, 0.1), (2000, 0.1), (2000, 0.02)):
            t, x, y = recipes.run(base._replace(steps=steps, lr=lr), task, seed)
            acc = float(tile.accuracy(t, x, y))
            print(f"{seed}     {base.signal.on:4s}    {steps:5d}  {lr:<5}  {acc:.3f}")
