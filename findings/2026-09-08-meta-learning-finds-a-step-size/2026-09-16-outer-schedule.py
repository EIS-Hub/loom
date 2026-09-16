"""The outer schedule chosen on ten seeds: `ONLINE_META`'s inner condition under five settings of
(outer step, outer steps, batch), through `recipes.train`; per run the medians of η and J over
the last 200 outer steps, the min and max of η over that tail (the excursion), and whether the
whole tail sits on the landscape's floor [20, 50]. One process per (setting, seed), run in
parallel on the CPU (fifty processes); the lines are collected into the note's table. A: the
first draft's round-2 schedule; D became the recipe."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys  # noqa: E402
import time  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from loom import recipes, tasks  # noqa: E402
from loom.recipes import ONLINE_META  # noqa: E402

TASK = tasks.junta(4, 2, k=2)
LO, HI = 20.0, 50.0  # the floor of the swept landscape (2026-09-16-landscape.py)
BASE = ONLINE_META._replace(lr=0.03, outer_steps=500, batch=16)
SETTINGS = {
    "A lr0.03 n500 b16": BASE,
    "B lr0.015 n800 b16": BASE._replace(lr=0.015, outer_steps=800),
    "C lr0.03 n500 b32": BASE._replace(batch=32),
    "D lr0.01 n1000 b16": BASE._replace(lr=0.01, outer_steps=1000),
    "E lr0.015 n800 b32": BASE._replace(lr=0.015, outer_steps=800, batch=32),
}

name, seed = sys.argv[1], int(sys.argv[2])
m = SETTINGS[name]
t0 = time.time()
etas, losses, _ = recipes.train(m, TASK, seed)
te, tj = float(jnp.median(etas[-200:])), float(jnp.median(losses[-200:]))
lo, hi = float(jnp.min(etas[-200:])), float(jnp.max(etas[-200:]))
verdict = "floor" if LO <= lo and hi <= HI else ("median-on-floor" if LO <= te <= HI else "OFF")
print(
    f"{name} | seed {seed}: {te:5.1f} / {tj:.4f}   tail {lo:5.1f}-{hi:6.1f}   {verdict}"
    f"   ({time.time() - t0:.0f}s)",
    flush=True,
)
