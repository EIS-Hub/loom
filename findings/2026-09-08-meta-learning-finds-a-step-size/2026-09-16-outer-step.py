"""The outer step: `ONLINE_META` at an outer step of 0.1 for 200 steps (the first draft's
schedule) on ten seeds, η along the run and the held-out accuracy at the end. At 0.03 for 500
steps (the recipe as pinned) every seed settles on the optimum: `2026-09-16-learned.py`. Here:
how often the faster climb overshoots the optimum and runs away."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from loom import recipes, tasks, tile  # noqa: E402
from loom.recipes import ONLINE_META  # noqa: E402

TASK = tasks.junta(4, 2, k=2)
FAST = ONLINE_META._replace(lr=0.1, outer_steps=200)
AT = (25, 50, 75, 100, 150, 200)

print(f"{FAST.label}, outer step 0.1, 200 steps: η/J at outer steps {AT}; held-out at the η found")
for seed in range(10):
    etas, losses, _ = recipes.train(FAST, TASK, seed)
    t, x, y = recipes.adapt(FAST, float(etas[-1]), TASK, seed, steps=500)
    cells = " · ".join(f"{float(etas[i - 1]):.3g}/{float(losses[i - 1]):.3f}" for i in AT)
    print(
        f"  seed {seed}: {cells}   held-out {float(tile.accuracy(t, x, y, 'hard')):.3f}", flush=True
    )
