"""The learned column on the same axes as the landscape: `ONLINE_META` under the true signal and
its three controls (`recipes.train`, six seeds), η and J along the run and the deployed loss of
the final states at the end; then the η each run ends on, driven for 500 steps on a fresh
held-out tile and task by the control's own signal (what that rule achieves) and by the true
signal (whether the η it parked at is a working step size at all). `BATCHED_META` under the true
signal, three seeds, for the path: the regime of the first draft."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time  # noqa: E402

from loom import recipes, tasks, tile  # noqa: E402
from loom.recipes import BATCHED_META, ONLINE_META  # noqa: E402

TASK = tasks.junta(4, 2, k=2)
CONTROLS = ("none", "flipped", "shuffled", "output_only")


def row(m, seed, at):
    t0 = time.time()
    etas, losses, deployed = recipes.train(m, TASK, seed)
    eta = float(etas[-1])
    accs = []
    for control in (m.control, "none"):
        t, x, y = recipes.adapt(m._replace(control=control), eta, TASK, seed, steps=500)
        accs.append(float(tile.accuracy(t, x, y, "hard")))
    cells = " · ".join(f"{float(etas[i - 1]):.3g}/{float(losses[i - 1]):.4f}" for i in at)
    print(
        f"  seed {seed}: {cells}   deployed {float(deployed[-1]):.4f}"
        f"   held-out {accs[0]:.3f} / {accs[1]:.3f}   ({time.time() - t0:.0f}s)",
        flush=True,
    )


for control in CONTROLS:
    m = ONLINE_META._replace(control=control)
    head = f"\nONLINE_META {m.label}, control={control}: eta/J at outer steps 100 ... 500"
    print(head + "; deployed loss at 500; held-out under the control / under the true signal")
    for seed in range(6):
        row(m, seed, (100, 200, 300, 400, 500))

head = f"\nBATCHED_META {BATCHED_META.label}, control=none: eta/J at outer steps 50, 100, 200"
print(head + "; deployed loss at 200; held-out under the true signal, twice")
for seed in range(3):
    row(BATCHED_META, seed, (50, 100, 200))
