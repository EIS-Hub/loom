"""The fixed-η baseline on the objective the loop descends (`recipes.landscape`: the mean soft
loss over the members the seed's first outer step draws), across η, under the true signal and its
three controls: online (`ONLINE_META`, W = 1, K = 64), and with every case per step
(`BATCHED_META`, K = 16) for the path. Where is the optimum, how wide is the floor, and does a
control's landscape have one at all? Cells: J per seed, then the mean and the deployed loss."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from loom import recipes, tasks  # noqa: E402
from loom.recipes import BATCHED_META, ONLINE_META  # noqa: E402

TASK = tasks.junta(4, 2, k=2)
ETAS = (3, 10, 20, 30, 50, 100, 200, 400, 1000)
CONTROLS = ("none", "flipped", "shuffled", "output_only")
SEEDS = range(3)

for name, base in (("ONLINE_META", ONLINE_META), ("BATCHED_META", BATCHED_META)):
    for control in CONTROLS:
        m = base._replace(control=control)
        head = f"\n{name} {m.label}, control={control}: rows η; cells J per seed"
        print(head + "; then the mean J / the mean deployed loss")
        for eta in ETAS:
            Js, Ds = zip(*(recipes.landscape(m, float(eta), TASK, s) for s in SEEDS), strict=True)
            cells = " · ".join(f"{float(j):.4f}" for j in Js)
            mean = f"**{sum(map(float, Js)) / 3:.4f} / {sum(map(float, Ds)) / 3:.4f}**"
            print(f"  η={eta:>5}: {cells}   {mean}", flush=True)
