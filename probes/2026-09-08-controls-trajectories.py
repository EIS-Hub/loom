"""The outer loop from a non-functional start under the true soft relay and its three controls,
on the flat tile: eta and J along 200 outer steps (batch 16, K=16, every case per step), then
the eta each run ends on, driven by the TRUE signal for 500 plain steps on a fresh held-out
2-junta (the deployed check for the true signal; for a control it only says whether the eta it
wandered to is a working step size)."""

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import itertools  # noqa: E402

import jax  # noqa: E402

from loom import meta, rule, tasks, tile  # noqa: E402
from loom.signals import Signal  # noqa: E402

SIG = Signal("soft", "relay", "entry")
TASK = tasks.junta(4, 2, k=2)
CONTROLS = ("none", "flipped", "shuffled", "output_only")
AT = (50, 100, 200)

for control in CONTROLS:
    print(
        f"\ncontrol={control}: eta / J at outer steps {AT}; then held-out accuracy, 500 plain steps"
    )
    for seed in range(3):
        k_train, k_held = jax.random.split(jax.random.key(seed))
        t0 = time.time()
        it = meta.learn(
            rule.init(),
            k_train,
            TASK,
            batch=16,
            hidden=(16, 8),
            signal=SIG,
            steps=16,
            control=control,
        )
        trace = {}
        params = rule.init()
        for i, (params, J) in enumerate(itertools.islice(it, 200), start=1):
            if i in AT:
                trace[i] = (float(rule.rate(params)), float(J))
        eta = float(rule.rate(params))
        k_task, k_tile, k_run = jax.random.split(k_held, 3)
        x, y = TASK(k_task)
        fresh = tile.init(k_tile, (4, 16, 8, 2))
        _, _, adapted = meta.rollout(eta, fresh, x, y, k_run, signal=SIG, steps=500)
        acc = float(tile.accuracy(adapted, x, y, "hard"))
        cells = " · ".join(f"{trace[i][0]:.3g}/{trace[i][1]:.4f}" for i in AT if i in trace)
        print(f"  seed {seed}: {cells}   held-out {acc:.3f}   ({time.time() - t0:.0f}s)")
