"""Probe: plain descent, Δ = −lr·s, swept over the rate on step 1's landmark shape
`(6, 32, 32, 16, 4)` at arity 3 (four hops for the error): 6-bit addition and 2-juntas 6→4. On the
soft pass the reference, the relay, the uniform split, the random bus and the bus masked to
reachable outputs, all to the entry; on the bits the same five (autodiff being straight-through)
and the flip credit. Hard accuracy at 1000 / 2000 / 4000 steps, probe seeds 0–2, the fraction of
runs at 1.000 and the first step at which every seed sits there. Which of chunk 3's findings, all
made under Adam, hold when the signal's magnitude is the step?

One batched run per cell: `jax.vmap` over seeds and over rates (the rate is an array), so a cell
is one process. The step is the one line of `descent.descend`, written here so it can be vmapped.
The tile is drawn from `key(seed)` as the earlier deep probes did, so the addition tiles are theirs.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import sys
import time

import jax
import jax.numpy as jnp

from loom import tasks, tile
from loom.signals import Signal, compute
from loom.tile import Tile

WIDTHS, ARITY, SEEDS = (6, 32, 32, 16, 4), 3, 3
RATES = jnp.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0])
STEPS, EVERY, BUDGETS = 4000, 250, (1000, 2000, 4000)
TASKS = {"addition 6-bit": tasks.addition(6), "2-junta 6→4": tasks.junta(6, 4, k=2)}
CELLS = [
    Signal(on, via, to)
    for on in ("soft", "hard")
    for via, to in (
        ("autodiff", "logit"),
        ("relay", "entry"),
        ("uniform", "entry"),
        ("direct", "entry"),
        ("reachable", "entry"),
        ("flip", "entry"),
    )
    if not (on == "soft" and via == "flip")  # a bits quantity
]


def draw(seed, task):
    """A probe seed: the tile from key(seed) as the earlier probes did, the task from a fold."""
    x, y = task(jax.random.fold_in(jax.random.key(seed), 1))
    return tile.init(jax.random.key(seed), WIDTHS, ARITY), x, y


def sweep(signal, task, rates):
    """Hard accuracy [rates, seeds, STEPS // EVERY] of plain descent on every (rate, seed)."""
    t, x, y = jax.vmap(lambda s: draw(s, task))(jnp.arange(SEEDS))  # y per seed (juntas differ)

    def segment(logits, wires, y, lr, key):
        def step(logits, k):
            s = compute(signal, Tile(logits, wires), x[0], y)
            return tuple(lg - lr * g for lg, g in zip(logits, s, strict=True)), None

        logits, _ = jax.lax.scan(step, logits, jax.random.split(key, EVERY))
        return logits, tile.accuracy(Tile(logits, wires), x[0], y, "hard")

    over_seeds = jax.vmap(segment, in_axes=(0, 0, 0, None, 0))
    batched = jax.jit(jax.vmap(over_seeds, in_axes=(0, None, None, 0, None)))
    logits = tuple(jnp.broadcast_to(lg, (len(rates), *lg.shape)) for lg in t.logits)  # one per rate
    keys = jax.vmap(lambda s: jax.random.key(s))(jnp.arange(SEEDS))
    record = []
    for i in range(STEPS // EVERY):
        keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, i)
        logits, acc = batched(logits, t.wires, y, rates, keys)
        record.append(acc)
    return jnp.stack(record, axis=-1)


def report(signal, task_name, rec, rates):
    at = " / ".join(map(str, BUDGETS))
    print(f"\n{task_name}: {signal.label}, hard accuracy at {at} steps")
    print(
        f"{'rate':>6s}  "
        + "  ".join(f"{'seed ' + str(s):>21s}" for s in range(SEEDS))
        + f"  {'mean ± sd (last)':>18s}  {'at 1.000':>8s}  {'all at 1.000 by':>15s}"
    )
    for r, row in zip(rates.tolist(), rec, strict=True):
        cols = [
            " / ".join(f"{float(row[s, b // EVERY - 1]):.3f}" for b in BUDGETS)
            for s in range(SEEDS)
        ]
        last = row[:, -1]
        every_seed = jnp.all(row == 1.0, axis=0)  # [STEPS // EVERY]
        first = int(jnp.argmax(every_seed)) + 1 if bool(jnp.any(every_seed)) else None
        by = f"{first * EVERY:d}" if first else "never"
        print(
            f"{r:6.0f}  "
            + "  ".join(f"{c:>21s}" for c in cols)
            + f"  **{float(jnp.mean(last)):.3f} ± {float(jnp.std(last)):.3f}**"
            + f"  {float(jnp.mean(last == 1.0)):8.2f}  {by:>15s}"
        )


if __name__ == "__main__":
    only = sys.argv[1:]  # optional: labels to run, for a quick look
    t0 = time.time()
    print(f"shape {WIDTHS} arity {ARITY}, plain descent Δ = −lr·s, seeds 0-2, CPU")
    for task_name, task in TASKS.items():
        for signal in CELLS:
            if only and signal.label not in only:
                continue
            t1 = time.time()
            report(signal, task_name, sweep(signal, task, RATES), RATES)
            print(f"  ({time.time() - t1:.0f} s)", flush=True)
    print(f"\nwall-clock {time.time() - t0:.0f} s")
