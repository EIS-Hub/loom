"""Probe: plain descent, Δ = −lr·s, swept over the rate on the flat tile (hidden `(16, 8)` at
arity 4, built to the task's width): 2-juntas and 2-bit addition; the reference, straight-through,
and the relay to the entry on both passes, batched; the reference and the soft relay online
(window 1). Hard accuracy at fixed budgets, probe seeds 0–2, and the first step at which every
seed sits at 1.000. Which rate does each signal want once nothing is normalised, and how many
steps does the reference need?

One batched run per cell: `jax.vmap` over seeds and over rates (the rate is an array), so a cell
is one process. The step is the one line of `descent.descend`, written here so it can be vmapped.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import sys
import time

import jax
import jax.numpy as jnp

from loom import tasks, tile
from loom.signals import REFERENCE, Signal, compute
from loom.tile import Tile

HIDDEN, ARITY, SEEDS = (16, 8), 4, 3  # the tile is built to the task's width: (4, 16, 8, n_out)
RATES = jnp.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0])
# the brief's grid stopped at 3000; the reference had no upper edge there, so it goes on to 1e5
EVERY = 50  # the record's grain: the first step at 1.000 is known to this many steps
TASKS = {"2-junta 4→2": tasks.junta(4, 2, k=2), "addition 2-bit": tasks.addition(4)}
BATCHED = [
    REFERENCE,
    Signal("hard"),
    Signal("soft", "relay", "entry"),
    Signal("hard", "relay", "entry"),
]
ONLINE = [REFERENCE, Signal("soft", "relay", "entry")]


def draw(seed, task):
    """A probe seed: the tile from key(seed) as the earlier probes did, the task from a fold."""
    x, y = task(jax.random.fold_in(jax.random.key(seed), 1))
    widths = (x.shape[1], *HIDDEN, y.shape[1])
    return tile.init(jax.random.key(seed), widths, ARITY), x, y


def sweep(signal, task, rates, steps, window):
    """Hard accuracy [rates, seeds, steps // EVERY] of plain descent on every (rate, seed)."""
    t, x, y = jax.vmap(lambda s: draw(s, task))(jnp.arange(SEEDS))  # y per seed (juntas differ)
    n = x.shape[1]

    def segment(logits, wires, y, lr, key):
        def step(logits, k):
            idx = jnp.arange(n) if window is None else jax.random.choice(k, n, (window,))
            s = compute(signal, Tile(logits, wires), x[0][idx], y[idx])
            return tuple(lg - lr * g for lg, g in zip(logits, s, strict=True)), None

        logits, _ = jax.lax.scan(step, logits, jax.random.split(key, EVERY))
        return logits, tile.accuracy(Tile(logits, wires), x[0], y, "hard")

    over_seeds = jax.vmap(segment, in_axes=(0, 0, 0, None, 0))
    batched = jax.jit(jax.vmap(over_seeds, in_axes=(0, None, None, 0, None)))
    logits = tuple(jnp.broadcast_to(lg, (len(rates), *lg.shape)) for lg in t.logits)  # one per rate
    keys = jax.vmap(lambda s: jax.random.key(s))(jnp.arange(SEEDS))
    record = []
    for i in range(steps // EVERY):
        keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, i)
        logits, acc = batched(logits, t.wires, y, rates, keys)
        record.append(acc)
    return jnp.stack(record, axis=-1)


def report(signal, task_name, rec, rates, budgets, window):
    label = signal.label + (" W=1" if window == 1 else "")
    print(f"\n{task_name}: {label}, hard accuracy at " + " / ".join(map(str, budgets)) + " steps")
    print(
        f"{'rate':>6s}  "
        + "  ".join(f"{'seed ' + str(s):>21s}" for s in range(SEEDS))
        + f"  {'mean ± sd (last)':>18s}  {'at 1.000':>8s}  {'all at 1.000 by':>15s}"
    )
    for r, row in zip(rates.tolist(), rec, strict=True):
        cols = [
            " / ".join(f"{float(row[s, b // EVERY - 1]):.3f}" for b in budgets)
            for s in range(SEEDS)
        ]
        last = row[:, budgets[-1] // EVERY - 1]
        every_seed = jnp.all(row == 1.0, axis=0)  # [steps // EVERY]
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
    print(f"hidden {HIDDEN} arity {ARITY}, plain descent Δ = −lr·s, seeds 0-2, CPU")
    for task_name, task in TASKS.items():
        for signal in BATCHED:
            if only and signal.label not in only:
                continue
            t1 = time.time()
            rec = sweep(signal, task, RATES, 2000, None)
            report(signal, task_name, rec, RATES, (500, 2000), None)
            print(f"  ({time.time() - t1:.0f} s)")
        for signal in ONLINE:
            if only and signal.label + " W=1" not in only:
                continue
            t1 = time.time()
            rec = sweep(signal, task, RATES, 3000, 1)
            report(signal, task_name, rec, RATES, (1000, 3000), 1)
            print(f"  ({time.time() - t1:.0f} s)")
    print(f"\nwall-clock {time.time() - t0:.0f} s")
