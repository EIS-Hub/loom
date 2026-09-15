"""Probe: the rate each signal wants. The two coarse sweeps (`…-flat.py`, `…-deep.py`, ×3-spaced
decades) left three plateaus too narrow or too edge-bound to pin a recipe at their centre; this
refines them ×1.5 and asks whether a low edge is a rate edge or a budget edge (the same rates for
three times the steps). Then a summary: per (shape, pass, signal) the working plateau, its
geometric centre and the earliest budget, so a recipe is pinned at a centre, never an edge, and
so the spread of best rates across signals is on the record (more than a decade apart means a
claim cannot swap the signal under one recipe).

The working plateau of a signal that reaches the target is the run of rates at which every seed
sits at 1.000 by the budget; of one that does not, the rates whose mean at the budget is within
0.02 of its best. One batched run per cell (`jax.vmap` over seeds and rates); the step is the one
line of `descent.descend`.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import math
import time

import jax
import jax.numpy as jnp

from loom import tasks, tile
from loom.signals import REFERENCE, Signal, compute
from loom.tile import Tile

SEEDS = 3
FLAT, DEEP = ((16, 8), 4), ((32, 32, 16), 3)  # (hidden, arity); the tile is built to the task
ADD2, JUNTA2 = tasks.addition(4), tasks.junta(4, 2, k=2)
ADD6, JUNTA6 = tasks.addition(6), tasks.junta(6, 4, k=2)
ST, RELAY = Signal("hard"), Signal("soft", "relay", "entry")

# (name, shape, task, signal, window, rates, steps, every)
CELLS = [
    (
        "deep addition, the reference, ×1.5 around 3000",
        DEEP,
        ADD6,
        REFERENCE,
        None,
        (1000, 1500, 2000, 3000, 4500, 7000, 10000),
        4000,
        250,
    ),
    (
        "deep addition, the reference, the low edge at 3× the budget",
        DEEP,
        ADD6,
        REFERENCE,
        None,
        (100, 300, 1000),
        12000,
        500,
    ),
    (
        "deep addition, soft.relay.entry, ×1.5 around 1000",
        DEEP,
        ADD6,
        RELAY,
        None,
        (300, 500, 700, 1000, 1500, 2000),
        4000,
        250,
    ),
    (
        "deep addition, soft.relay.entry, the same rates at 3× the budget",
        DEEP,
        ADD6,
        RELAY,
        None,
        (300, 500, 700, 1000, 1500, 2000),
        12000,
        500,
    ),
    (
        "deep addition, straight-through, ×1.5 around 1000",
        DEEP,
        ADD6,
        ST,
        None,
        (300, 500, 700, 1000, 1500, 2000, 3000),
        4000,
        250,
    ),
    (
        "deep 2-junta, straight-through, ×1.5 around 300–1000",
        DEEP,
        JUNTA6,
        ST,
        None,
        (100, 150, 200, 300, 450, 700, 1000, 1500, 2000),
        4000,
        250,
    ),
    (
        "flat addition, straight-through, ×1.5 around 300",
        FLAT,
        ADD2,
        ST,
        None,
        (100, 150, 200, 300, 450, 700, 1000),
        2000,
        50,
    ),
    (
        "flat 2-junta, straight-through, ×1.5 where addition's plateau is",
        FLAT,
        JUNTA2,
        ST,
        None,
        (200, 300, 450, 700, 1000),
        2000,
        50,
    ),
    (
        "flat 2-junta, the reference online, ×1.5 inside 10–1000",
        FLAT,
        JUNTA2,
        REFERENCE,
        1,
        (30, 50, 70, 100, 150, 200, 300),
        3000,
        50,
    ),
]


def draw(seed, task, hidden, arity):
    """A probe seed: the tile from key(seed) as the earlier probes did, the task from a fold."""
    x, y = task(jax.random.fold_in(jax.random.key(seed), 1))
    return tile.init(jax.random.key(seed), (x.shape[1], *hidden, y.shape[1]), arity), x, y


def sweep(shape, task, signal, window, rates, steps, every):
    """Hard accuracy [rates, seeds, steps // every] of plain descent on every (rate, seed)."""
    hidden, arity = shape
    t, x, y = jax.vmap(lambda s: draw(s, task, hidden, arity))(jnp.arange(SEEDS))
    n = x.shape[1]

    def segment(logits, wires, y, lr, key):
        def step(logits, k):
            idx = jnp.arange(n) if window is None else jax.random.choice(k, n, (window,))
            s = compute(signal, Tile(logits, wires), x[0][idx], y[idx])
            return tuple(lg - lr * g for lg, g in zip(logits, s, strict=True)), None

        logits, _ = jax.lax.scan(step, logits, jax.random.split(key, every))
        return logits, tile.accuracy(Tile(logits, wires), x[0], y, "hard")

    over_seeds = jax.vmap(segment, in_axes=(0, 0, 0, None, 0))
    batched = jax.jit(jax.vmap(over_seeds, in_axes=(0, None, None, 0, None)))
    logits = tuple(jnp.broadcast_to(lg, (len(rates), *lg.shape)) for lg in t.logits)
    keys = jax.vmap(lambda s: jax.random.key(s))(jnp.arange(SEEDS))
    record = []
    for i in range(steps // every):
        keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, i)
        logits, acc = batched(logits, t.wires, y, rates, keys)
        record.append(acc)
    return jnp.stack(record, axis=-1)


def plateau(rates, rec, every):
    """(lo, hi, centre, earliest budget, reaches the target) of the working plateau."""
    last = rec[:, :, -1]
    every_seed = jnp.all(rec == 1.0, axis=1)  # [rates, steps // every]
    reaches = jnp.any(every_seed, axis=1)
    if bool(jnp.any(reaches)):
        idx = [i for i in range(len(rates)) if bool(reaches[i])]
        first = [int(jnp.argmax(every_seed[i]) + 1) * every for i in idx]
    else:
        means = jnp.mean(last, axis=1)
        idx = [i for i in range(len(rates)) if float(means[i]) >= float(jnp.max(means)) - 0.02]
        first = []
    lo, hi = rates[idx[0]], rates[idx[-1]]
    return lo, hi, math.sqrt(lo * hi), (min(first) if first else None), bool(jnp.any(reaches))


def report(name, rates, rec, steps, every):
    budgets = (steps // 4, steps // 2, steps)
    print(f"\n{name}: hard accuracy at " + " / ".join(map(str, budgets)) + " steps")
    print(
        f"{'rate':>6s}  "
        + "  ".join(f"{'seed ' + str(s):>21s}" for s in range(SEEDS))
        + f"  {'mean ± sd (last)':>18s}  {'at 1.000':>8s}  {'all at 1.000 by':>15s}"
    )
    for r, row in zip(rates, rec, strict=True):
        cols = [
            " / ".join(f"{float(row[s, b // every - 1]):.3f}" for b in budgets)
            for s in range(SEEDS)
        ]
        last = row[:, -1]
        every_seed = jnp.all(row == 1.0, axis=0)
        first = int(jnp.argmax(every_seed)) + 1 if bool(jnp.any(every_seed)) else None
        by = str(first * every) if first else "never"
        print(
            f"{r:6.0f}  "
            + "  ".join(f"{c:>21s}" for c in cols)
            + f"  **{float(jnp.mean(last)):.3f} ± {float(jnp.std(last)):.3f}**"
            + f"  {float(jnp.mean(last == 1.0)):8.2f}  {by:>15s}"
        )


if __name__ == "__main__":
    t0 = time.time()
    print("plain descent Δ = −lr·s, seeds 0-2, CPU; the refinements, then the plateau of each")
    summary = []
    for name, shape, task, signal, window, rates, steps, every in CELLS:
        t1 = time.time()
        rec = sweep(shape, task, signal, window, jnp.array(rates, dtype=jnp.float32), steps, every)
        report(name, rates, rec, steps, every)
        print(f"  ({time.time() - t1:.0f} s)", flush=True)
        summary.append((name, *plateau(rates, rec, every)))
    print(f"\n{'cell':64s}{'plateau':>16s}{'centre':>8s}{'all seeds at 1.000 by':>24s}")
    for name, lo, hi, centre, first, reaches in summary:
        short = "" if reaches else " (short of the target: within 0.02 of its best)"
        span = f"[{lo:.0f}, {hi:.0f}]" + short
        print(f"{name:64s}{span:>16s}{centre:8.0f}{(str(first) if first else 'never'):>24s}")
    print(f"\nwall-clock {time.time() - t0:.0f} s")
