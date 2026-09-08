"""Probe: the deep pins under the recipe seeds. `recipes.setup` splits a seed's key differently
from the probes, so a rate pinned at the centre of a plateau measured on probe seeds is checked
here on the tiles the claims actually train: the reference and the relay to the entry on the deep
shape, 6-bit addition, at the pinned rate and its ×1.5 neighbours, 4000 steps, recipe seeds 0–2;
per-seed hard accuracy at 1000 / 2000 / 4000 and the first step at which every seed sits at
1.000. One batched run per cell (`jax.vmap` over seeds and rates); the step is the one line of
`descent.descend`."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import time

import jax
import jax.numpy as jnp

from loom import recipes, tasks, tile
from loom.recipes import DEEP_FLOOR, DEEP_RELAY
from loom.signals import compute
from loom.tile import Tile

SEEDS, STEPS, EVERY, BUDGETS = 3, 4000, 250, (1000, 2000, 4000)
ADD = tasks.addition(6)
CELLS = [(DEEP_FLOOR, (2000.0, 3000.0, 4500.0)), (DEEP_RELAY, (700.0, 1000.0, 1500.0, 2000.0))]


def sweep(recipe, rates):
    t, x, y, _ = jax.vmap(lambda s: recipes.setup(recipe, ADD, s))(jnp.arange(SEEDS))

    def segment(logits, wires, y, lr):
        def step(logits, _):
            s = compute(recipe.signal, Tile(logits, wires), x[0], y)
            return tuple(lg - lr * g for lg, g in zip(logits, s, strict=True)), None

        logits, _ = jax.lax.scan(step, logits, None, length=EVERY)
        return logits, tile.accuracy(Tile(logits, wires), x[0], y, "hard")

    over_seeds = jax.vmap(segment, in_axes=(0, 0, 0, None))
    batched = jax.jit(jax.vmap(over_seeds, in_axes=(0, None, None, 0)))
    logits = tuple(jnp.broadcast_to(lg, (len(rates), *lg.shape)) for lg in t.logits)
    record = []
    for _ in range(STEPS // EVERY):
        logits, acc = batched(logits, t.wires, y, rates)
        record.append(acc)
    return jnp.stack(record, axis=-1)


t0 = time.time()
print("deep shape, addition, plain descent, RECIPE seeds 0-2 (recipes.setup), CPU")
for recipe, rates in CELLS:
    rec = sweep(recipe, jnp.array(rates))
    at = " / ".join(map(str, BUDGETS))
    print(
        f"\n{recipe.signal.label} (pinned at {recipe.lr:.0f} for {recipe.steps}): hard acc at {at}"
    )
    print(
        f"{'rate':>6s}  "
        + "  ".join(f"{'seed ' + str(s):>21s}" for s in range(SEEDS))
        + f"  {'mean ± sd (last)':>18s}  {'all at 1.000 by':>15s}"
    )
    for r, row in zip(rates, rec, strict=True):
        cols = [
            " / ".join(f"{float(row[s, b // EVERY - 1]):.3f}" for b in BUDGETS)
            for s in range(SEEDS)
        ]
        last = row[:, -1]
        every_seed = jnp.all(row == 1.0, axis=0)
        first = int(jnp.argmax(every_seed)) + 1 if bool(jnp.any(every_seed)) else None
        by = str(first * EVERY) if first else "never"
        print(
            f"{r:6.0f}  "
            + "  ".join(f"{c:>21s}" for c in cols)
            + f"  **{float(jnp.mean(last)):.3f} ± {float(jnp.std(last)):.3f}**  {by:>15s}"
        )
print(f"\nwall-clock {time.time() - t0:.0f} s")
