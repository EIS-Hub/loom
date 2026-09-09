"""Recipes: training conditions as data, and the one place a condition meets a task.

A claim is a (recipe, check) pair, so the same check under another step size is another named
recipe and a different claim. Claims and notes refer to recipes by name; only probes may build
unnamed variants, because exploring the condition space is their job.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax

from loom.descent import fit, trajectory
from loom.signals import REFERENCE, Signal
from loom.tile import Tile, init

Task = Callable[[jax.Array], tuple[jax.Array, jax.Array]]  # key → (x, y)


class DescentRecipe(NamedTuple):
    """The condition of direct descent on the tables, Δ = −lr·s: what ``descent.fit`` runs."""

    signal: Signal  # which signal drives descent
    lr: float  # the plain rate, in logit units per unit average vote (one vote per case)
    steps: int
    window: int | None  # None: every case per step, the batched default; 1: fully online
    hidden: tuple[int, ...]  # widths between the task's inputs and outputs
    arity: int = 4
    scale: float = 1.0  # init scale of the logits


# Rates sit at the centre of each signal's working plateau on the probe seeds and budgets at twice
# the first step at which every probe seed reached the target, except at depth, where the plain
# reference has no rate that reaches on every tile and the budget is the tables' full 4000, so the
# claim is about the landscape, not the clock (notes/2026-09-08-the-floors-under-plain-descent.md).
SOFT_FLOOR = DescentRecipe(REFERENCE, lr=50.0, steps=400, window=None, hidden=(16, 8))
HARD_FLOOR = DescentRecipe(Signal("hard"), lr=150.0, steps=3000, window=None, hidden=(16, 8))
ONLINE_FLOOR = DescentRecipe(REFERENCE, lr=50.0, steps=500, window=1, hidden=(16, 8))
DEEP_FLOOR = DescentRecipe(
    REFERENCE, lr=750.0, steps=4000, window=None, hidden=(32, 32, 16), arity=3
)
DEEP_HARD_FLOOR = DescentRecipe(
    Signal("hard"), lr=250.0, steps=2000, window=None, hidden=(32, 32, 16), arity=3
)
# The soft relay to the entry wants its own rate at depth: the reference's leaves it at chance.
DEEP_RELAY = DEEP_FLOOR._replace(signal=Signal("soft", "relay", "entry"), lr=375.0)


def setup(
    recipe: DescentRecipe, task: Task, seed: int
) -> tuple[Tile, jax.Array, jax.Array, jax.Array]:
    """The task drawn and the tile built to its width, from one seed."""
    k_task, k_tile, k_run = jax.random.split(jax.random.key(seed), 3)
    x, y = task(k_task)
    tile = init(k_tile, (x.shape[1], *recipe.hidden, y.shape[1]), recipe.arity, recipe.scale)
    return tile, x, y, k_run


def run(recipe: DescentRecipe, task: Task, seed: int) -> tuple[Tile, jax.Array, jax.Array]:
    """Train under the recipe: the fitted tile and the cases it was fitted on."""
    tile, x, y, key = setup(recipe, task, seed)
    r = recipe
    return fit(tile, x, y, r.steps, lr=r.lr, window=r.window, key=key, signal=r.signal), x, y


def trace(recipe: DescentRecipe, task: Task, seed: int, every: int = 10):
    """Train under the recipe while recording the deploy gap: (tile, record, x, y)."""
    tile, x, y, key = setup(recipe, task, seed)
    r = recipe
    t, rec = trajectory(tile, x, y, r.steps, every, r.signal, lr=r.lr, window=r.window, key=key)
    return t, rec, x, y
