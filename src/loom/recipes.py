"""Recipes: training conditions as data, and the one place a condition meets a task.

A claim is a (recipe, check) pair, so the same check under another step size is another named
recipe and a different claim. Claims and notes refer to recipes by name; only probes may build
unnamed variants, because exploring the condition space is their job.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import islice
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from loom import meta, rule
from loom.descent import fit, trajectory
from loom.meta import Control
from loom.signals import REFERENCE, Signal
from loom.tile import Tile, init

Objective = Literal["soft"]  # what the outer loop differentiates: L^soft of the final state, today

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
    rule: str = "plain"  # a name in rule.RULES: the floor, or a hand-engineered rule to beat


# Rates sit at the centre of each signal's working plateau on the probe seeds and budgets at twice
# the first step at which every probe seed reached the target, except at depth, where the plain
# reference has no rate that reaches on every tile and the budget is the tables' full 4000, so the
# claim is about the landscape, not the clock
# (findings/2026-09-08-the-floors-under-plain-descent/note.md).
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
# The same signal under a readout at the cell, at two rates a decade apart (a band, where the plain
# rule has a point): the sign of the vote with no state, and RMSprop with one accumulator per entry.
DEEP_RELAY_TENTH = DEEP_RELAY._replace(lr=37.5)
DEEP_SIGN = DEEP_RELAY._replace(rule="sign", lr=0.1)
DEEP_SIGN_TENTH = DEEP_SIGN._replace(lr=0.01)
DEEP_RMSPROP = DEEP_RELAY._replace(rule="rmsprop", lr=0.1)
DEEP_RMSPROP_TENTH = DEEP_RMSPROP._replace(lr=0.01)


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
    t = fit(tile, x, y, r.steps, lr=r.lr, window=r.window, key=key, signal=r.signal, rule=r.rule)
    return t, x, y


def trace(recipe: DescentRecipe, task: Task, seed: int, every: int = 10):
    """Train under the recipe while recording the deploy gap: (tile, record, x, y)."""
    tile, x, y, key = setup(recipe, task, seed)
    r = recipe
    t, rec = trajectory(
        tile, x, y, r.steps, every, r.signal, lr=r.lr, window=r.window, key=key, rule=r.rule
    )
    return t, rec, x, y


class MetaRecipe(NamedTuple):
    """The condition of meta-learning the rule, named after the loop it runs (``meta.learn``): an
    inner cell (the signal the rule is fed) and an outer objective, its label the pair."""

    signal: Signal  # the inner cell: the signal the rule is fed inside the rollout
    steps: int  # K, the rollout's length: the horizon credit runs through
    window: int | None  # as in Descent: every case per step, or that many drawn
    batch: int  # states per outer step
    hidden: tuple[int, ...]
    outer_steps: int = 200
    lr: float = 0.1  # the outer optimiser's step, on log η
    eta0: float = 1e-2  # the non-functional start
    control: Control = "none"
    objective: Objective = "soft"  # the one value so far; L^ste is the fallback if the gap bites
    first_order: bool | None = None  # None: the signal as data on the hard pass (docs/meta.md)
    arity: int = 4
    scale: float = 1.0

    @property
    def label(self) -> str:
        """The inner cell and the outer objective, e.g. ``soft.relay.entry -> L^soft``: a display
        name and a test id (ASCII), never something code branches on."""
        return f"{'.'.join(self.signal)} -> L^{self.objective}"


# Online (one case per step) is the headline regime: there the objective has an optimum in η to
# find. With every case per step the objective is flat over decades and a found η says little
# (findings/2026-09-08-meta-learning-finds-a-step-size/note.md, the path). The outer step is 0.03:
# at 0.1 the climb from the non-functional start overshoots the optimum on one seed in ten.
ONLINE_META = MetaRecipe(
    Signal("soft", "relay", "entry"),
    steps=64,
    window=1,
    batch=16,
    hidden=(16, 8),
    outer_steps=500,
    lr=0.03,
)
BATCHED_META = MetaRecipe(
    Signal("soft", "relay", "entry"), steps=16, window=None, batch=16, hidden=(16, 8)
)


def _inner(m: MetaRecipe) -> dict:
    """The rollout's keyword arguments a recipe fixes."""
    return dict(
        signal=m.signal,
        steps=m.steps,
        window=m.window,
        control=m.control,
        first_order=m.first_order,
    )


def train(m: MetaRecipe, task: Task, seed: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Meta-learn the rule under the condition: η after every outer step, the objective it
    descended, and the deployed loss of the same final states (the deploy gap's other half). The
    seed's other half is kept for held-out draws (``adapt``)."""
    k_train, _ = jax.random.split(jax.random.key(seed))
    it = meta.learn(
        rule.init(m.eta0),
        k_train,
        task,
        batch=m.batch,
        hidden=m.hidden,
        arity=m.arity,
        scale=m.scale,
        lr=m.lr,
        **_inner(m),
    )
    cols = zip(*((rule.rate(p), J, d) for p, J, d in islice(it, m.outer_steps)), strict=True)
    etas, losses, deployed = (jnp.stack(c) for c in cols)
    return etas, losses, deployed


def landscape(m: MetaRecipe, eta: float, task: Task, seed: int) -> tuple[jax.Array, jax.Array]:
    """The objective at a fixed η, on the members the seed's first outer step draws: the fixed-η
    baseline measured on the objective the loop descends, and its deployed loss beside it."""
    k_train, _ = jax.random.split(jax.random.key(seed))
    _, k_mem, k_roll = jax.random.split(k_train, 3)  # as ``meta.learn`` splits it
    tiles, xs, ys = meta.members(
        k_mem, task, m.batch, hidden=m.hidden, arity=m.arity, scale=m.scale
    )
    J, (_, _, deployed) = meta.objective(rule.init(eta), tiles, xs, ys, k_roll, **_inner(m))
    return J, deployed


def adapt(
    m: MetaRecipe, eta: float, task: Task, seed: int, steps: int
) -> tuple[Tile, jax.Array, jax.Array]:
    """A fresh tile on a task drawn from the seed's held-out half, ``steps`` steps of the rule at
    η under the recipe's cell, window and control: the deployed check, and at any fixed η the
    plain-descent baseline. Swap the control to ``none`` to drive a control's η with the true
    signal."""
    _, k_held = jax.random.split(jax.random.key(seed))
    k_task, k_tile, k_run = jax.random.split(k_held, 3)
    x, y = task(k_task)
    t = init(k_tile, (x.shape[1], *m.hidden, y.shape[1]), m.arity, m.scale)
    return meta.rollout(eta, t, x, y, k_run, **{**_inner(m), "steps": steps})[2], x, y
