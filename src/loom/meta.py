"""Meta-learning: the rule's parameters learned from rollouts of the rule itself.

The inner loop runs the rule for K steps on a tile, from some state, with the signal a chip
would have. The outer loop takes the soft loss of the tile the rule ends on, differentiates it
with respect to the rule's parameters through those K steps (back-propagation through the
update, truncated at K), and takes an Adam step. Nothing here runs on a chip: the outer loop is
the training of the genome, offline, on the host. What a rollout starts from is the other half
of the method: fresh tiles on fresh tasks here; a pool of tile states of every age next.

Re-derived 2026-09 from blastema/training/rollout.py (windowed_rollout) and meta.py
(make_meta_step), without the providers.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import optax

from loom.rule import Params, rate, update
from loom.signals import Signal, compute, loss
from loom.tile import Tile, init

Control = Literal["none", "flipped", "shuffled", "output_only"]


def controlled(kind: Control, s: tuple[jax.Array, ...], key: jax.Array) -> tuple[jax.Array, ...]:
    """The signal under one of the check's controls. ``flipped``: −s, the direction control (η
    cannot go negative, so the outer loop must fail to recover a rule). ``shuffled``: the same
    numbers, permuted within each layer, so an entry receives a magnitude that has nothing to do
    with it, the information control. ``output_only``: the hidden layers silenced, only the output
    gates, which read their own residual, move: the baseline any learning must beat."""
    if kind == "none":
        return s
    if kind == "flipped":
        return tuple(-g for g in s)
    if kind == "shuffled":
        keys = jax.random.split(key, len(s))
        return tuple(
            jax.random.permutation(k, g.ravel()).reshape(g.shape)
            for k, g in zip(keys, s, strict=True)
        )
    if kind == "output_only":
        return (*(jnp.zeros_like(g) for g in s[:-1]), s[-1])
    raise ValueError(f"control must be none, flipped, shuffled or output_only, got {kind!r}")


def rollout(
    eta: jax.Array | float,
    tile: Tile,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    *,
    signal: Signal,
    steps: int,
    window: int | None = None,
    control: Control = "none",
    first_order: bool = False,
    clip: float | None = None,
) -> tuple[jax.Array, jax.Array, Tile]:
    """K steps of the rule from a state, in one scan.

    Returns the objective, the soft loss on every case of the tile the rule ends on; the record,
    the soft loss on each step's window before that step's update (what a deployed tile
    experiences, never the objective: a mean over steps would credit early updates through every
    later loss and the last through none); and the tile. ``window`` is ``descent``'s: all cases per
    step, or that many drawn with replacement. ``first_order`` treats the signal as data: on the
    bits every derivative through the signal is already zero, so there it changes nothing.
    """

    def step(logits, k):
        k_win, k_ctl = jax.random.split(k)
        idx = slice(None) if window is None else jax.random.choice(k_win, len(x), (window,))
        xw, yw, t = x[idx], y[idx], Tile(logits, tile.wires)
        before = loss(t, xw, yw, "soft")
        s = controlled(control, compute(signal, t, xw, yw), k_ctl)
        if first_order:
            s = jax.lax.stop_gradient(s)
        return update(eta, t, s, clip).logits, before

    logits, before = jax.lax.scan(step, tile.logits, jax.random.split(key, steps))
    t = Tile(logits, tile.wires)
    return loss(t, x, y, "soft"), before, t


def members(key: jax.Array, task, n: int, *, hidden: tuple[int, ...], arity: int = 4, scale=1.0):
    """n fresh tiles on n fresh tasks, stacked: every state carries the task it lives on."""
    k_task, k_tile = jax.random.split(key)
    xs, ys = jax.vmap(task)(jax.random.split(k_task, n))
    widths = (xs.shape[-1], *hidden, ys.shape[-1])
    tiles = jax.vmap(lambda k: init(k, widths, arity, scale))(jax.random.split(k_tile, n))
    return tiles, xs, ys


def objective(params: Params, tiles: Tile, xs, ys, key, **kw):
    """The objective over a batch of states: the mean loss the rule ends on, and the rollouts."""
    keys = jax.random.split(key, xs.shape[0])
    run = jax.vmap(lambda t, x, y, k: rollout(rate(params), t, x, y, k, **kw))
    J, before, rolled = run(tiles, xs, ys, keys)
    return jnp.mean(J), (before, rolled)


def step(params: Params, state, tiles: Tile, xs, ys, key, *, opt, **kw):
    """One outer step: the meta-gradient through the K unrolled steps, then the optimiser on the
    host. The rolled tiles come back as values: credit never runs past the K steps, whatever state
    they started from, so an old state is an input to the outer step, never a longer horizon."""
    (J, (before, rolled)), grads = jax.value_and_grad(objective, has_aux=True)(
        params, tiles, xs, ys, key, **kw
    )
    updates, state = opt.update(grads, state, params)
    return optax.apply_updates(params, updates), state, J, rolled


def learn(
    params: Params, key: jax.Array, task, *, batch: int, hidden, arity=4, scale=1.0, lr=0.1, **kw
):
    """The outer loop, without end: fresh states every step; yields (params, J) after each.

    Adam on the host, with the gradient's global norm clipped: with η = exp(raw) the gradient on
    ``raw`` vanishes with η, so a non-functional start needs a step whose size does not, which
    Adam's normalisation gives; the outer optimiser is not on the chip and not in the cost table.
    """
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = opt.init(params)
    outer = jax.jit(partial(step, opt=opt, **kw))
    while True:
        key, k_mem, k_roll = jax.random.split(key, 3)
        tiles, xs, ys = members(k_mem, task, batch, hidden=hidden, arity=arity, scale=scale)
        params, state, J, _ = outer(params, state, tiles, xs, ys, k_roll)
        yield params, J
