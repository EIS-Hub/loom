"""Mechanics of meta-learning, at a handful of steps on the flat tile: the meta-gradient matches a
finite difference; first order is a no-op on the bits and not on the soft pass; at η = 0 the
objective is the untrained loss; the controls do what they say; members are distinct; the outer
loop yields after every step and moves η."""

import itertools

import jax
import jax.numpy as jnp
import pytest

from loom import meta, rule, signals, tasks, tile
from loom.signals import Signal

WIDTHS = (4, 16, 8, 2)
SOFT = Signal("soft", "relay", "entry")
BITS = Signal("hard", "uniform", "entry")


def setup(seed):
    k_task, k_tile, k_run = jax.random.split(jax.random.key(seed), 3)
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    return tile.init(k_tile, WIDTHS), x, y, k_run


def objective(eta, t, x, y, k, **kw):
    return meta.rollout(eta, t, x, y, k, steps=3, **kw)[0]


def test_the_meta_gradient_matches_a_finite_difference():
    t, x, y, k = setup(0)
    eta, h = 30.0, 0.5  # float32: the step must move the loss above its resolution
    by_autodiff = jax.grad(objective)(eta, t, x, y, k, signal=SOFT)
    by_difference = (
        objective(eta + h, t, x, y, k, signal=SOFT) - objective(eta - h, t, x, y, k, signal=SOFT)
    ) / (2 * h)
    assert jnp.isclose(by_autodiff, by_difference, rtol=1e-2)


def test_first_order_is_a_no_op_on_the_bits_and_not_on_the_soft_pass():
    t, x, y, k = setup(1)
    for sig, same in ((BITS, True), (SOFT, False)):
        full = jax.grad(objective)(30.0, t, x, y, k, signal=sig)
        first = jax.grad(objective)(30.0, t, x, y, k, signal=sig, first_order=True)
        assert bool(jnp.isclose(full, first)) == same


def test_at_eta_zero_the_objective_is_the_untrained_loss_and_the_tile_does_not_move():
    t, x, y, k = setup(2)
    J, before, ended = meta.rollout(0.0, t, x, y, k, signal=SOFT, steps=3)
    assert jnp.isclose(J, signals.loss(t, x, y, "soft"))
    assert jnp.allclose(before, J) and before.shape == (3,)
    assert all(jnp.array_equal(a, b) for a, b in zip(ended.logits, t.logits, strict=True))


def test_the_controls_do_what_they_say():
    t, x, y, k = setup(3)
    s = signals.compute(SOFT, t, x, y)
    flipped = meta.controlled("flipped", s, k)
    assert all(jnp.array_equal(f, -g) for f, g in zip(flipped, s, strict=True))
    only = meta.controlled("output_only", s, k)
    assert all(jnp.all(o == 0) for o in only[:-1]) and jnp.array_equal(only[-1], s[-1])
    shuffled = meta.controlled("shuffled", s, k)
    for a, b in zip(shuffled, s, strict=True):
        assert jnp.array_equal(jnp.sort(a.ravel()), jnp.sort(b.ravel()))  # the same numbers
        assert not jnp.array_equal(a, b)  # elsewhere
    with pytest.raises(ValueError, match="control"):
        meta.controlled("random", s, k)  # type: ignore[arg-type]


def test_members_are_distinct_tiles_on_distinct_tasks():
    tiles, xs, ys = meta.members(jax.random.key(4), tasks.junta(4, 2, k=2), 3, hidden=(16, 8))
    assert [a.shape for a in tiles.logits] == [(3, 16, 16), (3, 8, 16), (3, 2, 16)]
    assert xs.shape == (3, 16, 4) and ys.shape == (3, 16, 2)
    assert not jnp.array_equal(tiles.logits[0][0], tiles.logits[0][1])
    assert not jnp.array_equal(ys[0], ys[1])


def test_learn_yields_after_every_step_and_moves_eta():
    params = rule.init()
    it = meta.learn(
        params,
        jax.random.key(5),
        tasks.junta(4, 2, k=2),
        batch=2,
        hidden=(8,),
        signal=SOFT,
        steps=2,
    )
    (p1, J1), (p2, J2) = itertools.islice(it, 2)
    assert jnp.isfinite(J1) and jnp.isfinite(J2)
    assert not jnp.array_equal(p1.raw, params.raw) and not jnp.array_equal(p2.raw, p1.raw)
