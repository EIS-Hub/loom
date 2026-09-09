"""Mechanics of the rule: the rate is never negative, the default start moves nothing, one step is
z − η·s, the bound holds every logit and the wiring never moves."""

import jax
import jax.numpy as jnp

from loom import rule, signals, tasks, tile
from loom.signals import Signal

WIDTHS = (4, 16, 8, 2)


def setup(seed):
    k_task, k_tile = jax.random.split(jax.random.key(seed))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    return tile.init(k_tile, WIDTHS), x, y


def test_the_rate_is_never_negative():
    assert jnp.isclose(rule.rate(rule.init(1e-2)), 1e-2)
    assert rule.rate(rule.Params(jnp.asarray(-80.0))) >= 0.0


def test_one_step_is_z_minus_eta_s_and_at_eta_zero_nothing_moves():
    t, x, y = setup(0)
    s = signals.compute(Signal("soft", "relay", "entry"), t, x, y)
    still = rule.update(0.0, t, s)
    assert all(jnp.array_equal(a, b) for a, b in zip(still.logits, t.logits, strict=True))
    moved = rule.update(2.0, t, s)
    for z0, z1, g in zip(t.logits, moved.logits, s, strict=True):
        assert jnp.allclose(z1, z0 - 2.0 * g)
    assert all(jnp.array_equal(a, b) for a, b in zip(moved.wires, t.wires, strict=True))


def test_the_bound_holds_every_logit():
    t, x, y = setup(1)
    s = signals.compute(Signal("soft", "relay", "entry"), t, x, y)
    bounded = rule.update(1e6, t, s, clip=2.0)
    assert all(jnp.all(jnp.abs(z) <= 2.0) for z in bounded.logits)
    assert any(jnp.any(jnp.abs(z) == 2.0) for z in bounded.logits)  # some entry hit the bound
