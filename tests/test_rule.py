"""Mechanics of the rule: the rate is never negative, the default start moves nothing, one step is
z − η·s, the bound holds every logit and the wiring never moves."""

from typing import cast

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


def test_update_is_the_plain_rule_and_state_matches_the_ledger():
    t, x, y = setup(2)
    s = signals.compute(Signal("soft", "relay", "entry"), t, x, y)
    by_apply, state = rule.apply("plain", (), t, s, 2.0)
    assert all(
        jnp.array_equal(a, b)
        for a, b in zip(by_apply.logits, rule.update(2.0, t, s).logits, strict=True)
    )
    assert state == ()
    for name, n in rule.STATE.items():
        st = rule.state0(name, t)
        assert len(st) == n + (1 if name == "adam" else 0)  # Adam also counts its steps


def test_each_composition_matches_optax_for_a_few_steps():
    import optax

    t0, x, y = setup(3)
    pairs = {
        "plain": optax.sgd(2.0),
        "momentum": optax.sgd(2.0, momentum=0.9),
        "rmsprop": optax.rmsprop(2.0, decay=0.99, eps=1e-8),
        "lion": optax.lion(2.0, b1=0.9, b2=0.99, weight_decay=0.0),  # optax decays by default
        "adam": optax.adam(2.0, b1=0.9, b2=0.999, eps=1e-8),
    }
    for name, opt in pairs.items():
        ours, state = t0, rule.state0(name, t0)
        theirs, opt_state = t0.logits, opt.init(t0.logits)
        for _ in range(4):
            s = signals.compute(signals.REFERENCE, ours, x, y)
            ours, state = rule.apply(name, state, ours, s, 2.0)
            g = signals.compute(signals.REFERENCE, tile.Tile(theirs, t0.wires), x, y)
            updates, opt_state = opt.update(g, opt_state, theirs)
            theirs = cast(tuple[jax.Array, ...], optax.apply_updates(theirs, updates))
        for a, b in zip(ours.logits, theirs, strict=True):
            assert jnp.allclose(a, b, rtol=1e-4, atol=1e-4), name  # float32 at logits of ~20


def test_the_sign_rule_moves_every_entry_with_a_vote_by_the_rate():
    t, x, y = setup(4)
    s = signals.compute(Signal("soft", "relay", "entry"), t, x, y)
    moved, _ = rule.apply("sign", (), t, s, 0.5)
    for z0, z1, g in zip(t.logits, moved.logits, s, strict=True):
        assert jnp.allclose(jnp.abs(z1 - z0), 0.5 * (g != 0))
