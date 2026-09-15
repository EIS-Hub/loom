"""Mechanics of the descent loop, at the default rate and a handful of steps: the logits move and
the wiring does not; fit returns the n-th tile of descend; trajectory records where asked."""

import itertools

import jax
import jax.numpy as jnp

from loom import descent, rule, signals, tasks, tile

WIDTHS = (4, 16, 8, 2)


def test_one_step_moves_the_logits_and_not_the_wiring():
    k_task, k_tile = jax.random.split(jax.random.key(0))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    t0 = tile.init(k_tile, WIDTHS)
    t1 = next(descent.descend(t0, x, y))
    assert all(not jnp.array_equal(a, b) for a, b in zip(t0.logits, t1.logits, strict=True))
    assert all(jnp.array_equal(a, b) for a, b in zip(t0.wires, t1.wires, strict=True))


def test_fit_is_the_nth_tile_of_descend_and_trajectory_records_where_asked():
    k_task, k_tile = jax.random.split(jax.random.key(1))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    t0 = tile.init(k_tile, WIDTHS)
    third = list(itertools.islice(descent.descend(t0, x, y), 3))[-1]
    assert all(
        jnp.array_equal(a, b)
        for a, b in zip(third.logits, descent.fit(t0, x, y, steps=3).logits, strict=True)
    )
    _, rec = descent.trajectory(t0, x, y, steps=8, every=3)
    assert rec["step"].tolist() == [3, 6, 8] and rec["train"].shape == rec["hard"].shape == (3,)


def test_one_step_of_descent_is_the_rules_update_at_that_rate():
    k_task, k_tile = jax.random.split(jax.random.key(2))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    t0 = tile.init(k_tile, WIDTHS)
    by_descent = next(descent.descend(t0, x, y))  # at the default rate
    by_rule = rule.update(descent.RATE, t0, signals.compute(signals.REFERENCE, t0, x, y))
    assert all(jnp.allclose(a, b) for a, b in zip(by_descent.logits, by_rule.logits, strict=True))


def test_descent_under_a_named_rule_carries_its_state():
    k_task, k_tile = jax.random.split(jax.random.key(3))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    t0 = tile.init(k_tile, WIDTHS)
    a, b = (next(descent.descend(t0, x, y, rule=r)) for r in ("plain", "sign"))
    assert all(not jnp.array_equal(p, q) for p, q in zip(a.logits, b.logits, strict=True))
