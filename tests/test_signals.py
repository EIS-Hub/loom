"""Mechanics of signals: straight-through is the bits in value, residuals are bits, shapes hold."""

import jax
import jax.numpy as jnp
import pytest

from loom import signals, tasks, tile
from loom.signals import REFERENCE, Signal

WIDTHS = (4, 16, 8, 2)


def test_straight_through_tables_are_the_bits_in_value():
    t = tile.init(jax.random.key(0), WIDTHS)
    x = tasks.inputs(4)
    ste = tile.run(tuple(signals.straight_through(lg) for lg in t.logits), t.wires, x)[-1]
    assert jnp.array_equal(ste, tile.forward(t, x, "hard"))


def test_the_residual_is_the_error_in_bits_and_a_signal_has_the_logits_shape():
    k_task, k_tile = jax.random.split(jax.random.key(1))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    t = tile.init(k_tile, WIDTHS)
    assert jnp.all(jnp.isin(signals.residual(t, x, y, "hard"), jnp.array([-1.0, 0.0, 1.0])))
    for sig in (REFERENCE, Signal("hard")):
        g = signals.compute(sig, t, x, y)
        assert [a.shape for a in g] == [a.shape for a in t.logits]
        assert all(jnp.all(jnp.isfinite(a)) for a in g)


def test_unsupported_coordinates_raise_until_their_chunk():
    t = tile.init(jax.random.key(2), WIDTHS)
    x, y = tasks.inputs(4), jnp.zeros((16, 2))
    with pytest.raises(NotImplementedError):
        signals.compute(Signal(surrogate=False), t, x, y)
