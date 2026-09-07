"""Mechanics of the tile: the read is exact on bits, the hard read emits bits, a bad read raises."""

import jax
import jax.numpy as jnp
import pytest

from loom import tasks, tile

WIDTHS = (4, 16, 8, 2)


def test_read_is_exact_on_bits():
    # One XOR gate, table [0, 1, 1, 0], first input = least significant address bit.
    xor = jnp.array([[0.0, 1.0, 1.0, 0.0]])
    a, b = tasks.inputs(2).T
    out = tile.read(xor, jnp.stack([a, b], axis=1)[:, :, None])  # [B, arity, gates=1]
    assert jnp.array_equal(out[:, 0], jnp.logical_xor(a, b).astype(jnp.float32))


def test_the_hard_read_emits_bits_and_the_wiring_is_in_range():
    t = tile.init(jax.random.key(0), WIDTHS)
    hard = tile.forward(t, tasks.inputs(4), "hard")
    assert jnp.all(jnp.isin(hard, jnp.array([0.0, 1.0])))
    for w, n_prev in zip(t.wires, WIDTHS[:-1], strict=True):
        assert w.shape[0] == 4 and jnp.all((w >= 0) & (w < n_prev))


def test_a_bad_read_mode_raises():
    with pytest.raises(ValueError, match="read mode"):
        tile.tables(jnp.zeros((1, 16)), "ste")  # type: ignore[arg-type]
