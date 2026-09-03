"""Step 0, one tile computes: descent reaches the target on the hard read; soft ≡ hard at deploy."""

import jax
import jax.numpy as jnp

from loom import descent, tasks, tile

WIDTHS = (4, 16, 8, 2)  # the smallest tile with a hidden layer: 26 gates of arity 4


def test_read_is_exact_on_bits():
    # One XOR gate, table [0, 1, 1, 0], first input = least significant address bit.
    xor = jnp.array([[0.0, 1.0, 1.0, 0.0]])
    a, b = tasks.inputs(2).T
    out = tile.read(xor, jnp.stack([a, b], axis=1)[:, :, None])  # [B, arity, gates=1]
    assert jnp.array_equal(out[:, 0], jnp.logical_xor(a, b).astype(jnp.float32))
    # And the hard read of any tile emits bits.
    hard = tile.forward(tile.init(jax.random.key(0), WIDTHS), tasks.inputs(4), hard=True)
    assert jnp.all(jnp.isin(hard, jnp.array([0.0, 1.0])))


def test_descent_reaches_a_k_junta_on_the_hard_read():
    key = jax.random.key(1)
    x = tasks.inputs(4)
    for seed in range(3):
        k_task, k_tile = jax.random.split(jax.random.fold_in(key, seed))
        y = tasks.k_junta(k_task, 4, 2, k=2)
        t = descent.fit(tile.init(k_tile, WIDTHS), x, y)
        assert tile.accuracy(t, x, y, hard=True) == 1.0
        assert tile.accuracy(t, x, y, hard=False) == 1.0  # soft ≡ hard at deploy


def test_descent_reaches_two_bit_addition_with_carry():
    x, y = tasks.add(4)
    t = descent.fit(tile.init(jax.random.key(2), (4, 16, 8, 3)), x, y)
    assert tile.accuracy(t, x, y, hard=True) == 1.0
