"""Step 1, signals I: the read mode is an axis; descent is driven by a signal."""

import jax
import jax.numpy as jnp

from loom import descent, signals, tasks, tile

WIDTHS = (4, 16, 8, 2)


def test_the_straight_through_value_is_the_deployed_circuit():
    t = tile.init(jax.random.key(0), WIDTHS)
    x = tasks.inputs(4)
    assert jnp.array_equal(tile.forward(t, x, "ste"), tile.forward(t, x, "hard"))


def test_training_the_deployed_circuit_directly_reaches_the_target():
    x = tasks.inputs(4)
    for seed in range(3):
        k_task, k_tile = jax.random.split(jax.random.fold_in(jax.random.key(1), seed))
        y = tasks.k_junta(k_task, 4, 2, k=2)
        t = descent.fit(tile.init(k_tile, WIDTHS), x, y, lr=0.02, mode="ste")
        assert tile.accuracy(t, x, y, "hard") == 1.0  # no gap to close: it trained on bits


def test_the_residual_is_the_error_in_bits_and_the_gradient_has_the_logits_shape():
    k_task, k_tile = jax.random.split(jax.random.key(2))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    t = tile.init(k_tile, WIDTHS)
    r = signals.residual(t, x, y, "hard")
    assert jnp.all(jnp.isin(r, jnp.array([-1.0, 0.0, 1.0])))
    g = signals.gradient(t, x, y, "soft")
    assert [a.shape for a in g] == [a.shape for a in t.logits]
