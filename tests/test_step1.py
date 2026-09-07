"""Step 1, signals I: a signal is coordinates; descent on the bits; the deploy gap in training."""

import jax
import jax.numpy as jnp

from loom import descent, signals, tasks, tile
from loom.signals import Signal

WIDTHS = (4, 16, 8, 2)


def test_straight_through_tables_are_the_bits_in_value():
    t = tile.init(jax.random.key(0), WIDTHS)
    x = tasks.inputs(4)
    ste = tile.run(tuple(signals.straight_through(lg) for lg in t.logits), t.wires, x)[-1]
    assert jnp.array_equal(ste, tile.forward(t, x, "hard"))


def test_descent_on_the_bits_reaches_the_target():
    x = tasks.inputs(4)
    for seed in range(3):
        k_task, k_tile = jax.random.split(jax.random.fold_in(jax.random.key(1), seed))
        y = tasks.k_junta(k_task, 4, 2, k=2)
        t = descent.fit(tile.init(k_tile, WIDTHS), x, y, steps=2000, lr=0.02, signal=Signal("hard"))
        assert tile.accuracy(t, x, y, "hard") == 1.0  # no gap to close: it trained on bits


def test_the_deploy_gap_is_zero_on_the_bits_and_closes_on_the_soft_pass():
    k_task, k_tile = jax.random.split(jax.random.key(2))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    t0 = tile.init(k_tile, WIDTHS)
    _, soft = descent.trajectory(t0, x, y, steps=500, every=10)
    _, bits = descent.trajectory(t0, x, y, steps=2000, every=10, lr=0.02, signal=Signal("hard"))
    # On the bits the training view is the deployed circuit: no gap at any step...
    assert jnp.array_equal(bits["train"], bits["hard"])
    # ...while soft training runs ahead of its deployed accuracy until the tables saturate.
    gap = soft["train"] - soft["hard"]
    assert jnp.max(gap) > 0 and gap[-1] == 0


def test_the_residual_is_the_error_in_bits_and_the_signal_has_the_logits_shape():
    k_task, k_tile = jax.random.split(jax.random.key(3))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    t = tile.init(k_tile, WIDTHS)
    assert jnp.all(jnp.isin(signals.residual(t, x, y, "hard"), jnp.array([-1.0, 0.0, 1.0])))
    g = signals.compute(Signal(), t, x, y)
    assert [a.shape for a in g] == [a.shape for a in t.logits]
