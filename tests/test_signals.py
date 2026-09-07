"""Mechanics of signals: straight-through is the bits in value, residuals are bits, the address
distribution is the read, the relay is autodiff, the surrogate changes no sign, shapes hold."""

import jax
import jax.numpy as jnp
import pytest

from loom import signals, tasks, tile
from loom.signals import REFERENCE, Signal

SHAPES = {"flat": ((4, 16, 8, 2), 4), "deep": ((6, 24, 12, 4), 3)}  # arity 3 on 6 inputs: deep


def setup(seed, shape="flat"):
    k_task, k_tile = jax.random.split(jax.random.key(seed))
    widths, arity = SHAPES[shape]
    x, y = tasks.inputs(widths[0]), tasks.k_junta(k_task, widths[0], widths[-1], k=2)
    return tile.init(k_tile, widths, arity), x, y


def test_straight_through_tables_are_the_bits_in_value():
    t, x, _ = setup(0)
    ste = tile.run(tuple(signals.straight_through(lg) for lg in t.logits), t.wires, x)[-1]
    assert jnp.array_equal(ste, tile.forward(t, x, "hard"))


def test_the_residual_is_the_error_in_bits_and_a_signal_has_the_logits_shape():
    t, x, y = setup(1)
    assert jnp.all(jnp.isin(signals.residual(t, x, y, "hard"), jnp.array([-1.0, 0.0, 1.0])))
    for sig in (REFERENCE, Signal("hard")):
        g = signals.compute(sig, t, x, y)
        assert [a.shape for a in g] == [a.shape for a in t.logits]
        assert all(jnp.all(jnp.isfinite(a)) for a in g)


def test_the_address_distribution_against_the_table_is_the_read():
    t, _, _ = setup(2, "deep")
    inputs = jax.random.uniform(jax.random.key(3), (5, *t.wires[0].shape))  # [B, arity, gates]
    luts = tile.tables(t.logits[0], "soft")
    p = signals.address(inputs)
    assert jnp.allclose(p.sum(-1), 1.0)
    assert jnp.allclose(jnp.einsum("bga,ga->bg", p, luts), tile.read(luts, inputs), atol=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("on", ["soft", "hard"])
def test_the_relay_reproduces_autodiff_on_both_passes(shape, on):
    t, x, y = setup(4, shape)
    by_relay = signals.compute(Signal(on, "relay"), t, x, y)
    by_autodiff = signals.compute(Signal(on, "autodiff"), t, x, y)
    for a, b in zip(by_relay, by_autodiff, strict=True):
        assert jnp.allclose(a, b, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize("on", ["soft", "hard"])
def test_dropping_the_surrogate_changes_no_sign(on):
    t, x, y = setup(5, "deep")
    kept = signals.compute(Signal(on, "relay", surrogate=True), t, x, y)
    dropped = signals.compute(Signal(on, "relay", surrogate=False), t, x, y)
    for a, b in zip(kept, dropped, strict=True):
        assert jnp.array_equal(jnp.sign(a), jnp.sign(b))


def test_the_uniform_transport_is_the_relay_at_the_output_layer_only():
    t, x, y = setup(6, "deep")
    by_relay = signals.compute(Signal("soft", "relay"), t, x, y)
    blind = signals.compute(Signal("soft", "uniform"), t, x, y)
    assert jnp.allclose(by_relay[-1], blind[-1])  # no transport yet at the output layer
    assert not jnp.allclose(by_relay[0], blind[0])


def test_autodiff_cannot_drop_the_surrogate():
    t, x, y = setup(7)
    with pytest.raises(NotImplementedError):
        signals.compute(Signal(surrogate=False), t, x, y)
