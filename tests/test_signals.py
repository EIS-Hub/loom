"""Mechanics of signals: straight-through is the bits in value, residuals are bits, the address
distribution is the read, the relay is autodiff, the partial to the entry changes no sign, feedback
alignment on the wiring's path counts is the uniform split, labels are names only, shapes hold."""

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
    p = signals.address(inputs)  # [B, gates, 2**arity], a distribution over entries per case
    assert jnp.allclose(p.sum(-1), 1.0)
    expectation = jnp.sum(p * luts[None], axis=-1)  # Σ_a P(a | u) T[a], per case and gate
    assert jnp.allclose(expectation, tile.read(luts, inputs), atol=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("on", ["soft", "hard"])
def test_the_relay_reproduces_autodiff_on_both_passes(shape, on):
    t, x, y = setup(4, shape)
    by_relay = signals.compute(Signal(on, "relay"), t, x, y)
    by_autodiff = signals.compute(Signal(on, "autodiff"), t, x, y)
    for a, b in zip(by_relay, by_autodiff, strict=True):
        assert jnp.allclose(a, b, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize("on", ["soft", "hard"])
def test_the_partial_to_the_entry_changes_no_sign(on):
    t, x, y = setup(5, "deep")
    to_logit = signals.compute(Signal(on, "relay", "logit"), t, x, y)
    to_entry = signals.compute(Signal(on, "relay", "entry"), t, x, y)
    for a, b in zip(to_logit, to_entry, strict=True):
        assert jnp.array_equal(jnp.sign(a), jnp.sign(b))


@pytest.mark.parametrize("via", ["uniform", "direct", "reachable"])
def test_the_blind_transports_are_the_relay_at_the_output_layer_only(via):
    t, x, y = setup(6, "deep")
    by_relay = signals.compute(Signal("soft", "relay"), t, x, y)
    blind = signals.compute(Signal("soft", via), t, x, y)
    assert jnp.allclose(by_relay[-1], blind[-1])  # an output gate reads its own residual
    assert not jnp.allclose(by_relay[0], blind[0])


def test_direct_feedback_through_the_path_counts_is_the_uniform_split():
    t, x, y = setup(7, "deep")
    acts = tile.activations(t, x, "soft")
    n_out = y.shape[1]
    # the number of wiring paths from every gate to every output: the layered adjoint of the
    # all-sums network, seeded with one output at a time
    per_output = [a[:n_out] for a in acts]  # one "case" per output; the carry ignores the values
    counts = [c.T for c in signals.backward(t, per_output, "soft", jnp.eye(n_out), signals.ones)]
    via_bus = signals.last_hop(
        signals.broadcast(signals.seed(acts, y), counts), t, acts, "soft", "entry"
    )
    via_layers = signals.compute(Signal("soft", "uniform", "entry"), t, x, y)
    for a, b in zip(via_bus, via_layers, strict=True):
        assert jnp.allclose(a, b, atol=1e-6)


def test_reachability_counts_wiring_paths_and_masks_the_bus():
    t, x, y = setup(9, "deep")
    acts = tile.activations(t, x, "soft")
    counts = signals.path_counts(t, acts, y.shape[1])
    assert all(jnp.all(c >= 0) and jnp.all(c == jnp.round(c)) for c in counts)  # whole paths
    assert jnp.all(counts[-1] == jnp.eye(y.shape[1]))  # an output gate reaches itself only
    assert any(jnp.any(c == 0) for c in counts[:-1])  # some gate cannot reach some output
    masked = signals.compute(Signal("hard", "reachable", "entry"), t, x, y)
    assert all(jnp.all(jnp.isfinite(a)) for a in masked)


def test_the_errors_are_the_per_gate_signal_and_to_gate_broadcasts_them():
    t, x, y = setup(10, "deep")
    sig = Signal("hard", "uniform", "gate")
    lams = signals.gate_errors(sig, t, x, y)
    assert [a.shape for a in lams] == [(x.shape[0], lg.shape[0]) for lg in t.logits]
    coarse = signals.compute(sig, t, x, y)
    for lam, c in zip(lams, coarse, strict=True):
        assert jnp.allclose(c, jnp.sum(lam, 0)[:, None])  # every entry of a gate gets its error
    with pytest.raises(ValueError):
        signals.gate_errors(Signal("hard", "flip"), t, x, y)


def test_every_via_inside_the_frame_has_an_adjoint():
    inside = {s.via for s in signals.CELLS} - {"autodiff", "flip"}  # the check, and the exception
    assert set(signals.ADJOINTS) == inside


def test_labels_are_ascii_names_and_no_code_reads_them():
    import pathlib

    assert REFERENCE.label == "soft.autodiff.logit" and REFERENCE.label.isascii()
    src = (pathlib.Path(signals.__file__).parent).glob("*.py")
    uses = [line for f in src for line in f.read_text().splitlines() if ".label" in line]
    assert not uses, f"a label is a name, never a branch: {uses}"


def test_autodiff_cannot_take_the_partial_to_the_entry():
    t, x, y = setup(8)
    with pytest.raises(NotImplementedError):
        signals.compute(Signal(to="entry"), t, x, y)
