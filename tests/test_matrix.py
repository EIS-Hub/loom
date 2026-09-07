"""The combinatorial test, mechanics half: every substrate × signal cell runs and yields a
well-shaped, finite signal. Whether a cell reaches its target is a claim (claims/), not a test."""

import jax
import jax.numpy as jnp
import pytest

from loom import signals, tasks, tile
from loom.signals import REFERENCE, Signal

SUBSTRATES = {"lut": lambda key: tile.init(key, (4, 16, 8, 2))}
SIGNALS = {s.label: s for s in (REFERENCE, Signal("hard"))}  # named by their coordinates


@pytest.mark.parametrize("substrate", SUBSTRATES)
@pytest.mark.parametrize("signal", SIGNALS)
def test_cell_runs(substrate, signal):
    k_task, k_sub = jax.random.split(jax.random.key(0))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    t = SUBSTRATES[substrate](k_sub)
    g = signals.compute(SIGNALS[signal], t, x, y)
    assert [a.shape for a in g] == [a.shape for a in t.logits]
    assert all(jnp.all(jnp.isfinite(a)) for a in g)
