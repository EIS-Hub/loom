"""The combinatorial test: every substrate × optimiser × signal the code claims to support runs,
and its step check holds. Grows one cell at a time; a cell that cannot pass is a finding."""

import jax
import pytest

from loom import descent, tasks, tile
from loom.signals import Signal

SUBSTRATES = {"lut": lambda key: tile.init(key, (4, 16, 8, 2))}
OPTIMISERS = {"descent": descent.fit}
SIGNALS = {  # signal → (step size, steps): bits chatter at the soft rate
    "soft": (Signal("soft"), 0.1, 500),
    "hard": (Signal("hard"), 0.02, 2000),
}


@pytest.mark.parametrize("substrate", SUBSTRATES)
@pytest.mark.parametrize("optimiser", OPTIMISERS)
@pytest.mark.parametrize("signal", SIGNALS)
def test_cell_holds(substrate, optimiser, signal):
    k_task, k_sub = jax.random.split(jax.random.key(0))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    sig, lr, steps = SIGNALS[signal]
    t = OPTIMISERS[optimiser](SUBSTRATES[substrate](k_sub), x, y, steps=steps, lr=lr, signal=sig)
    assert tile.accuracy(t, x, y, "hard") == 1.0
