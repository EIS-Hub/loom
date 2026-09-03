"""The combinatorial test: every substrate × optimiser × signal the code claims to support runs,
and its step check holds. Grows one cell at a time; a cell that cannot pass is a finding."""

import jax
import pytest

from loom import descent, tasks, tile

SUBSTRATES = {"lut": lambda key: tile.init(key, (4, 16, 8, 2))}
OPTIMISERS = {"descent": descent.fit}
SIGNALS = ["none"]  # step 1 adds what a local update may read


@pytest.mark.parametrize("substrate", SUBSTRATES)
@pytest.mark.parametrize("optimiser", OPTIMISERS)
@pytest.mark.parametrize("signal", SIGNALS)
def test_cell_holds(substrate, optimiser, signal):
    k_task, k_sub = jax.random.split(jax.random.key(0))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    t = OPTIMISERS[optimiser](SUBSTRATES[substrate](k_sub), x, y)
    assert tile.accuracy(t, x, y, hard=True) == 1.0
