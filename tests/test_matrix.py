"""The combinatorial test: every substrate × optimiser × signal × read mode the code claims to
support runs, and its step check holds. Grows one cell at a time; a cell that cannot pass is a
finding."""

import jax
import pytest

from loom import descent, signals, tasks, tile

SUBSTRATES = {"lut": lambda key: tile.init(key, (4, 16, 8, 2))}
OPTIMISERS = {"descent": descent.fit}
SIGNALS = {"gradient": signals.gradient}  # the next chunk adds the relay and the uniform adjoint
MODES = {"soft": 0.1, "ste": 0.02}  # read mode → step size; bits chatter at the soft rate


@pytest.mark.parametrize("substrate", SUBSTRATES)
@pytest.mark.parametrize("optimiser", OPTIMISERS)
@pytest.mark.parametrize("signal", SIGNALS)
@pytest.mark.parametrize("mode", MODES)
def test_cell_holds(substrate, optimiser, signal, mode):
    k_task, k_sub = jax.random.split(jax.random.key(0))
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    t = OPTIMISERS[optimiser](
        SUBSTRATES[substrate](k_sub), x, y, lr=MODES[mode], signal=SIGNALS[signal], mode=mode
    )
    assert tile.accuracy(t, x, y, "hard") == 1.0
