"""Claims of notes/2026-09-07-straight-through-is-a-signal-cell.md: descent on the bits."""

import jax.numpy as jnp
import pytest

from loom import recipes, tasks, tile
from loom.recipes import BITS_FLOOR, SOFT_FLOOR

CELLS = {"soft": SOFT_FLOOR, "bits": BITS_FLOOR}  # the matrix's signal axis, each under its recipe


def test_a_2_junta_under_the_bits_floor():
    for seed in range(3):
        t, x, y = recipes.run(BITS_FLOOR, tasks.junta(4, 2, k=2), seed)
        assert tile.accuracy(t, x, y, "hard") == 1.0


def test_no_deploy_gap_at_any_step_on_the_bits():
    _, rec, _, _ = recipes.trace(BITS_FLOOR, tasks.junta(4, 2, k=2), 0)
    assert jnp.array_equal(rec["train"], rec["hard"])


def test_the_gap_opens_then_closes_on_the_soft_pass():
    _, rec, _, _ = recipes.trace(SOFT_FLOOR, tasks.junta(4, 2, k=2), 0)
    gap = rec["train"] - rec["hard"]
    assert jnp.max(gap) > 0 and gap[-1] == 0


@pytest.mark.parametrize("cell", CELLS)
def test_every_matrix_cell_reaches_its_target(cell):
    t, x, y = recipes.run(CELLS[cell], tasks.junta(4, 2, k=2), 0)
    assert tile.accuracy(t, x, y, "hard") == 1.0
