"""Claims of notes/2026-09-03-one-tile-computes.md: one tile reaches its target by descent."""

from loom import recipes, tasks, tile
from loom.recipes import ONLINE_FLOOR, SOFT_FLOOR


def test_a_2_junta_under_the_soft_floor_with_no_deploy_gap():
    for seed in range(3):
        t, x, y = recipes.run(SOFT_FLOOR, tasks.junta(4, 2, k=2), seed)
        assert tile.accuracy(t, x, y, "hard") == 1.0
        assert tile.accuracy(t, x, y, "soft") == 1.0


def test_two_bit_addition_with_carry_under_the_soft_floor():
    t, x, y = recipes.run(SOFT_FLOOR, tasks.addition(4), 0)
    assert tile.accuracy(t, x, y, "hard") == 1.0


def test_a_2_junta_online_one_case_per_step():
    for seed in range(3):
        t, x, y = recipes.run(ONLINE_FLOOR, tasks.junta(4, 2, k=2), seed)
        assert tile.accuracy(t, x, y, "hard") == 1.0
