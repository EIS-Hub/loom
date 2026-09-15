"""Claims of notes/2026-09-08-meta-learning-finds-a-step-size.md: from a non-functional start the
outer loop finds a step size under the soft relay and not under its controls, and the η it finds
adapts a fresh tile to a task the loop never saw."""

from loom import recipes, tasks, tile
from loom.recipes import SOFT_META

TASK = tasks.junta(4, 2, k=2)
SEEDS = range(3)


def test_the_outer_loop_finds_a_step_size_that_adapts_a_held_out_task():
    for seed in SEEDS:
        etas, losses = recipes.train(SOFT_META, TASK, seed)
        assert losses[-1] < losses[0] / 3  # the objective falls well below the untrained loss
        assert etas[-1] > 100 * SOFT_META.eta0  # by orders of magnitude, not a nudge
        t, x, y = recipes.adapt(SOFT_META, float(etas[-1]), TASK, seed, steps=500)
        assert tile.accuracy(t, x, y, "hard") == 1.0


def test_under_the_sign_flipped_signal_eta_falls_and_nothing_trains():
    for seed in SEEDS:
        etas, losses = recipes.train(SOFT_META._replace(control="flipped"), TASK, seed)
        assert etas[-1] < SOFT_META.eta0  # η can only shrink: the direction control
        assert losses[-1] > 0.9 * losses[0]  # untrained


def test_under_the_shuffled_and_the_output_only_signals_the_loss_stays_near_untrained():
    for seed in SEEDS:
        trained = recipes.train(SOFT_META, TASK, seed)[1][-1]
        for control in ("shuffled", "output_only"):
            _, losses = recipes.train(SOFT_META._replace(control=control), TASK, seed)
            assert losses[-1] > (losses[0] + trained) / 2  # closer to untrained than to trained
