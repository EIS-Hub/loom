"""The claim of note.md: online, where the objective has an optimum in η, the outer loop settles
on it from a non-functional start under the true signal (the median η over the last 200 outer
steps lies on the floor of the swept landscape, a band less than a factor 5 wide; the endpoint
alone carries transients) and that η adapts a fresh tile to a task the loop never saw; under the
sign flip η falls and nothing trains; under the shuffled and the output-only signals the loss
stays closer to untrained than to trained. One test per seed and control, so the runner spreads
them; the true-signal run is computed once per seed."""

from functools import cache

import jax.numpy as jnp
import pytest

from loom import recipes, tasks, tile
from loom.recipes import ONLINE_META

TASK = tasks.junta(4, 2, k=2)
SEEDS = (0, 1, 2)
GRID = (3, 10, 20, 30, 50, 100, 200, 400)


@cache
def trained(seed):
    """η, J and the deployed loss along the run under the true signal, once per seed."""
    return recipes.train(ONLINE_META, TASK, seed)


@cache
def floor():
    """The η on which the swept objective (mean over the seeds) is within 4× its minimum."""
    J = {
        eta: sum(float(recipes.landscape(ONLINE_META, eta, TASK, s)[0]) for s in SEEDS)
        for eta in GRID
    }
    low = [eta for eta in GRID if J[eta] <= 4 * min(J.values())]
    assert J[GRID[0]] > 10 * min(J.values()) < J[GRID[-1]] / 10  # a landscape, not a plateau
    assert max(low) / min(low) < 5  # with a floor narrower than a factor 5: something to find
    return min(low), max(low)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_learned_step_size_settles_on_the_landscape_floor_and_adapts_a_held_out_task(seed):
    lo, hi = floor()
    etas, losses, _ = trained(seed)
    eta, J = (float(jnp.median(a[-200:])) for a in (etas, losses))  # where the loop settles
    assert lo <= eta <= hi  # the loop settles where the sweep says the optimum is
    assert J < losses[0] / 10  # and the objective is at that floor
    t, x, y = recipes.adapt(ONLINE_META, eta, TASK, seed, steps=500)
    assert tile.accuracy(t, x, y, "hard") == 1.0


@pytest.mark.parametrize("seed", SEEDS)
def test_under_the_sign_flipped_signal_eta_falls_and_nothing_trains(seed):
    etas, losses, _ = recipes.train(ONLINE_META._replace(control="flipped"), TASK, seed)
    assert etas[-1] < ONLINE_META.eta0  # η can only shrink: the direction control
    assert losses[-1] > 0.9 * losses[0]  # untrained


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("control", ("shuffled", "output_only"))
def test_under_the_shuffled_and_the_output_only_signals_the_loss_stays_near_untrained(
    seed, control
):
    _, losses, _ = recipes.train(ONLINE_META._replace(control=control), TASK, seed)
    assert (
        losses[-1] > (losses[0] + trained(seed)[1][-1]) / 2
    )  # closer to untrained than to trained
