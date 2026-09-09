"""Claims of notes/2026-09-09-a-readout-at-the-cell-makes-the-deep-floor-a-band.md: at depth the
plain rule reaches the target at one rate and not a decade below; the sign of the vote (no state)
and RMSprop (one accumulator) reach at both."""

from loom import recipes, tasks, tile
from loom.recipes import (
    DEEP_RELAY,
    DEEP_RELAY_TENTH,
    DEEP_RMSPROP,
    DEEP_RMSPROP_TENTH,
    DEEP_SIGN,
    DEEP_SIGN_TENTH,
)

ADD = tasks.addition(6)
SEEDS = range(3)


def reaches(recipe):
    return [float(tile.accuracy(*recipes.run(recipe, ADD, seed), "hard")) == 1.0 for seed in SEEDS]


def test_the_plain_rule_reaches_at_its_rate_and_not_a_decade_below():
    assert all(reaches(DEEP_RELAY))
    assert not all(reaches(DEEP_RELAY_TENTH))


def test_the_sign_of_the_vote_reaches_at_two_rates_a_decade_apart():
    assert all(reaches(DEEP_SIGN)) and all(reaches(DEEP_SIGN_TENTH))


def test_rmsprop_reaches_at_two_rates_a_decade_apart():
    assert all(reaches(DEEP_RMSPROP)) and all(reaches(DEEP_RMSPROP_TENTH))
