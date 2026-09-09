"""Claims of notes/2026-09-08-the-floors-under-plain-descent.md: what holds once descent is the
plain update, Δ = −lr·s, with every rate and budget re-pinned."""

from loom import recipes, tasks, tile
from loom.recipes import DEEP_FLOOR, DEEP_HARD_FLOOR, DEEP_RELAY, HARD_FLOOR
from loom.signals import Signal

ADD = tasks.addition(6)
SEEDS = range(3)


def test_at_depth_the_plain_reference_comes_within_two_bits_and_the_relay_to_the_entry_reaches():
    """Re-states the retracted `test_addition_under_the_deep_floor` (Adam reached the target on
    every seed): under plain descent the reference has no rate that does so on every tile, while
    the relay to the entry, at its own rate, does."""
    for seed in SEEDS:
        t, x, y = recipes.run(DEEP_FLOOR, ADD, seed)
        assert tile.accuracy(t, x, y, "hard") >= 0.98  # within two bits of the 256 in the table
        t, x, y = recipes.run(DEEP_RELAY, ADD, seed)
        assert tile.accuracy(t, x, y, "hard") == 1.0


def test_the_same_signal_is_short_under_the_references_rate_and_reaches_under_its_own():
    """A signal's magnitude is now the step: the relay to the entry under the reference's rate
    falls to the bits regime and stays well short of the target."""
    for seed in SEEDS:
        t, x, y = recipes.run(DEEP_FLOOR._replace(signal=DEEP_RELAY.signal), ADD, seed)
        assert tile.accuracy(t, x, y, "hard") < 0.8
        t, x, y = recipes.run(DEEP_RELAY, ADD, seed)
        assert tile.accuracy(t, x, y, "hard") == 1.0


def test_straight_through_trains_the_bits_at_depth_where_the_relay_to_the_entry_does_not():
    """Under Adam neither left chance at four layers; the plain update keeps σ′ as a weight, and
    with it straight-through trains, well clear of chance and short of the target."""
    for seed in SEEDS:
        t, x, y = recipes.run(DEEP_HARD_FLOOR, ADD, seed)
        assert tile.accuracy(t, x, y, "hard") > 0.8
        t, x, y = recipes.run(
            DEEP_HARD_FLOOR._replace(signal=Signal("hard", "relay", "entry")), ADD, seed
        )
        assert (
            abs(float(tile.accuracy(t, x, y, "hard")) - 0.5) < 0.1
        )  # chance: majority class 0.516


def test_two_bit_addition_under_the_hard_floor():
    """The hard floor of the flat tile now holds on both flat tasks (its rate is the centre of the
    band straight-through wants on addition, which the junta's band contains)."""
    for seed in SEEDS:
        t, x, y = recipes.run(HARD_FLOOR, tasks.addition(4), seed)
        assert tile.accuracy(t, x, y, "hard") == 1.0
