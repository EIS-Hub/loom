"""Claims of notes/2026-09-07-the-relay-is-autodiff-and-the-bits-learn-by-alignment.md: the three
transports on a shape where depth is forced (6-input addition on four layers of arity 3)."""

from loom import recipes, signals, tasks, tile
from loom.recipes import DEEP_FLOOR, DEEP_HARD_FLOOR
from loom.signals import REFERENCE, Signal

ADD = tasks.addition(6)
SEEDS = range(3)


def sign_by_layer(sig, t, x, y):
    g, r = signals.compute(sig, t, x, y), signals.compute(REFERENCE, t, x, y)
    return [float(signals.score((g[i],), (r[i],))["sign"]) for i in range(len(g))]


def test_addition_under_the_deep_floor():
    for seed in SEEDS:
        t, x, y = recipes.run(DEEP_FLOOR, ADD, seed)
        assert tile.accuracy(t, x, y, "hard") == 1.0


def test_the_uniform_split_loses_the_reference_sign_after_one_hop():
    hidden = []
    for seed in SEEDS:
        t, x, y, _ = recipes.setup(DEEP_FLOOR, ADD, seed)
        blind = sign_by_layer(Signal("soft", "uniform", surrogate=False), t, x, y)
        assert blind[-1] > 0.99  # at the output layer nothing has been transported yet
        assert all(s < 0.75 for s in blind[:-1])  # lost from the first hop on (the relay's is 1.0)
        hidden += blind[:-1]
    assert abs(sum(hidden) / len(hidden) - 0.5) < 0.1  # and at chance, over layers and seeds


def test_descent_on_the_relay_reaches_the_target_where_the_blind_split_stalls():
    for seed in SEEDS:
        for via, reaches in (("relay", True), ("uniform", False)):
            recipe = DEEP_FLOOR._replace(signal=Signal("soft", via, surrogate=False))
            t, x, y = recipes.run(recipe, ADD, seed)
            assert (tile.accuracy(t, x, y, "hard") == 1.0) == reaches


def test_on_the_bits_at_depth_the_exact_relay_is_dead_and_the_blind_split_is_not():
    for seed in SEEDS:
        acc = {}
        for via in ("relay", "uniform"):
            recipe = DEEP_HARD_FLOOR._replace(signal=Signal("hard", via, surrogate=False))
            t, x, y = recipes.run(recipe, ADD, seed)
            acc[via] = float(tile.accuracy(t, x, y, "hard"))
        assert abs(acc["relay"] - 0.5) < 0.15  # chance: the signal reaches almost no gate
        assert acc["uniform"] > 0.75  # well clear of it, though short of the target
