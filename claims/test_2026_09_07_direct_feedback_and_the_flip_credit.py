"""Claims of notes/2026-09-07-direct-feedback-and-the-flip-credit.md: on the bits at depth, the
flip credit reaches a fixed point, and the blind transports order as wiring-shaped, random, none."""

import jax.numpy as jnp

from loom import recipes, signals, tasks, tile
from loom.recipes import DEEP_HARD_FLOOR
from loom.signals import Signal

ADD = tasks.addition(6)
SEEDS = range(3)


def on_the_bits(via, seed):
    t, x, y = recipes.run(DEEP_HARD_FLOOR._replace(signal=Signal("hard", via, "entry")), ADD, seed)
    return t, x, y, float(tile.accuracy(t, x, y, "hard"))


def still_pushed(via, seed):
    """After training, how many entries the signal still pushes toward a flip, and the accuracy."""
    t, x, y, acc = on_the_bits(via, seed)
    s = signals.compute(Signal("hard", via, "entry"), t, x, y)
    toward_a_flip = [
        jnp.sum(a * (1.0 - 2.0 * tile.tables(lg, "hard")) < 0)
        for a, lg in zip(s, t.logits, strict=True)
    ]
    return sum(int(n) for n in toward_a_flip), acc


def test_the_flip_credit_comes_to_rest_where_the_blind_split_keeps_pushing():
    for seed in SEEDS:
        at_rest, acc = still_pushed("flip", seed)
        pushed, _ = still_pushed("uniform", seed)
        assert (
            at_rest * 10 < pushed
        )  # a fixed point of single flips (zero on most seeds), not a drift
        assert acc > 0.8  # a local optimum well clear of chance, short of the target


def test_on_the_bits_the_blind_transports_order_as_wiring_shaped_random_none():
    for seed in SEEDS:
        acc = {via: on_the_bits(via, seed)[3] for via in ("uniform", "direct", "relay")}
        assert acc["uniform"] > acc["direct"] > acc["relay"]
        assert abs(acc["relay"] - 0.5) < 0.15  # the exact relay stays at chance
