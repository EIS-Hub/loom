"""Probe: the corpus audit's counterexample to the flip credit's exactness (Codex, 2026-09-08),
run against the code. A line h branches into two buffers that reconverge on an AND. With h = 0
and the target 1, each buffer alone cannot move the AND, so the relay and the reach both see
nothing at h, yet flipping h flips both buffers and fixes the output: the credit under-estimates.
With h = 1 and the target 0 both paths are live and the credit counts them twice: it over-counts.
Built by hand as a four-layer tile of arity-2 gates with logits at ±10 (bits)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax.numpy as jnp

from loom import signals, tile
from loom.signals import Signal
from loom.tile import Tile

BIG = 10.0
identity = jnp.array([0.0, 1.0, 0.0, 1.0])  # T[a] = bit 0 of a: copy the first input
land = jnp.array([0.0, 0.0, 0.0, 1.0])  # T[a] = bit 0 · bit 1: AND of the two inputs
logits = (
    BIG * (2 * identity - 1)[None],  # h: copies the input line
    BIG * (2 * identity - 1)[None].repeat(2, 0),  # two buffers, each copying h
    BIG * (2 * land - 1)[None],  # the AND of the two buffers
)
wires = (
    jnp.array([[0], [0]]),  # h reads the one input line on both pins
    jnp.array([[0, 0], [0, 0]]),  # each buffer reads h on both pins (only pin 0 matters)
    jnp.array([[0], [1]]),  # the AND reads buffer a on pin 0 and buffer b on pin 1
)
t = Tile(logits, wires)
FLIP = Signal("hard", "flip", "entry")

for x_bit, target in ((0.0, 1.0), (1.0, 0.0)):
    x, y = jnp.array([[x_bit]]), jnp.array([[target]])
    s = signals.compute(FLIP, t, x, y)[0]  # the credit at h's entries
    a = int(x_bit) * 3  # h addresses entry 00 when the input is 0, entry 11 when it is 1
    h_bit = float(tile.tables(t.logits[0], "hard")[0, a])
    predicted = float(s[0, a] * (1 - 2 * h_bit))
    flipped = t._replace(logits=(t.logits[0].at[0, a].multiply(-1.0), *t.logits[1:]))
    actual = float(signals.loss(flipped, x, y, "hard") - signals.loss(t, x, y, "hard"))
    out = int(tile.forward(t, x, "hard")[0, 0])
    print(f"input {int(x_bit)}, target {int(target)}: h = {int(h_bit)}, output {out};")
    print(
        f"    flipping h's addressed entry: predicted ΔL {predicted:+.2f}, actual ΔL {actual:+.2f}"
    )
