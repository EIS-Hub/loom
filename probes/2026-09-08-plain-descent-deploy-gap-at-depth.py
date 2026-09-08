"""Probe: why the reference on the deep shape stops short of the target below its rate. Two
readings: a deploy gap (plain descent scales the step by the residual, so the push vanishes as
the soft output nears the target; if the tables have not saturated by then the soft read is right
and the bits are not, and nothing closes it), or a soft-pass optimum short of the target (the soft
read is wrong too, and the gradient is small or it is not: a minimum, or a plateau descent is
still crossing). Measured on `(6, 32, 32, 16, 4)` at arity 3 on 6-bit addition, probe seeds 0–2,
4000 steps: soft and hard accuracy, the loss, the fraction of table entries still unsaturated
(|z| < 2) and the gradient's norm against its norm at initialisation, for the reference at
300 / 1000 / 3000 and the relay to the entry at 700 / 1500."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import time

import jax
import jax.numpy as jnp

from loom import signals, tasks, tile
from loom.signals import REFERENCE, Signal, compute
from loom.tile import Tile

WIDTHS, ARITY, SEEDS, STEPS = (6, 32, 32, 16, 4), 3, 3, 4000
x, y = tasks.add(6)
CELLS = [(REFERENCE, (300.0, 1000.0, 3000.0)), (Signal("soft", "relay", "entry"), (700.0, 1500.0))]


def fit(signal, t, lr):
    def step(logits, _):
        s = compute(signal, Tile(logits, t.wires), x, y)
        return tuple(lg - lr * g for lg, g in zip(logits, s, strict=True)), None

    def norm(tt):
        return jnp.sqrt(sum(jnp.sum(g**2) for g in compute(REFERENCE, tt, x, y)))

    logits, _ = jax.lax.scan(step, t.logits, None, length=STEPS)
    end = Tile(logits, t.wires)
    unsaturated = jnp.mean(jnp.concatenate([jnp.abs(lg.ravel()) < 2.0 for lg in logits]))
    return (
        tile.accuracy(end, x, y, "soft"),
        tile.accuracy(end, x, y, "hard"),
        signals.loss(end, x, y, "soft"),
        unsaturated,
        norm(end) / norm(t),
    )


t0 = time.time()
print(f"shape {WIDTHS} arity {ARITY}, addition, plain descent for {STEPS} steps, seeds 0-2, CPU")
head = f"{'soft acc':>9s}  {'hard acc':>9s}  {'loss':>8s}  {'|z|<2':>6s}  {'|g|/|g0|':>8s}"
print(f"{'signal':20s}{'rate':>6s}  {'seed':>4s}  {head}")
for signal, rates in CELLS:
    batched = jax.jit(
        jax.vmap(jax.vmap(fit, in_axes=(None, 0, None)), in_axes=(None, None, 0)), static_argnums=0
    )
    tiles = jax.vmap(lambda s: tile.init(jax.random.key(s), WIDTHS, ARITY))(jnp.arange(SEEDS))
    out = batched(signal, tiles, jnp.array(rates))
    for i, lr in enumerate(rates):
        for s in range(SEEDS):
            soft, hard, loss, unsat, g = (float(o[i, s]) for o in out)
            row = f"{soft:9.3f}  {hard:9.3f}  {loss:8.5f}  {unsat:6.2f}  {g:8.4f}"
            print(f"{signal.label:20s}{lr:6.0f}  {s:4d}  {row}")
print(f"\nwall-clock {time.time() - t0:.0f} s")
