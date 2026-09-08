"""Audit probe (adversarial audit of 2026-09-07, lifted 2026-09-08): where does the exact relay
stop training on the bits? The majority-class baseline of the task; the relay against the uniform
split at one, two and three hidden layers; smaller rates, sign-SGD and init scales at depth; and
two junta tasks on the deep shape. "A signal for shallow circuits" gets its number of layers."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax
import jax.numpy as jnp
import optax

from loom import signals, tasks, tile
from loom.signals import Signal

DEEP, ARITY = (6, 32, 32, 16, 4), 3
x, y = tasks.add(6)
RELAY, ST, UNIFORM, FLIP = (
    Signal("hard", "relay", "entry"),
    Signal("hard", "relay", "logit"),
    Signal("hard", "uniform", "entry"),
    Signal("hard", "flip", "entry"),
)


def train(sig, t, x, y, steps=2000, opt=None):
    opt = optax.adam(0.02) if opt is None else opt
    state = opt.init(t.logits)

    @jax.jit
    def step(logits, state):
        updates, state = opt.update(
            signals.compute(sig, t._replace(logits=logits), x, y), state, logits
        )
        return optax.apply_updates(logits, updates), state

    logits = t.logits
    for _ in range(steps):
        logits, state = step(logits, state)
    return float(tile.accuracy(t._replace(logits=logits), x, y, "hard"))


def row(name, sig, widths=DEEP, scale=1.0, x=x, y=y, **kw):
    accs = [
        train(sig, tile.init(jax.random.key(s), widths, ARITY, scale), x, y, **kw) for s in (0, 1)
    ]
    print(f"{name:44s}" + "  ".join(f"{a:.3f}" for a in accs), flush=True)


majority = float(jnp.mean(jnp.maximum(y.mean(0), 1 - y.mean(0))))
print(f"6-bit addition: majority-class baseline {majority:.3f}\n")
print("depth (arity 3, Adam 0.02, 2000 steps): hard accuracy, seeds 0, 1")
for widths in ((6, 16, 4), (6, 32, 4), (6, 32, 16, 4), DEEP):
    for name, sig in (
        ("relay.entry", RELAY),
        ("relay.logit (straight-through)", ST),
        ("uniform.entry", UNIFORM),
    ):
        row(f"{widths} {name}", sig, widths)
print("\nthe deep shape under other conditions")
row("relay.entry Adam 0.005 x 4000", RELAY, steps=4000, opt=optax.adam(0.005))
row("relay.entry Adam 0.001 x 4000", RELAY, steps=4000, opt=optax.adam(0.001))
row(
    "relay.entry sign-SGD 0.002 x 5000",
    RELAY,
    steps=5000,
    opt=optax.chain(optax.scale_by_sign(), optax.sgd(0.002)),
)
for scale in (0.3, 3.0, 10.0):
    row(f"relay.entry init scale {scale}", RELAY, scale=scale)
    row(f"uniform.entry init scale {scale}", UNIFORM, scale=scale)
print("\njunta tasks on the deep shape (Adam 0.02, 2000 steps)")
for k in (2, 3):
    xj, yj = tasks.inputs(6), tasks.k_junta(jax.random.key(100), 6, 4, k=k)
    for name, sig in (("relay.entry", RELAY), ("uniform.entry", UNIFORM), ("flip.entry", FLIP)):
        row(f"{k}-junta {name}", sig, x=xj, y=yj)
