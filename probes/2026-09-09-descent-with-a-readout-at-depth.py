"""Plain descent at depth is a point in the rate; what a readout at the cell buys. The exact relay to
the entry on the deep tile under six hand-written rules, each a few lines: plain (no state), the
sign of the vote (no state), momentum, RMSprop and Lion (one accumulator per entry), Adam (two).
Rates per vote; hard accuracy at 1000, 2000 and 4000 steps, three probe seeds. The zero-state sign
readout gives the floor a band where plain descent has a point: the depth attenuation is a
magnitude problem, solved at the cell without touching the adjoint."""

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from loom import tasks, tile  # noqa: E402
from loom.signals import Signal, compute  # noqa: E402
from loom.tile import Tile, init  # noqa: E402

WIDTHS, ARITY = (6, 32, 32, 16, 4), 3
x, y = tasks.add(6)
SIG = Signal("soft", "relay", "entry")
B1, B2, B2_ADAM, EPS = 0.9, 0.99, 0.999, 1e-8


def ema(beta, state, s):
    return tuple(beta * a + (1 - beta) * g for a, g in zip(state, s, strict=True))


def plain(st, s, lr):
    return st, tuple(-lr * g for g in s)


def sign(st, s, lr):
    return st, tuple(-lr * jnp.sign(g) for g in s)


def momentum(st, s, lr):
    m = ema(B1, st[0], s)
    return (m,), tuple(-lr * a for a in m)


def rmsprop(st, s, lr):
    v = ema(B2, st[0], tuple(g * g for g in s))
    return (v,), tuple(-lr * g / (jnp.sqrt(a) + EPS) for a, g in zip(v, s, strict=True))


def adam(st, s, lr):
    m, v, t = st
    t = t + 1
    m, v = ema(B1, m, s), ema(B2_ADAM, v, tuple(g * g for g in s))
    mh = tuple(a / (1 - B1**t) for a in m)
    vh = tuple(a / (1 - B2_ADAM**t) for a in v)
    return (m, v, t), tuple(-lr * a / (jnp.sqrt(b) + EPS) for a, b in zip(mh, vh, strict=True))


def lion(st, s, lr):
    c = ema(B1, st[0], s)  # the interpolation whose sign is the step
    return (ema(B2, st[0], s),), tuple(-lr * jnp.sign(a) for a in c)


def zeros(t):
    return tuple(jnp.zeros_like(z) for z in t.logits)


def state0(name, t):
    if name in ("plain", "sign"):
        return ()
    if name == "adam":
        return (zeros(t), zeros(t), jnp.zeros(()))
    return (zeros(t),)


RULES = {
    "plain": plain,
    "sign": sign,
    "momentum": momentum,
    "rmsprop": rmsprop,
    "adam": adam,
    "lion": lion,
}
PER_VOTE = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]  # rules that keep the vote's size
PER_STEP = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]  # rules whose step is a logit unit
GRIDS = {"plain": PER_VOTE, "momentum": PER_VOTE}

tiles = [init(jax.random.key(s), WIDTHS, ARITY) for s in range(3)]
stack = jax.tree.map(lambda *a: jnp.stack(a), *tiles)


def run(name):
    rule = RULES[name]

    def one(t, lr):
        def step(carry, _):
            logits, st = carry
            st, d = rule(st, compute(SIG, Tile(logits, t.wires), x, y), lr)
            return (tuple(z + dz for z, dz in zip(logits, d, strict=True)), st), None

        carry, accs = (t.logits, state0(name, t)), []
        for chunk in (1000, 1000, 2000):
            carry, _ = jax.lax.scan(step, carry, None, length=chunk)
            accs.append(tile.accuracy(Tile(carry[0], t.wires), x, y, "hard"))
        return jnp.stack(accs)

    rates = jnp.array(GRIDS.get(name, PER_STEP))
    over_rates = jax.vmap(one, in_axes=(None, 0))
    return rates, jax.jit(jax.vmap(over_rates, in_axes=(0, None)))(stack, rates)


print(
    "deep tile, 6-bit addition, soft.relay.entry (one vote per case), hard accuracy at 4000 steps;"
    " seeds 0·1·2, bold mean; * = every seed at 1.000\n"
    "(state per entry: plain 0 · sign 0 · momentum 1 · rmsprop 1 · lion 1 · adam 2)"
)
for name in RULES:
    t0 = time.time()
    rates, a = run(name)
    print(f"  {name} ({time.time() - t0:.0f}s)")
    for i, lr in enumerate(rates):
        at4 = a[:, i, 2]
        star = "*" if bool(jnp.all(at4 == 1.0)) else " "
        cells = " · ".join(f"{float(v):.3f}" for v in at4)
        early = " ".join(f"{float(v):.2f}" for v in a[:, i, 0])
        print(
            f"    lr={float(lr):>7g} {star} {cells}  **{float(jnp.mean(at4)):.3f}**  (at 1000: {early})"
        )
