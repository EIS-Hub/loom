"""Audit probe (adversarial audit of 2026-09-07, lifted 2026-09-08): does the reach count the
outputs a flip changes, or the live paths to them? Flip every hidden line per case and count the
outputs that actually change against reach·N. Then the credit's exactness, entry by entry: the
predicted loss change of one flip against the actual one, at initialisation and at the end of
training; and an exhaustive single-flip scan of the end state, with an oracle continuation."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import jax
import jax.numpy as jnp
import optax

from loom import signals, tasks, tile
from loom.signals import Signal
from loom.tile import activations, tables

WIDTHS, ARITY, SEEDS = (6, 32, 32, 16, 4), 3, 3
x, y = tasks.add(6)
N, n_out = y.size, y.shape[1]
FLIP = Signal("hard", "flip", "entry")


def reach_vs_outputs(t):
    acts = activations(t, x, "hard")
    ones = jnp.ones((x.shape[0], n_out)) / N
    reach = signals.backward(
        t, acts, "hard", ones, lambda tb, u: jnp.abs(signals.sensitivity(tb, u))
    )
    hard = tuple(tables(lg, "hard") for lg in t.logits)
    out = []
    for layer in range(len(t.logits) - 1):
        predicted = reach[layer] * N  # [B, gates]: live paths to the outputs
        changed = []
        for g in range(t.logits[layer].shape[0]):
            flipped = acts[layer + 1].at[:, g].set(1 - acts[layer + 1][:, g])
            outs = tile.run(hard[layer + 1 :], t.wires[layer + 1 :], flipped)[-1]
            changed.append(jnp.sum(outs != acts[-1], -1))
        actual = jnp.stack(changed, 1)
        out.append(
            (
                float(jnp.mean(predicted == actual)),
                float(jnp.mean(predicted > actual)),
                float(jnp.mean((predicted > 0) & (actual == 0))),
                int(predicted.max()),
            )
        )
    return out


def flip_deltas(t):
    """The actual hard-loss change of flipping every single entry, per layer."""
    base = signals.loss(t, x, y, "hard")
    out = []
    for i, lg in enumerate(t.logits):
        gs, As = (v.ravel() for v in jnp.meshgrid(*map(jnp.arange, lg.shape), indexing="ij"))

        def flipped(g, a, i=i):
            new = t.logits[i].at[g, a].multiply(-1.0)
            return signals.loss(
                t._replace(logits=t.logits[:i] + (new,) + t.logits[i + 1 :]), x, y, "hard"
            )

        out.append((jax.vmap(flipped)(gs, As) - base).reshape(lg.shape))
    return out


def exactness(t):
    s, d = signals.compute(FLIP, t, x, y), flip_deltas(t)
    rows = []
    for a, lg, dl in zip(s, t.logits, d, strict=True):
        predicted = a * (1 - 2 * tables(lg, "hard"))
        either = (predicted != 0) | (dl != 0)
        exact = jnp.isclose(predicted, dl, atol=1e-6)[either]
        sign = (jnp.sign(predicted) == jnp.sign(dl))[either]
        rows.append(f"{float(jnp.mean(exact)):.2f}/{float(jnp.mean(sign)):.2f}")
    return " ".join(rows), sum(int((dl < 0).sum()) for dl in d)


def train(t, steps=2000):
    opt = optax.adam(0.02)
    state = opt.init(t.logits)

    @jax.jit
    def step(logits, state):
        updates, state = opt.update(
            signals.compute(FLIP, t._replace(logits=logits), x, y), state, logits
        )
        return optax.apply_updates(logits, updates), state

    logits = t.logits
    for _ in range(steps):
        logits, state = step(logits, state)
    return t._replace(logits=logits)


def oracle(t, budget=40):
    """Take the best actual single flip repeatedly from the end state; how much is left?"""
    for k in range(budget):
        flat = jnp.concatenate([dl.ravel() for dl in flip_deltas(t)])
        j = int(jnp.argmin(flat))
        if float(flat[j]) >= 0:
            return k, float(tile.accuracy(t, x, y, "hard"))
        off = 0
        for i, lg in enumerate(t.logits):
            if j < off + lg.size:
                g, a = divmod(j - off, lg.shape[1])
                new = lg.at[g, a].multiply(-1.0)
                t = t._replace(logits=t.logits[:i] + (new,) + t.logits[i + 1 :])
                break
            off += lg.size
    return budget, float(tile.accuracy(t, x, y, "hard"))


print(f"shape {WIDTHS} arity {ARITY}, addition, bits; per hidden layer input → output")
print("reach vs the outputs actually changed by flipping a line:")
print("equal / reach larger / reach > 0 but none changed / max reach")
for seed in range(SEEDS):
    t = tile.init(jax.random.key(seed), WIDTHS, ARITY)
    print(
        f"seed {seed}: "
        + "   ".join(f"{e:.2f} / {gt:.2f} / {z:.2f} / {m}" for e, gt, z, m in reach_vs_outputs(t))
    )
print("\ncredit exactness per layer, over entries where either side is nonzero:")
print("exact fraction / sign agreement")
for seed in range(SEEDS):
    t = tile.init(jax.random.key(seed), WIDTHS, ARITY)
    at_init, _ = exactness(t)
    done = train(t)
    at_end, improving = exactness(done)
    k, after = oracle(done)
    acc = float(tile.accuracy(done, x, y, "hard"))
    print(f"seed {seed}: init {at_init}   end {at_end}   acc {acc:.3f}")
    print(
        f"        improving single flips left {improving}; oracle +{k} flips → {after:.3f}",
        flush=True,
    )
