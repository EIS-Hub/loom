"""Probe: do the gates make whatever fixed feedback they are given right? For a fixed feedback
B[g, o] (direct: random ±1 per hidden gate; uniform: the wiring's path counts, all positive), the
alignment is the sign agreement between B and the circuit's actual Jacobian ∂r_o/∂r_g on the bits
(the layered adjoint with the true sensitivities, seeded one output at a time), over the pairs
where that Jacobian is nonzero. Measured per hidden layer along training under each signal."""

import jax
import jax.numpy as jnp
import optax

from loom import signals, tasks, tile
from loom.signals import Signal

WIDTHS, ARITY, SEEDS = (6, 32, 32, 16, 4), 3, 3
x, y = tasks.add(6)
n_out = y.shape[1]
hidden = range(len(WIDTHS) - 2)


def jacobian(t):
    """∂r_o/∂r_g on the bits for every case, hidden layer by layer: [B, gates, n_out]."""
    acts = tile.activations(t, x, "hard")
    per_out = []
    for o in range(n_out):
        e = jnp.zeros((x.shape[0], n_out)).at[:, o].set(1.0)
        per_out.append(signals.layered(t, acts, "hard", e, signals.sensitivity)[:-1])
    return [jnp.stack([po[i] for po in per_out], axis=-1) for i in hidden]


def alignment(t, feedback):
    jac = jacobian(t)
    out = []
    for j, b in zip(jac, feedback, strict=True):
        live = j != 0
        out.append(
            float(
                jnp.sum((jnp.sign(j) == jnp.sign(b)[None]) & live) / jnp.maximum(jnp.sum(live), 1)
            )
        )
    return out


def train(sig, t, steps):
    opt = optax.adam(0.02)
    state = opt.init(t.logits)

    @jax.jit
    def step(logits, state):
        s = signals.compute(sig, t._replace(logits=logits), x, y)
        updates, state = opt.update(s, state, logits)
        return optax.apply_updates(logits, updates), state

    logits = t.logits
    for _ in range(steps):
        logits, state = step(logits, state)
    return t._replace(logits=logits)


def counts(t):
    acts = [a[:n_out] for a in tile.activations(t, x, "soft")]
    return [c.T for c in signals.layered(t, acts, "soft", jnp.eye(n_out), signals.ones)][:-1]


print(
    f"shape {WIDTHS} arity {ARITY}, addition, Adam 0.02: alignment of the Jacobian to the feedback"
)
print(f"{'signal':20s} {'step':>5s}   hidden layers input → output   hard acc")
for sig in (
    Signal("hard", "direct", "entry"),
    Signal("hard", "uniform", "entry"),
    Signal("hard", "relay", "entry"),
):
    for steps in (0, 200, 1000, 2000):
        al, acc = [], []
        for seed in range(SEEDS):
            t = train(sig, tile.init(jax.random.key(seed), WIDTHS, ARITY), steps)
            fb = signals.feedback(t, n_out)[:-1] if sig.via == "direct" else counts(t)
            al.append(alignment(t, fb)), acc.append(float(tile.accuracy(t, x, y, "hard")))
        means = [sum(a[i] for a in al) / SEEDS for i in hidden]
        print(
            f"{sig.label:20s} {steps:5d}   "
            + " ".join(f"{m:.2f}" for m in means)
            + f"              {sum(acc) / SEEDS:.3f}",
            flush=True,
        )
