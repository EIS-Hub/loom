"""Probe: on the bits, who learns? Both hard signals are identical at the output layer (nothing
has been transported yet), so any difference between them is what their hidden-layer updates do.
Adam 0.02 with the output layer always updated and the hidden layers frozen (p = 0), throttled
(p = 0.02, 0.1) or free (p = 1). If the blind split with free hidden layers does no better than
frozen ones, its hidden signal does nothing useful and it wins by doing no harm."""

import jax
import optax

from loom import signals, tasks, tile
from loom.signals import Signal

WIDTHS, ARITY, STEPS, SEEDS = (6, 32, 32, 16, 4), 3, 2000, 3
x, y = tasks.add(6)


def run(sig, p_hidden, t, key):
    opt = optax.adam(0.02)
    state = opt.init(t.logits)

    @jax.jit
    def step(logits, state, key):
        s = signals.compute(sig, t._replace(logits=logits), x, y)
        keys = jax.random.split(key, len(logits))
        ps = [p_hidden] * (len(logits) - 1) + [1.0]
        masked = tuple(
            a * jax.random.bernoulli(k, p, a.shape) for a, k, p in zip(s, keys, ps, strict=True)
        )
        updates, state = opt.update(masked, state, logits)
        return optax.apply_updates(logits, updates), state

    logits = t.logits
    for _ in range(STEPS):
        key, k = jax.random.split(key)
        logits, state = step(logits, state, k)
    return float(tile.accuracy(t._replace(logits=logits), x, y, "hard"))


print(f"shape {WIDTHS} arity {ARITY}, addition, Adam 0.02, {STEPS} steps: hard accuracy per seed")
print(f"{'signal':16s} {'hidden p':>9s}   seeds 0, 1, 2")
for sig in (Signal("hard", "relay", "entry"), Signal("hard", "uniform", "entry")):
    for p in (0.0, 0.02, 0.1, 1.0):
        accs = [
            run(sig, p, tile.init(jax.random.key(s), WIDTHS, ARITY), jax.random.key(s))
            for s in range(SEEDS)
        ]
        print(f"{sig.label:16s} {p:9.2f}   " + "  ".join(f"{a:.3f}" for a in accs), flush=True)
