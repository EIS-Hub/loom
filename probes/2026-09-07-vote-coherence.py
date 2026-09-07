"""Probe: the votes an entry receives across the cases that address it: do they agree? For each
signal and layer, the per-entry coherence |Σ_b e_b P_b(a)| / Σ_b |e_b| P_b(a), averaged over entries
that receive anything (1 = every case pushes the same way, 0 = the cases cancel). The relay's
message per case carries a product of ±1 sensitivities that depends on that case's other inputs;
the uniform split's carries a constant. Exact per case may still be noise per entry."""

import jax
import jax.numpy as jnp

from loom import signals, tasks, tile
from loom.signals import Signal

WIDTHS, ARITY, SEEDS = (6, 32, 32, 16, 4), 3, 3
x, y = tasks.add(6)
layers = range(len(WIDTHS) - 1)


def coherence(sig, t):
    acts = tile.activations(t, x, sig.on)
    e = (acts[-1] - y) / y.size
    carry = signals.sensitivity if sig.via == "relay" else (lambda luts, u: jnp.ones_like(u))
    out = []
    for lg, w, u in reversed(list(zip(t.logits, t.wires, acts[:-1], strict=True))):
        luts, inputs = tile.tables(lg, sig.on), u[:, w]
        p = signals.address(inputs)
        net = jnp.abs(jnp.einsum("bg,bga->ga", e, p))
        gross = jnp.einsum("bg,bga->ga", jnp.abs(e), p)
        out.append(float(jnp.mean(net[gross > 0] / gross[gross > 0])))
        e = jnp.zeros_like(u).at[:, w].add(e[:, None, :] * carry(luts, inputs))
    return out[::-1]


print(f"shape {WIDTHS} arity {ARITY}, addition, at initialisation, {SEEDS} seeds")
print("vote coherence per entry, layers input → output (1 = the cases agree, 0 = they cancel)")
for sig in (
    Signal("soft", "relay", False),
    Signal("soft", "uniform", False),
    Signal("hard", "relay", False),
    Signal("hard", "uniform", False),
):
    rows = [coherence(sig, tile.init(jax.random.key(s), WIDTHS, ARITY)) for s in range(SEEDS)]
    print(f"{sig.label:16s}" + "".join(f"{sum(r[i] for r in rows) / SEEDS:8.2f}" for i in layers))
