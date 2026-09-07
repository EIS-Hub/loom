"""Probe: how different are the reference gradient (soft pass) and the gradient on the bits?
Sparsity, cosine and sign agreement at initialisation, on random tiles and 2-juntas."""

import jax
import jax.numpy as jnp

from loom import signals, tasks, tile
from loom.signals import REFERENCE, Signal

x = tasks.inputs(4)
print("tile  nonzero(soft)  nonzero(bits)  cosine  sign-agreement where both nonzero")
for seed in range(3):
    k_task, k_tile = jax.random.split(jax.random.key(seed))
    y = tasks.k_junta(k_task, 4, 2, k=2)
    t = tile.init(k_tile, (4, 16, 8, 2))
    gs = jnp.concatenate([g.ravel() for g in signals.compute(REFERENCE, t, x, y)])
    gh = jnp.concatenate([g.ravel() for g in signals.compute(Signal("hard"), t, x, y)])
    cos = jnp.dot(gs, gh) / (jnp.linalg.norm(gs) * jnp.linalg.norm(gh) + 1e-12)
    both = (gs != 0) & (gh != 0)
    agree = jnp.mean(jnp.sign(gs[both]) == jnp.sign(gh[both]))
    print(
        f"{seed}     {float(jnp.mean(gs != 0)):.2f}           {float(jnp.mean(gh != 0)):.2f}"
        f"           {float(cos):+.2f}    {float(agree):.2f}"
    )
