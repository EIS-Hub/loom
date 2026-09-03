"""Tasks are truth tables: every input pattern, and the bits demanded of each output line.

Re-lifted 2026-09 from blastema/tasks (build_task_x, sample_k_junta_y, binary_add), the first two
themselves from boolean_nca_cc. Kept: the per-output-bit k-junta family and two-operand addition.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def inputs(n_in: int) -> jax.Array:
    """All 2**n_in input patterns as bits, least significant first: [2**n_in, n_in]."""
    ints = jnp.arange(1 << n_in)
    return ((ints[:, None] >> jnp.arange(n_in)[None, :]) & 1).astype(jnp.float32)


def k_junta(key: jax.Array, n_in: int, n_out: int, k: int, balanced: bool = True) -> jax.Array:
    """A random task whose every output bit depends on k random inputs: [2**n_in, n_out].

    Each output bit draws its own k-subset of the inputs and its own table over them; ``balanced``
    tables have exactly half ones, so no output bit is a constant. Solving it means discovering
    which inputs matter for each output and what function of them is asked.
    """
    x = inputs(n_in).astype(jnp.int32)
    powers = 1 << jnp.arange(k)
    half = jnp.concatenate([jnp.zeros((1 << k) // 2), jnp.ones((1 << k) - (1 << k) // 2)])

    def one_output(k_sub, k_tab):
        subset = jax.random.permutation(k_sub, n_in)[:k]
        if balanced:
            table = jax.random.permutation(k_tab, half)
        else:
            table = jax.random.bernoulli(k_tab, 0.5, (1 << k,))
        return table[jnp.sum(x[:, subset] * powers, axis=-1)].astype(jnp.float32)

    k_subs, k_tabs = jax.random.split(key, 2)
    y = jax.vmap(one_output)(jax.random.split(k_subs, n_out), jax.random.split(k_tabs, n_out))
    return y.T


def add(n_in: int) -> tuple[jax.Array, jax.Array]:
    """Add the two halves of the input as unsigned integers; n_in//2 + 1 output bits (the carry)."""
    half = n_in // 2
    ints = jnp.arange(1 << n_in)
    total = (ints & ((1 << half) - 1)) + (ints >> half)
    n_out = half + 1
    y = ((total[:, None] >> jnp.arange(n_out)[None, :]) & 1).astype(jnp.float32)
    return inputs(n_in), y
