"""One tile: a layered circuit of look-up tables (LUTs) over a fixed wiring.

A tile is data: per-layer table logits and per-layer wiring, nothing else. Two reads of the same
data: the soft read (tables as probabilities) is what gradients flow through; the hard read (tables
rounded to bits) is the deployed circuit. Every step above this one is measured on the hard read.

Re-lifted 2026-09 from blastema/substrate/circuit.py (run_layer, gen_wires), itself lifted from
boolean_nca_cc. Dropped: gate groups, gate masks, the nop and noise inits.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Tile(NamedTuple):
    logits: tuple[jax.Array, ...]  # layer l: [gates_l, 2**arity], the table logits
    wires: tuple[jax.Array, ...]  # layer l: [arity, gates_l], indices into the previous layer


def init(key: jax.Array, widths: tuple[int, ...], arity: int = 4, scale: float = 1.0) -> Tile:
    """Random tables over a random fixed wiring.

    ``widths`` = (n_in, hidden..., n_out) is the number of lines leaving each layer. Each gate draws
    ``arity`` inputs from the previous layer; a permutation makes every previous line feed at least
    one gate whenever the fan-in allows it.
    """
    logits, wires = [], []
    for n_prev, gates in zip(widths[:-1], widths[1:], strict=True):
        key, k_w, k_l = jax.random.split(key, 3)
        edges = gates * arity
        w = jax.random.permutation(k_w, max(n_prev, edges))[:edges].reshape(arity, gates) % n_prev
        wires.append(w)
        logits.append(scale * jax.random.normal(k_l, (gates, 2**arity)))
    return Tile(tuple(logits), tuple(wires))


def read(tables: jax.Array, inputs: jax.Array) -> jax.Array:
    """Read every gate's table at its inputs.

    ``tables`` [gates, 2**arity] in [0, 1]; ``inputs`` [B, arity, gates] in [0, 1]. Each input bit
    halves the table (a binary decision diagram, first input = least significant address bit); with
    soft inputs this is the table's expectation under the product distribution of its inputs, so the
    read is exact on bits and differentiable in between.
    """
    out = jnp.broadcast_to(tables, (inputs.shape[0], *tables.shape))  # [B, gates, 2**arity]
    for i in range(inputs.shape[1]):
        x = inputs[:, i, :, None]
        out = (1.0 - x) * out[..., ::2] + x * out[..., 1::2]
    return out[..., 0]


def activations(tile: Tile, x: jax.Array, hard: bool = False) -> list[jax.Array]:
    """Every layer's output, input first. ``x`` [B, n_in] in [0, 1].

    ``hard`` rounds the tables to bits: the deployed circuit, whose gates emit bits for bit inputs.
    """
    acts = [x]
    for lgt, w in zip(tile.logits, tile.wires, strict=True):
        tables = jax.nn.sigmoid(lgt)
        if hard:
            tables = jnp.round(tables)
        acts.append(read(tables, acts[-1][:, w]))  # x[:, w] gathers [B, arity, gates]
    return acts


def forward(tile: Tile, x: jax.Array, hard: bool = False) -> jax.Array:
    """The output lines, [B, n_out]."""
    return activations(tile, x, hard)[-1]


def accuracy(tile: Tile, x: jax.Array, y: jax.Array, hard: bool = True) -> jax.Array:
    """Fraction of output bits right over the batch; on the hard read, the deployed accuracy."""
    return jnp.mean(jnp.round(forward(tile, x, hard)) == y)
