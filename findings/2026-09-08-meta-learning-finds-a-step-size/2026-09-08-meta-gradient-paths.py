"""The two derivative paths of the meta-gradient: through eta's linear entry only (first order,
the signal as data) and through the signal as well; on the soft relay and on the bits' uniform
split, across eta, with a central finite difference beside them; how large is the second-order
term on the soft pass, and is it exactly zero on the bits?"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import partial  # noqa: E402

import jax  # noqa: E402

from loom import meta, tasks, tile  # noqa: E402
from loom.signals import Signal  # noqa: E402

CELLS = {
    "soft.relay.entry": Signal("soft", "relay", "entry"),
    "hard.uniform.entry": Signal("hard", "uniform", "entry"),
}
ETAS = [3.0, 30.0, 300.0]
K, H = 8, 0.5


def setup(seed):
    k_task, k_tile, k_run = jax.random.split(jax.random.key(seed), 3)
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    return tile.init(k_tile, (4, 16, 8, 2)), x, y, k_run


def objective(eta, t, x, y, k, sig, first_order=False):
    return meta.rollout(eta, t, x, y, k, signal=sig, steps=K, first_order=first_order)[0]


for name, sig in CELLS.items():
    print(f"\n{name}, K={K}: dJ/deta full / first-order / finite difference (h={H}), seeds 0-2")
    for eta in ETAS:
        rows = []
        for seed in range(3):
            t, x, y, k = setup(seed)
            j = partial(objective, t=t, x=x, y=y, k=k, sig=sig)
            full = float(jax.grad(j)(eta))
            first = float(jax.grad(partial(j, first_order=True))(eta))
            fd = float((j(eta + H) - j(eta - H)) / (2 * H))
            rows.append(f"{full:+.2e}/{first:+.2e}/{fd:+.2e}")
        print(f"  eta={eta:>6.1f}  " + " · ".join(rows))
