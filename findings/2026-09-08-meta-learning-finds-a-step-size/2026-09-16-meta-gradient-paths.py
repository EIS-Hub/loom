"""The meta-gradient's two derivative paths per cell: through η's linear entry only (first order,
the signal as data) and through the signal as well; a central finite difference beside them.
Which cells have a second-order term: the soft pass (the Hessian); the ``entry`` cells of the hard
pass (none: the signal has no derivative in z); the ``logit`` cells of the hard pass (a surrogate
one, through σ′ of the stored logit and the straight-through tables). K = 8, three tiles; the
mechanics test pins the same statement at K = 3. A measurement the loop cannot express: it takes
the rollout's gradient at one tile, so it calls `meta.rollout` on the recipe's shape directly."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import partial  # noqa: E402

import jax  # noqa: E402

from loom import meta, tasks, tile  # noqa: E402
from loom.signals import Signal  # noqa: E402

CELLS = (
    "soft.relay.entry",
    "soft.relay.logit",
    "hard.uniform.entry",
    "hard.relay.entry",
    "hard.flip.entry",
    "hard.relay.logit",
    "hard.autodiff.logit",
)
ETAS = (3.0, 30.0)
K, H = 8, 0.5


def setup(seed):
    k_task, k_tile, k_run = jax.random.split(jax.random.key(seed), 3)
    x, y = tasks.inputs(4), tasks.k_junta(k_task, 4, 2, k=2)
    return tile.init(k_tile, (4, 16, 8, 2)), x, y, k_run


def objective(eta, t, x, y, k, sig, first_order):
    return meta.rollout(eta, t, x, y, k, signal=sig, steps=K, first_order=first_order)[0]


for name in CELLS:
    sig = Signal(*name.split("."))  # type: ignore[arg-type]
    print(f"\n{name}, K={K}: dJ/dη full / first-order / finite difference (h={H}), seeds 0-2")
    for eta in ETAS:
        rows = []
        for seed in range(3):
            t, x, y, k = setup(seed)
            j = partial(objective, t=t, x=x, y=y, k=k, sig=sig)
            full = float(jax.grad(j)(eta, first_order=False))
            first = float(jax.grad(j)(eta, first_order=True))
            fd = float((j(eta + H, first_order=False) - j(eta - H, first_order=False)) / (2 * H))
            rows.append(f"{full:+.2e}/{first:+.2e}/{fd:+.2e}")
        print(f"  η={eta:>5.1f}  " + " · ".join(rows), flush=True)
