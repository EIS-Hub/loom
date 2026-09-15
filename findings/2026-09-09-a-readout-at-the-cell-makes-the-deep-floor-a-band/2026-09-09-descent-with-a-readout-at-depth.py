"""Plain descent at depth is a point in the rate; what a readout at the cell buys. The six rules of
`rule.py` on the relay to the entry on the deep tile, each run through `recipes.trace` as an
unnamed variant of `DEEP_RELAY` (`_replace(rule=…, lr=…)`) on the recipe seeds 0, 1, 2, so the
numbers are produced by the claim's own code and seeds and the claim's six recipes are six cells
of the table. Rates per vote for the rules that keep the vote's size (plain, momentum), a grid
anchored on the recipe's rate; per step for the rules that erase it. Hard accuracy at the recipe's
budget (4000 steps), and at 1000 as a record. Prints the note's table, then every cell per seed.

Re-run on 2026-09-15 in this form; the first version (2026-09-09) re-implemented the six rules
inline and seeded its own tiles, which is what round 1 of the review could not relate to the claim.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a probe never takes a GPU on its own

import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from statistics import mean, pstdev

from loom import recipes, tasks, tile
from loom.recipes import DEEP_RELAY
from loom.rule import RULES, STATE

ADD = tasks.addition(6)
SEEDS = (0, 1, 2)
PER_VOTE = tuple(DEEP_RELAY.lr * f for f in (1 / 30, 1 / 10, 1 / 3, 1, 3, 10))  # 12.5 … 3750
PER_STEP = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)  # a logit unit per step, once the magnitude is erased
GRID = {name: PER_VOTE if name in ("plain", "momentum") else PER_STEP for name in RULES}


def cell(job):
    """One (rule, rate, seed) under the recipe: hard accuracy at 1000 and at the budget."""
    name, lr, seed = job
    t, rec, x, y = recipes.trace(DEEP_RELAY._replace(rule=name, lr=lr), ADD, seed, every=1000)
    return name, lr, seed, float(rec["hard"][0]), float(tile.accuracy(t, x, y, "hard"))


def summary(vals):
    star = "★ " if all(v == 1.0 for v in vals) else ""
    return f"{star}**{mean(vals):.3f}**"


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(n, lr, s) for n in RULES for lr in GRID[n] for s in SEEDS]
    with ProcessPoolExecutor(max_workers=12, mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(cell, jobs))
    at_end = {(n, lr): [] for n in RULES for lr in GRID[n]}
    at_1000 = {(n, lr): [] for n in RULES for lr in GRID[n]}
    for n, lr, _, a1, a4 in rows:
        at_1000[(n, lr)].append(a1)
        at_end[(n, lr)].append(a4)

    print(
        f"deep tile, 6-bit addition, {DEEP_RELAY.signal} (one vote per case), through recipes.trace"
        f" on DEEP_RELAY._replace(rule, lr), recipe seeds {SEEDS}; hard accuracy at"
        f" {DEEP_RELAY.steps} steps; bold the mean; ★ every seed at 1.000"
        f" ({time.time() - t0:.0f}s)\n"
    )
    vote, step = ("plain", "momentum"), ("sign", "rmsprop", "lion", "adam")
    head = "| rate (per vote) | " + " | ".join(f"{n} ({STATE[n]})" for n in vote)
    head += " | | rate (per step) | " + " | ".join(f"{n} ({STATE[n]})" for n in step) + " |"
    print(head)
    print("|---" * (len(vote) + len(step) + 3) + "|")
    for lv, ls in zip(PER_VOTE, PER_STEP, strict=True):
        row = f"| {lv:g} | " + " | ".join(summary(at_end[(n, lv)]) for n in vote)
        row += f" | | {ls:g} | " + " | ".join(summary(at_end[(n, ls)]) for n in step) + " |"
        print(row)

    print("\n| rule | rate | seed 0 · 1 · 2 at the budget | mean ± sd | at 1000 |")
    print("|---|---|---|---|---|")
    for n in RULES:
        for lr in GRID[n]:
            a, e = at_end[(n, lr)], at_1000[(n, lr)]
            cells = " · ".join(f"{v:.3f}" for v in a)
            early = " ".join(f"{v:.2f}" for v in e)
            print(f"| {n} | {lr:g} | {cells} | **{mean(a):.3f} ± {pstdev(a):.3f}** | {early} |")
