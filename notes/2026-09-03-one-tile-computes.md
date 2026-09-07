# 2026-09-03 — One tile computes

*Written 2026-09-07 as the first note, from the step-0 chunk (loom PR #2) and its review.*

**Question.** Can one tile represent a target function and reach it by direct descent, with the
hard read agreeing with the soft read once trained; and can it do so online, one case per step?

**Floor.** The untrained tile: random tables over a random wiring, reading at chance on the
demanded bits.

**Conditions.** `SOFT_FLOOR` (the reference gradient on the soft pass, Adam at 0.1 for 500
steps, every case per step) and `ONLINE_FLOOR` (the same signal, Adam at 0.05 for 3000 steps, one
case per step). Shape `(4, 16, 8, 2)` at arity 4, twenty-six gates, inherited from blastema's
first rung rather than derived (see `docs/tile.md`: a single layer of two arity-4 gates already
represents every function of four inputs).

**Measured.** Under `SOFT_FLOOR`, hard accuracy 1.000 on random 2-juntas for three seeds, with soft
accuracy 1.000 on the same tiles (no deploy gap once trained), and 1.000 on two-bit addition with
carry. Under `ONLINE_FLOOR`, hard accuracy 1.000 on 2-juntas for three seeds: descent adapts the
tables seeing one case at a time.

**Claims left behind.** `claims/test_2026_09_03_one_tile_computes.py`: the junta and the addition
under the soft floor with no deploy gap; the junta online.

**What would change it.** A shape where depth is forced (arity below the input width), where a
gate cannot see all inputs and carry chains exist; larger tasks; a window between one and all.
