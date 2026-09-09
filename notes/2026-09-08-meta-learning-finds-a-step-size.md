# 2026-09-08 — Meta-learning finds a step size: the outer loop from a non-functional start, under the true signal and its controls

*From the step-2 chunk 4a (loom PR #TBD) and the probes of the same date, on the flat tile, before
the pool. The path is recorded: the objective chosen and the one not chosen, the parameterisation,
what the controls leak, and where the window moves the step size.*

**Question.** With the rule Δ = −η·s and only η learned, does the outer loop, differentiating the
loss of the tile the rule ends on through K steps of the rule, find a working step size from a
start where nothing moves; does it fail to under a sign-flipped signal; does nothing train under
a shuffled or an output-only signal; and does the η found adapt a fresh tile on a task the loop
never saw? Underneath: what does the η found *mean*?

**Floor and shape.** The flat tile `(4, 16, 8, 2)` at arity 4 on 2-juntas (4 → 2), the smoke shape of
steps 0 and 1; the signal `soft.relay.entry`, the exact adjoint to the entry, the one a soft fabric
could carry. Plain descent is the floor since the record chunk of the same date
(`notes/2026-09-08-the-floors-under-plain-descent.md`): the rule at a fixed η is that floor.

**Conditions.** `SOFT_META`: K = 16 steps of the rule per rollout, every case per step, a batch of
16 fresh tiles on 16 fresh tasks per outer step, 200 outer steps of Adam at 0.1 on log η from
η₀ = 0.01, the gradient's global norm clipped at 1. The claims swap the recipe's `control` and
nothing else. The probes also visited windows W = 1 and W = 4 and rollouts of 32 and 64 steps.
Every run on the CPU.

**The objective, and the one not chosen.** J = the soft loss on every case of the tile after K
steps. The alternative, the mean of the online losses along the rollout (each step's window
scored before its update), weights the steps: an early update is credited through every later
loss, the last through none. `probes/2026-09-08-meta-objective-landscape.py` measured both across
η (cells: final loss / online mean, three probe seeds, **bold** the mean over seeds):

| η | W = 1, K = 64 | W = 4, K = 32 | W = all, K = 16 |
|---|---|---|---|
| 1 | 0.1258/0.1263 · 0.1260/0.1239 · 0.1236/0.1262 (**0.1252 / 0.1255**) | 0.1263/0.1269 · 0.1260/0.1240 · 0.1244/0.1260 (**0.1256 / 0.1256**) | 0.1267/0.1270 · 0.1259/0.1261 · 0.1249/0.1252 (**0.1259 / 0.1261**) |
| 3 | 0.1246/0.1263 · 0.1250/0.1247 · 0.1203/0.1258 (**0.1233 / 0.1256**) | 0.1250/0.1265 · 0.1257/0.1240 · 0.1225/0.1253 (**0.1244 / 0.1253**) | 0.1259/0.1266 · 0.1254/0.1258 · 0.1240/0.1248 (**0.1251 / 0.1257**) |
| 10 | 0.0763/0.1171 · 0.0651/0.1155 · 0.0773/0.1194 (**0.0729 / 0.1173**) | 0.1208/0.1257 · 0.1221/0.1236 · 0.1114/0.1223 (**0.1181 / 0.1239**) | 0.1242/0.1255 · 0.1240/0.1251 · 0.1204/0.1232 (**0.1228 / 0.1246**) |
| 30 | 0.0067/0.0822 · 0.0005/0.0500 · 0.0014/0.0698 (**0.0029 / 0.0673**) | 0.0365/0.1049 · 0.0058/0.0838 · 0.0169/0.0881 (**0.0197 / 0.0923**) | 0.1154/0.1229 · 0.1079/0.1219 · 0.0877/0.1148 (**0.1037 / 0.1199**) |
| 100 | 0.0066/0.0765 · 0.0000/0.0421 · 0.0000/0.0503 (**0.0022 / 0.0563**) | 0.0003/0.0602 · 0.0002/0.0371 · 0.0005/0.0435 (**0.0003 / 0.0469**) | 0.0014/0.0728 · 0.0009/0.0605 · 0.0023/0.0570 (**0.0015 / 0.0634**) |
| 300 | 0.1773/0.2179 · 0.0509/0.1162 · 0.0000/0.0733 (**0.0761 / 0.1358**) | 0.0001/0.0968 · 0.0000/0.0426 · 0.0000/0.0538 (**0.0001 / 0.0644**) | 0.0001/0.0357 · 0.0001/0.0299 · 0.0002/0.0268 (**0.0001 / 0.0308**) |
| 1000 | 0.1425/0.2058 · 0.0000/0.0592 · 0.0369/0.1062 (**0.0598 / 0.1237**) | 0.2526/0.1980 · 0.0000/0.0550 · 0.1636/0.1518 (**0.1387 / 0.1350**) | 0.0000/0.0942 · 0.0000/0.0338 · 0.0000/0.0250 (**0.0000 / 0.0510**) |
| 3000 | 0.1793/0.1663 · 0.2274/0.2120 · 0.1460/0.1538 (**0.1842 / 0.1774**) | 0.1495/0.1910 · 0.0000/0.0384 · 0.1719/0.1815 (**0.1071 / 0.1369**) | 0.0000/0.0652 · 0.0000/0.0399 · 0.0000/0.0339 (**0.0000 / 0.0463**) |

Two things are in the table. The step size the objective wants depends on the window: fully
online (W = 1) the final loss has an interior optimum at η ≈ 30–100 and degrades beyond, at W = 4
it sits at 100–300, and with every case per step it has none inside this grid, the loss reaching
zero from η ≈ 300 up on this benign shape (an exact signal, a tile that represents every
function of four inputs). And the two objectives disagree where they should: with every case per
step the online mean turns up again past η ≈ 300 (0.031 → 0.051 → 0.046) while the final loss stays
at zero, the mean punishing the early overshoot of a large step that the final tile has long
recovered from. The final loss asks where the rule *arrives*; the mean asks how it *travels*.

**The derivative path, measured** (`probes/2026-09-08-meta-gradient-paths.py`, K = 8; cells: the
full meta-gradient / the first-order one, the signal as data / a central finite difference with
h = 0.5; three probe seeds):

| signal | η | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| soft.relay.entry | 3 | −2.22e−4 / −2.34e−4 / −2.22e−4 | −1.43e−4 / −1.50e−4 / −1.43e−4 | −2.39e−4 / −2.42e−4 / −2.39e−4 |
| soft.relay.entry | 30 | −1.08e−4 / −9.65e−5 / −1.08e−4 | −1.33e−4 / −1.03e−4 / −1.33e−4 | −4.19e−4 / −3.18e−4 / −4.19e−4 |
| soft.relay.entry | 300 | −4.67e−6 / −9.81e−6 / −4.67e−6 | −2.69e−6 / −7.68e−6 / −2.69e−6 | −7.10e−6 / −1.62e−5 / −7.10e−6 |
| hard.uniform.entry | 3 | −1.98e−3 / −1.98e−3 / −8.76e−4 | −3.05e−3 / −3.05e−3 / −2.64e−3 | −2.76e−3 / −2.76e−3 / −4.44e−3 |
| hard.uniform.entry | 30 | −9.98e−4 / −9.98e−4 / −9.94e−4 | −1.44e−3 / −1.44e−3 / −3.91e−2 | −6.72e−4 / −6.72e−4 / −6.72e−4 |
| hard.uniform.entry | 300 | −8.66e−8 / −8.66e−8 / −8.57e−8 | −9.60e−9 / −9.60e−9 / −2.24e−8 | −7.07e−7 / −7.07e−7 / −7.08e−7 |

On the soft pass the full meta-gradient is the finite difference to three digits, and the
first-order path is not the whole of it: the term through the signal is a few percent at η = 3, a
quarter at η = 30 and more than the first-order term itself at η = 300, growing with the step
because the signal changes more along a longer trajectory. On the bits the full and the
first-order meta-gradient are the same number to the last digit, as the declaration in
`docs/meta.md` says they must be: nothing differentiates through rounded tables. There the finite
difference disagrees on three cells of nine: the objective on the bits is piecewise smooth in η,
and a difference of ±0.5 in η that crosses a flip inside the rollout measures a jump, not a slope.
The mechanics test uses the soft relay for that reason.

**The controls** (`probes/2026-09-08-controls-trajectories.py`; batch 16, K = 16, every case per
step, 200 outer steps from η₀ = 0.01; cells: η / final loss at outer steps 50, 100, 200; then the
η the run ends on, driven by the *true* signal for 500 plain steps on a fresh held-out 2-junta,
hard accuracy; three probe seeds):

| control | seed 0 | seed 1 | seed 2 | held-out at the η found |
|---|---|---|---|---|
| none (the true signal) | 5.06/0.1245 · 536/0.0000 · 287/0.0001 | 5.04/0.1249 · 265/0.0001 · 197/0.0004 | 5.11/0.1247 · 517/0.0000 · 318/0.0001 | 1.000 · 1.000 · 1.000 |
| flipped | 5.6e−4/0.1268 · 2.6e−4/0.1263 · 1.1e−4/0.1256 | 5.3e−4/0.1269 · 2.4e−4/0.1261 · 1.0e−4/0.1264 | 5.4e−4/0.1267 · 2.4e−4/0.1263 · 1.0e−4/0.1259 | 0.469 · 0.438 · 0.344 |
| shuffled | 4.02/0.1265 · 59.7/0.1254 · 65.6/0.1249 | 3.66/0.1265 · 211/0.1251 · 109/0.1252 | 4.05/0.1264 · 135/0.1257 · 36.8/0.1256 | 1.000 · 1.000 · 1.000 |
| output-only | 4.85/0.1255 · 266/0.1168 · 310/0.1162 | 4.91/0.1257 · 259/0.1158 · 293/0.1151 | 4.92/0.1255 · 296/0.1169 · 270/0.1159 | 1.000 · 1.000 · 1.000 |

Under the true signal η rises by a constant factor per outer step (5 at step 50 means nine
doublings from 0.01 in fifty steps), overshoots into the region where the final loss is already
zero, and settles back to a few hundred, where some batch members begin to overshoot and pull it
down: the loss is at its floor from step 50 on, and the η found adapts a fresh tile on a task the
loop never saw to 1.000 in 500 plain steps, on every seed. Under the flipped signal η can only
shrink, and does, geometrically, to a hundredth of its start; nothing trains and the tile stays
at chance. Under the shuffled signal the loop finds an η of the same order as the true signal's
(37 to 109) and the loss does not move: the numbers carry no information about which entry they
reach. Under output-only the output gates learn what they can on their own residual, a drop of a
tenth in the loss and no more. The last column for the two information controls says only that
the η they wandered to is a working step size *for the true signal*, which every η between ten
and a thousand is on this tile; it is not a result about the controls.

**Path.**

1. *The objective.* The first design carried the mean of the online losses, on the reading that a
   deployed tile experiences the stream. Gabriel asked whether back-propagation through the K
   steps did not make the final loss enough, and whether the mean did not weight the steps. It
   does (above): the final loss became the objective, the online losses a record.
2. *The parameterisation.* η = exp(raw) rather than softplus: at a non-functional start the two
   agree, but a constant factor per Adam step reaches a working scale in log(scale)/lr steps, and
   the flipped control then drives raw to −∞ geometrically, which the table shows (η falls by a
   factor 2.3 every fifty steps). Adam on the host is kept for the same reason: the gradient on
   raw is η times the gradient on η and vanishes exactly where the check starts.
3. *What the shuffled control leaks.* A permutation within a layer keeps the layer's mean, and the
   mean is a real component of the gradient, so the loop's η under it is not zero and not
   meaningful; the claim is therefore worded on the loss, never on η. The sign flip is the only
   control whose η is a result.
4. *The held-out set.* Tasks are draws from a key; the seed's key is split once into a training
   half and a held-out half, and `adapt` draws from the second. Fixed here, never looked at while
   step 3's rule is designed.
5. *The second-order term.* Read as negligible on the soft pass before it was measured; the
   derivative-path table says it grows with η and exceeds the first-order term at η = 300. It is
   kept in the objective on the soft pass (the rollout differentiates through the signal unless
   told not to); the bits never have it.

**Measured and read.** Measured: the meta-gradient is the finite difference on the soft pass and
identical to its first-order term on the bits; from a non-functional start the loop finds a step
size under the true signal on every seed, fails to under the sign flip, and leaves the loss
untrained under the shuffled and output-only signals; the η found adapts a held-out task; the step
size the objective wants depends on the window. Read, not tested: that at first order the check
measures the cosine of the accumulated steps with the soft gradient at the end, which is why a
sign-consistent signal is found and a shuffled one is not, and that on this benign tile the
final-tile objective has no interior optimum in η because the exact soft signal cannot overshoot
what a sixteen-entry table can absorb. What η means on the bits, where the design pass saw the
final loss go flat above a threshold and the loop walk η into the thousands, is the next chunk's
question, with the stored logit bounded.

**Claims left behind.** `claims/test_2026_09_08_meta_learning_finds_a_step_size.py`, under
`SOFT_META` and its `control` swaps, three recipe seeds: the loop finds a step size that adapts a
held-out task to 1.000 in 500 plain steps; under the sign flip η ends below its start and the loss
stays untrained; under the shuffled and output-only signals the loss stays closer to untrained
than to trained.

**What would change it.** A shape where the exact soft signal can overshoot, so that the
final-tile objective has an interior optimum at every window (the deep shape, next chunk, records
this on the bits). A pool of states of every age, which asks the same loop for an η that also
holds what it built. The rule as a function of the three factors (step 3), where the outer loop
has more than one number to move and Adam's normalisation is no longer a footnote.
