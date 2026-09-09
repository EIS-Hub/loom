# Queue — the next chunks, in order

One live file, rewritten in place. A chunk is a pull request: one row each, keyed by its PR number,
or by its branch until it opens; the order is the prerequisite. Gabriel pulls from the top; no chunk
opens before the one above it lands. A chunk is at most ~150 hand-written lines plus its test, read in
full in one sitting. The progression below the table is the plan the rows are drawn from.

| PR | step | chunk | kind | reading | status |
|---|---|---|---|---|---|
| #2 | 0 | One tile computes: LUT tables as a pytree, wiring as indices, soft and hard read, truth-table tasks, direct descent as the floor | science | 25 min | landed |
| #3 | 1 | Signals I: the read mode as an axis of the combinatorial test (soft, straight-through on the hard tables); what a local update may read (nothing, task identity, error) | science | 20 min | landed |
| #4 | 1 | Signals II and III, one story under the adjoint frame: the relay and the uniform split as two carries of the layered adjoint; the partial to the entry; direct feedback as a second adjoint; the flip credit outside the frame; the ladder scored on a shape where depth is forced; the physical-cost table; the maths in order in `docs/signals.md`; two notes | science | 70 min (oversize, Gabriel's call) | landed |
| #5 | 1 | Descent is plain: Δ = −lr·s with nothing normalised; the step 0 and 1 floors re-derived under it (the rate sweep per pass, signal and shape; recipes re-pinned; every claim re-run, what held and what moved on record); `Recipe` becomes `Descent`, the condition named after the loop it runs | record | 20 min | open |
| `step2/rule` | 2 | The rule, one canonical step (`rule.py`: η the only parameter, never negative; one step z − η·s, optionally held within a bound, the logit as a finite counter); `descent` runs it at a fixed rate | foundation | 10 min | built |
| `step2/family` | 2 | The hand-engineered family at the cell: the primitives (scale, sign, clip, first and second moment, decay, bound) and the named compositions plain · sign · momentum · RMSprop · Lion · Adam, each a few lines verified once against optax; `Descent` gains a `rule`; one claim: at depth plain descent is a point in the rate and the sign readout and RMSprop are bands; the family table with its state and arithmetic columns in `docs/rule.md`: what the learned rule must beat, at equal memory | science | 25 min | built |
| `step2/meta-i` | 2 | Meta-learning I: the two loops (`meta.py`: K steps of the rule in one scan; the objective the soft loss of the tile the rule ends on; the meta-gradient through the K steps; Adam on the host); fresh states; the check from a non-functional start on the flat tile under the soft relay and its three controls (sign-flipped, shuffled, output-only); the η found adapts a held-out task; `Meta` beside `Descent` | science | 30 min (oversize: the docstrings) | built |
| — | 2 | Meta-learning II: the pool of tile states of every age (`pool.py`); the bits on the deep shape, the persisted logit a bounded counter and its depth `clip/η` in votes the first axis (every bits number so far was one vote deep); the ledger's first row; the `path_counts` window fix | science | 25 min | queued |
| — | 3 | The rule as a function: a small shared g over the three factors, replacing the product in the same loops; judged against the family at equal memory; value-awareness at the rule level as an ablation (g on the fine signal, then on the per-gate signal with the gate's own inputs and output) | science | 40 min | queued |

## The progression the queue is drawn from (2026-09-08)

The three factors of the gradient, the error `e` brought by a transport, the addressed entry
`P(a | u)` and the slope `σ′`, are the learning signal, the eligibility and the post-synaptic factor
of the three-factor rules of the local-learning literature. The rule grows along that reading, and
the bid-critical experiments come before the rule's own elaborations:

1. **The smallest rule** (meta-learning I and II): the three factors multiplied, η learned from a
   non-functional start; on the bits the stored logit a bounded counter and η its resolution.
2. **The rule as a function** (the g chunk): a shared `g` over the three factors, free to weight, gate
   or ignore them.
3. **Damage and heal** (step 4): gates knocked out as a perturbation of the pool's states; the rule
   heals under immediate feedback; the basin measure over degenerate solutions. As soon as a rule
   learns and before it grows further (the programme assessment of 2026-09-08).
4. **The evaluation discipline** (step 5): {train, held-out} wiring × task, the input-case split,
   the memorisation gap, paired seeds, the window as a visible axis. The held-out task keys are
   fixed at meta-learning I and never looked at while `g` is designed.
5. **Wiring as configuration** (step 6): the first routing experiment, a fabric with one necessary
   route severed, where a cut-off gate receives no task error over the broken route (the
   assessment's counter-question to WP1's "healing repairs the computation and its error route at
   once"); with direct feedback there is no backward wiring to decide.
6. **A hidden state per gate**: `g` gains a carry; with `window = 1` and an error that arrives
   after the addressing the carry has to bridge the delay. The state is called an eligibility trace
   only once its credit-bearing role against a delayed error is shown (resetting or scrambling it
   must selectively impair the delayed task); the compatibility of one shared rule with a state
   per gate is known (Maoutsa; Shervani-Tabar and Rosenbaum), so the question is transfer. The
   online axis is the forward side of the adjoint duality, its own object with its own costs.
   Isolated from the experiments above, and after them.
7. **A learned transport**: the message a gate sends back becomes an output of `g`. Relay, uniform
   and direct feedback become points the rule can find; the alignment finding says it should
   prefer the stable ones.
8. **Sparse error**: pool rollouts with one late error train the trace and the relay half;
   rollouts with no error supply states and host the rule's autonomous dynamics, but they train
   nothing by themselves: a missing error is not a measured zero, and learning those dynamics
   needs a later objective they connect to, or an explicit auxiliary one. A sparse scalar reward
   likewise needs its own estimator.
9. **The second substrate** (step 7: the soft pass by coin flips, a bits fabric with stochastic
   inputs computes the soft read in expectation), **the maze** (step 8).

## Decisions of record

Decided in review of step 0 (2026-09-03): step 2's inner update is the smallest rule, with only η
meta-learned from a non-functional start, the tables never (the pool holds tile states of every
age); the check ties step 1 to step 2: a working η is recovered under a sign-consistent signal and
not under a sign-flipped one; step 3 grows it into g. The read mode (soft, straight-through) is an
axis from step 1; softjax enters at step 6 with selection. `fit` already takes a `window`: the
online regime is a stream of case windows, W = all is the batched floor, W = 1 fully online. Tasks
gain an output mask, not zero padding, when several share a fabric.

Found in chunk 3 (2026-09-07, audited 2026-09-08): on the bits past one hidden layer the exact
relay (straight-through, to the logit or to the entry) does not train; any fixed feedback whose
support is the wiring's reachability does, whatever its signs (the wiring-shaped split, or a random
bus masked to reachable outputs); the flip credit does, to a fixed point of single flips; an
unmasked random bus trains worst (`notes/2026-09-07-the-relay-is-autodiff-and-the-bits-learn-by-alignment.md`,
`notes/2026-09-07-direct-feedback-and-the-flip-credit.md`). Decided the same day: Signals III folds
into chunk 3 under the adjoint frame, so the workshop is trained on the signals a chip would have.

Decided 2026-09-08, before meta-learning opens (the discussion the corpus audit's programme
assessment asked for). **Plain descent is the floor**: `descent` is Δ = −lr·s with nothing
normalised, and the step 0 and 1 floors are re-derived under it (chunk 4), because Adam normalised
away a signal's magnitude and σ′, carried two moments per logit that the cost table excludes, and
at W = 1 on the bits amplifies rarely addressed entries; Adam never enters a signal claim again,
and the outer loop keeps it on the host. **The outer objective** is the soft loss of the tile the
rule ends on after K steps, every update credited equally; the mean of the online losses is a
record, not the objective (it weights early updates), and the loss on the bits has no derivative.
**Step 2 is two chunks**: fresh states and the η check on the flat tile, then the pool and the bits
on the deep shape, where the persisted logit is a **bounded counter** and the claims run at the
batched window with W = 1 recorded. **The queue is re-ordered**: damage and heal, the evaluation
split and the first routing experiment come before the per-gate carry and the learned transport.
**Names**: the step is meta-learning, not "the training workshop"; the modules are `rule`, `meta`,
`pool`; the recipes `Descent` and `Meta`, each named after the loop it runs. **The ledger** (WP1)
starts with the first meta-learning experiment: parameters, state per entry, reads per case,
wiring for the error, writes per step. The assessment (vault thread
`2026-09-08T105000Z-loom-corpus-audit-review`, reply of 12:20Z) was the agenda; what it names for
later steps (resource accounting beyond the ledger; the WP2 transfer test as written, freeze the
shared machinery and refit the state-reading part) enters at the step it names.

Decided 2026-09-09 (Gabriel): the **seed is one vote per case** (the residual over the window's
cases, summed over the outputs a gate reaches); every rate is shown with the step it implies per
layer; **plain descent stays the floor** of every signal claim and its point at depth is recorded,
not repaired; the depth attenuation is a magnitude problem solved by a readout at the cell (sign,
no state; RMSprop, one accumulator), so the adjoint frame stays as landed and no wire-side carry
scale is built; **the hand-engineered family** (Adam included) is what the learned rule must beat,
at equal memory, and enters as its own chunk before meta-learning; the meta-learning chunk is split into the rule
(one canonical step) and meta-learning I.
