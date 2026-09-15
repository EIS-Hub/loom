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
| #5 | 1 | Descent is plain: Δ = −lr·s with nothing normalised; the step 0 and 1 floors re-derived under it (the rate sweep per pass, signal and shape; recipes re-pinned; every claim re-run, what held and what moved on record); `Recipe` becomes `Descent`, the condition named after the loop it runs | record | 20 min | landed |
| #6 | 2 | The rule, one canonical step (`rule.py`: η the only parameter, never negative; one step z − η·s, optionally held within a bound, the logit as a finite counter); `descent` runs it at a fixed rate | foundation | 10 min | landed |
| #7 | 2 | The hand-engineered family at the cell: the primitives (scale, sign, clip, first and second moment, decay, bound) and the named compositions plain · sign · momentum · RMSprop · Lion · Adam, each a few lines verified once against optax; `Descent` gains a `rule`; one claim: at depth plain descent is a point in the rate and the sign readout and RMSprop are bands; the family table with its state and arithmetic columns in `docs/rule.md`: what the learned rule must beat, at equal memory | science | 25 min | open |
| `step2/meta-i` | 2 | Meta-learning I: the two loops (`meta.py`: K steps of the rule in one scan; the objective the soft loss of the tile the rule ends on; the meta-gradient through the K steps; Adam on the host); fresh states; the check from a non-functional start on the flat tile under the soft relay and its three controls (sign-flipped, shuffled, output-only); the η found adapts a held-out task; `Meta` beside `Descent` | science | 30 min (oversize: the docstrings) | built |
| — | 2 | Meta-learning II: the pool of tile states of every age (`pool.py`); the bits on the deep shape, the persisted logit a bounded counter and its depth `clip/η` in votes the first axis (every bits number so far was one vote deep); the ledger's first row; the `path_counts` window fix | science | 25 min | queued |
| — | 2 | The stream: a task whose cases arrive in time order, with drift; the window over time instead of over cases; the learner scores a case before it updates on it; from here every claim carries an online row on a stream, not on a sample of cases | foundation | 10 min | queued |
| — | 3 | The rule as a function: a small shared g over the three factors, replacing the product in the same loops; judged against the family at equal memory; value-awareness at the rule level as an ablation (g on the fine signal, then on the per-gate signal with the gate's own inputs and output) | science | 40 min | queued |

## The progression the queue is drawn from (2026-09-08; the records placed 2026-09-15)

The three factors of the gradient, the error `e` brought by a transport, the addressed entry
`P(a | u)` and the slope `σ′`, are the learning signal, the eligibility and the post-synaptic factor
of the three-factor rules of the local-learning literature. The rule grows along that reading, and
the bid-critical experiments come before the rule's own elaborations:

1. **The smallest rule** (meta-learning I and II): the three factors multiplied, η learned from a
   non-functional start; on the bits the stored logit a bounded counter and η its resolution. The
   counter is the first write policy the fabric has; its depth in votes is not adaptation time,
   since an entry that is rarely addressed waits, so visitation is reported beside the depth
   (synthesis §7.3, §7.4).
2. **The stream** (the online row; one tiny chunk after meta-learning II): a task whose cases arrive
   in time order, with drift, the window over time instead of over cases, and the learner scoring a
   case before it updates on it; the target is the next observation, so no label is needed; a
   controlled synthetic stream first, so what changed is known; the persistent learner beside a
   fresh-start and a frozen comparator. Step 2's own "online window" made real, and the online row
   of every later claim (synthesis §9.2; Codex's recommendation of 2026-09-14). Mechanics, no claim
   of its own.
3. **The rule as a function** (the g chunk): a shared `g` over the three factors, free to weight,
   gate or ignore them. Its first output is decided at a brainstorm before the chunk opens: a write
   gate, when to write, or a weighting of the three factors (synthesis §7.4; the 2026-09-14
   journal's "when to update?"). If the gate: its first test is selective adaptation on the stream,
   benign drift and a sustained event as separate conditions, a frozen predictor, continuous
   adaptation, a fixed gate and the learned gate compared on prediction quality, adaptation time,
   writes, and whether adapting suppresses the event's own signal, every method given the same
   observable clues and the event's identity kept for evaluation. On a synthetic stream the contrast
   is designed, so this is a mechanism test of g, never an application claim.
4. **Damage and heal** (step 4): gates knocked out as a perturbation of the pool's states; the rule
   heals under immediate feedback; the basin measure over degenerate solutions. As soon as a rule
   learns and before it grows further (the programme assessment of 2026-09-08). The measure counts
   paths, not endpoints: the probability of reaching a working configuration after a named
   perturbation, the writes and the transient loss along the way, and whether the recovery used
   other resources (synthesis §7.2).
5. **The evaluation discipline** (step 5): {train, held-out} wiring × task, the input-case split,
   the memorisation gap, paired seeds, the window as a visible axis. The held-out task keys are
   fixed at meta-learning I and never looked at while `g` is designed. The lifetime evaluation lives
   here: one persistent configuration and adaptation state through a sequence of tasks, damage
   introduced later as its own condition, the shared rule frozen, matched-difficulty held-out tasks,
   fresh-start and frozen-configuration comparators, a global reset a comparator and never the
   default; error after a switch, updates to regain a declared accuracy, loss accumulated in
   recovery and failures within budget, all against the machine's age; sustained service, not peak
   recovery. Continual here means the preserved ability to adapt; forgetting is allowed (PC review,
   reply of 2026-09-14; synthesis §7.3, §8).
6. **Wiring as configuration** (step 6): the first routing experiment, a fabric with one necessary
   route severed, where a cut-off gate receives no task error over the broken route (the
   assessment's counter-question to WP1's "healing repairs the computation and its error route at
   once"); with direct feedback there is no backward wiring to decide. Three feedback conditions on
   the severed route: task feedback over the surviving data routes only, physical-neighbour
   messages, a broadcast reference; recovery probability, messages, writes and interruption scored
   (synthesis §7.1).
7. **A hidden state per gate**: `g` gains a carry; with `window = 1` and an error that arrives
   after the addressing the carry has to bridge the delay. The state is called an eligibility trace
   only once its credit-bearing role against a delayed error is shown (resetting or scrambling it
   must selectively impair the delayed task); the compatibility of one shared rule with a state
   per gate is known (Maoutsa; Shervani-Tabar and Rosenbaum), so the question is transfer. The
   online axis is the forward side of the adjoint duality, its own object with its own costs.
   Isolated from the experiments above, and after them.
8. **A learned transport**: the message a gate sends back becomes an output of `g`. Relay, uniform
   and direct feedback become points the rule can find; the alignment finding says it should
   prefer the stable ones.
9. **Sparse error**: pool rollouts with one late error train the trace and the relay half;
   rollouts with no error supply states and host the rule's autonomous dynamics, but they train
   nothing by themselves: a missing error is not a measured zero, and learning those dynamics
   needs a later objective they connect to, or an explicit auxiliary one. A sparse scalar reward
   likewise needs its own estimator.
10. **The second substrate** (step 7: the soft pass by coin flips, a bits fabric with stochastic
    inputs computes the soft read in expectation), **the maze** (step 8). With a cost-matched
    hand-designed baseline, and the share of the adaptation the encoder learns reported, since a
    strong encoder hides the algorithm (synthesis §7.6).
11. **Beyond the tile, on the record and on no row.** Predictive coding as a bounded comparison:
    strict predictive coding with movable beliefs against the relay on the same tasks, with state,
    messages, rounds and writes counted and a reconverging case included; the fixed-table query
    experiment, a half-adder run forward, inverse and partial, and a small redundant code for
    recovery from an initial state; opened once the tile is owned, and after a brainstorm of its own
    on whether it enters the frame as a `via` or stands outside it, which is open, not obvious (PC
    review; synthesis §7.5; the design note only through the PC review's five qualifications).
    Applications: industrial vibration first, wearable pulse second, a frozen backbone with an
    online tail as the pragmatic baseline the fabric must beat (synthesis §§5–6). Recovery when the
    rule itself is damaged (synthesis §7.7). Two repo-side observations from the records, acted on
    when their item opens: the bounded logit is a float, not a counter (item 1); `descend(window=1)`
    samples cases, not a stream (item 2).

**Brainstorms owed**, one session each with the vault's pages open, before the chunk each gates:
predictive coding in or outside the frame (gates item 11); g's first output, gate or weighting, and
what the stream lets the rule read (gates item 3); the lifetime evaluation's design (gates item 5).

**Records cited above** (vault threads, by section or by reply date): the PC review
`2026-09-11T084903Z-predictive-coding-review` (Codex; its reply of 2026-09-14T08:26Z); the synthesis
`2026-09-14T093529Z-scm-research-and-applications-synthesis` (Codex, 2026-09-14); the design note
`2026-09-11T110500Z-pc-plasticity-design-note` (a blind web session on the old SODC paper, read only
through the PC review's qualifications); the corpus audit
`2026-09-08T105000Z-loom-corpus-audit-review`, folded in on 2026-09-08. The vault distils them on
`wiki/concepts/predictive-coding-local-credit` and `wiki/concepts/unlabelled-online-adaptation`.

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
unmasked random bus trains worst (`findings/2026-09-07-the-relay-is-autodiff-and-the-bits-learn-by-alignment/note.md`,
`findings/2026-09-07-direct-feedback-and-the-flip-credit/note.md`). Decided the same day: Signals III folds
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

Decided 2026-09-15 (Gabriel, after the weekly of 2026-09-14 and Codex's recommendation of the same
day). The channel's records since the corpus audit, the PC review, the design note and the
synthesis, had landed nothing here; they are now **placed** in the progression above, each item
citing its record by section, so a proposal carries the step at which it becomes live. **The map's
order stands**, and the stop rule with it: the stream enters as one tiny chunk after meta-learning
II, because the online window is step 2's own definition and a stream is a task object plus a
protocol, not a step; **selective adaptation is a step 3 question**, since a learned write gate is
an expansion of g, so gate versus weighting is decided at the g brainstorm and not before it, and
damage and heal keep the slot given on 2026-09-08; **predictive coding is a bounded comparison**
after the tile is owned, with a brainstorm before any cell, because whether it is a `via` of the
frame or an object outside it is open. Codex's alternative, the stream, then selective adaptation as
the central question before g and before damage, then the route, was read and declined on those
grounds; what the order dropped is kept in item 11.
