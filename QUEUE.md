# Queue — the next chunks, in order

One live file, rewritten in place. Gabriel pulls from the top; no chunk opens that is not next.
A chunk is at most ~150 hand-written lines plus its test, read in full in one sitting. The table
holds the next three; the progression below it is the plan they are drawn from.

| # | step | chunk | kind | reading | prerequisite | status |
|---|---|---|---|---|---|---|
| 1 | 0 | One tile computes: LUT tables as a pytree, wiring as indices, soft and hard read, truth-table tasks, direct descent as the floor | science | 25 min | none | landed (#2) |
| 2 | 1 | Signals I: the read mode as an axis of the combinatorial test (soft, straight-through on the hard tables); what a local update may read (nothing, task identity, error) | science | 20 min | 1 | landed (#3) |
| 3 | 1 | Signals II and III, one story under the adjoint frame: the relay and the uniform split as two carries of the layered adjoint; the partial to the entry; direct feedback as a second adjoint; the flip credit outside the frame; the ladder scored on a shape where depth is forced; the physical-cost table; the maths in order in `docs/signals.md`; two notes | science | 70 min (oversize, Gabriel's call) | 2 | open (#4) |
| 4 | 2 | Workshop I: the pool of tile states of every age; the smallest rule Δ = −η · e · P · σ′ (the three factors multiplied) with only η meta-learned from a non-functional start, a fixed operand budget; the train/held-out task split; run with the signals a chip would have (the soft relay to the entry; on the bits, the wiring-shaped feedback and the flip credit); controls kept throughout: an output-only baseline, and the feedback mechanism varied separately from any state; η kept nonnegative so the sign-flipped control means what it says; held-out tasks drawn from the junta family, since `addition` ignores its key; the outer objective and its derivative path through the inner update declared before any hard-pass meta-training (`loss(mode="hard")` rounds, it does not straight-through) | science | 30 min | 3 | queued |
| 5 | 3 | The rule: a small shared g over the three factors, replacing the product in the same workshop; value-awareness at the rule level as an ablation: g reads the fine signal alone, then the coarse per-gate signal (`signals.gate_errors`) with its own inputs and output, and must reconstruct the address itself | science | 40 min | 5 | queued |

## The progression the queue is drawn from (2026-09-07)

The three factors of the gradient, the error `e` brought by a transport, the addressed entry
`P(a | u)` and the slope `σ′`, are the learning signal, the eligibility and the post-synaptic factor
of the three-factor rules of the local-learning literature. The rule grows along that reading:

1. **The smallest rule** (chunk 4): the three factors multiplied, η learned. The literature's
   hand-designed three-factor rule, as the inner update of the workshop.
2. **The rule as a function** (chunk 5): a shared `g` over the three factors, free to weight, gate
   or ignore them.
3. **A hidden state per gate** (step 3, continued): `g` gains a carry. With `window = 1` and an
   error that arrives after the addressing, the carry has to bridge the delay to work; nothing is
   added by hand, the regime forces it. The state is called an eligibility trace only once its
   credit-bearing role against the delayed error is shown, not from persistence alone (the corpus
   audit's rule, 2026-09-08). The trace is state, not parameters, which is how one shared rule
   gives every gate its own; that this is compatible is known (Maoutsa; Shervani-Tabar and
   Rosenbaum), so the question here is transfer, not existence. The online axis is the forward
   side of the adjoint duality, its own object with its own costs.
4. **A learned transport** (step 3, continued): the message a gate sends back becomes an output of
   `g`. Relay, uniform and direct feedback become points the rule can find; the alignment finding
   says it should prefer the stable ones.
5. **Sparse error** (steps 2–3): pool rollouts with one late error train the trace and the relay
   half; rollouts with no error supply states and host the rule's autonomous dynamics, but they
   train nothing by themselves: a missing error is not a measured zero, and learning those
   dynamics needs a later objective they connect to, or an explicit auxiliary one (Codex,
   2026-09-08). A sparse scalar reward likewise needs its own estimator; relabelling the residual
   bus does not supply one. A knob on the pool, plus an objective to name.
6. **Damage and heal** (step 4), **evaluation discipline** (step 5), **wiring as configuration**
   (step 6: with direct feedback there is no backward wiring to decide, a bus and a per-gate
   coefficient the rule could own), **the second substrate** (step 7: the soft pass by coin flips,
   a bits fabric with stochastic inputs computes the soft read in expectation and an analogue rate
   is a probability by construction), **the maze** (step 8).

The hypothesis the workshop tests, in the corpus audit's falsifiable form: **one bounded-state
local rule can learn useful computation on fresh sparse hard LUT fabrics, across declared
topology and task shifts, under a specified feedback budget.** Shared plasticity, eligibility,
learned feedback, LUT training and connection selection each have prior art; the combination is
the claim, and it needs the output-only baseline and explicit transfer tests to mean anything.

Before chunk 4 opens: Codex's programme assessment against the bid (vault agent channel, thread
`2026-09-08T105000Z-loom-corpus-audit-review`, reply of 2026-09-08T12:20Z) is the agenda of the
next discussion: the minimum evaluation split inside the first workshop; perturbation and recovery
as soon as a rule learns; a small routing experiment soon after; an old pool state is not a long
meta-gradient; resource accounting from the first experiment; the bid's WP2 transfer test as
written (freeze the shared machinery, refit the state-reading part, compare with full retraining).

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
