# Queue — the next chunks, in order

One live file, rewritten in place. Gabriel pulls from the top; no chunk opens that is not next.
A chunk is at most ~150 hand-written lines plus its test, read in full in one sitting. The table
holds the next three; the progression below it is the plan they are drawn from.

| # | step | chunk | kind | reading | prerequisite | status |
|---|---|---|---|---|---|---|
| 1 | 0 | One tile computes: LUT tables as a pytree, wiring as indices, soft and hard read, truth-table tasks, direct descent as the floor | science | 25 min | none | landed (#2) |
| 2 | 1 | Signals I: the read mode as an axis of the combinatorial test (soft, straight-through on the hard tables); what a local update may read (nothing, task identity, error) | science | 20 min | 1 | landed (#3) |
| 3 | 1 | Signals II and III, one story under the adjoint frame: the relay and the uniform split as two carries of the layered adjoint; the partial to the entry; direct feedback as a second adjoint; the flip credit outside the frame; the ladder scored on a shape where depth is forced; the physical-cost table; the maths in order in `docs/signals.md`; two notes | science | 70 min (oversize, Gabriel's call) | 2 | open (#4) |
| 4 | 2 | Workshop I: the pool of tile states of every age; the smallest rule Δ = −η · e · P · σ′ (the three factors multiplied) with only η meta-learned from a non-functional start; the train/held-out task split; run with the signals a chip would have (the soft relay without σ′; on the bits, whatever chunk 4 finds trains) | science | 30 min | 3 | queued |
| 5 | 3 | The rule: a small shared g over the three factors, replacing the product in the same workshop; value-awareness at the rule level as an ablation: g reads the signal alone, then its own inputs and output too | science | 40 min | 5 | queued |

## The progression the queue is drawn from (2026-09-07)

The three factors of the gradient, the error `e` brought by a transport, the addressed entry
`P(a | u)` and the slope `σ′`, are the learning signal, the eligibility and the post-synaptic factor
of the three-factor rules of the local-learning literature. The rule grows along that reading:

1. **The smallest rule** (chunk 4): the three factors multiplied, η learned. The literature's
   hand-designed three-factor rule, as the inner update of the workshop.
2. **The rule as a function** (chunk 5): a shared `g` over the three factors, free to weight, gate
   or ignore them.
3. **A hidden state per gate** (step 3, continued): `g` gains a carry. With `window = 1` and an
   error that arrives after the addressing, the carry has to become an eligibility trace to work;
   nothing is added by hand, the regime forces it. The trace is state, not parameters, which is how
   one shared rule gives every gate its own trace.
4. **A learned transport** (step 3, continued): the message a gate sends back becomes an output of
   `g`. Relay, uniform and direct feedback become points the rule can find; the alignment finding
   says it should prefer the stable ones.
5. **Sparse error** (steps 2–3): pool rollouts with no error at all train the unsupervised half
   (homeostasis, self-organisation); rollouts with one late error train the trace and the relay
   half; the residual bus of direct feedback carries a sparse reward. A knob on the pool, not new
   machinery.
6. **Damage and heal** (step 4), **evaluation discipline** (step 5), **wiring as configuration**
   (step 6: with direct feedback there is no backward wiring to decide, a bus and a per-gate
   coefficient the rule could own), **the second substrate** (step 7: the soft pass by coin flips,
   a bits fabric with stochastic inputs computes the soft read in expectation and an analogue rate
   is a probability by construction), **the maze** (step 8).

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
