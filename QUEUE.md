# Queue — the next chunks, in order

One live file, rewritten in place. Gabriel pulls from the top; no chunk opens that is not next.
A chunk is at most ~150 hand-written lines plus its test, read in full in one sitting.

| # | step | chunk | kind | reading | prerequisite | status |
|---|---|---|---|---|---|---|
| 1 | 0 | One tile computes: LUT tables as a pytree, wiring as indices, soft and hard read, truth-table tasks, direct descent as the floor | science | 25 min | none | landed (#2) |
| 2 | 1 | Signals I: the read mode as an axis of the combinatorial test (soft, straight-through on the hard tables); what a local update may read (nothing, task identity, error) | science | 20 min | 1 | landed (#3) |
| 3 | 1 | Signals II: the relay and the uniform transports as `via` coordinates, dropping the surrogate; the ladder scored against the reference (sparsity, cosine, sign agreement) on a shape where depth is forced; `docs/signals.md` extended with the transports; the second note | science | 40 min | 2 | open (#4) |
| 4 | 2 | Workshop I: the pool of tile states of every age; the smallest rule Δ = −η·signal with η meta-learned from a non-functional start; the train/held-out task split | science | 30 min | 3 | queued |
| 5 | 1 | Signals III: a signal that credits a flip exactly on the bits (the signed error and the reach of a flip as two relayed channels), and the leaky carry between relay and uniform; scored where chunk 3 found the bits fail | science | 30 min | 3 | queued, order to decide |

Decided in review of step 0 (2026-09-03): step 2's inner update is the smallest rule, Δ = −η·signal, with only η meta-learned from a non-functional start, the tables never (the pool holds tile states of every age); the check ties step 1 to step 2: a working η is recovered under a sign-consistent signal and not under a sign-flipped one; step 3 grows it into g(logit, signal). The read mode (soft, straight-through) is an axis from step 1; softjax enters at step 6 with selection. `fit` already takes a `window`: the online regime is a stream of case windows, W = all is the batched floor, W = 1 fully online. Tasks gain an output mask, not zero padding, when several share a fabric.

Found in chunk 3 (2026-09-07): on the bits at depth, the exact relay (straight-through with or
without σ′) does not train and the blind uniform split does, by feedback alignment
(`notes/2026-09-07-the-relay-is-autodiff-and-the-bits-learn-by-alignment.md`). Open for Gabriel:
whether Signals III (a flip-exact signal) goes before Workshop I, and which signals the workshop's
Δ = −η·signal is first run with (the soft relay without σ′, the blind split on the bits).
