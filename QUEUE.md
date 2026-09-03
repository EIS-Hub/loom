# Queue — the next chunks, in order

One live file, rewritten in place. Gabriel pulls from the top; no chunk opens that is not next.
A chunk is at most ~150 hand-written lines plus its test, read in full in one sitting.

| # | step | chunk | kind | reading | prerequisite | status |
|---|---|---|---|---|---|---|
| 1 | 0 | One tile computes: LUT tables as a pytree, wiring as indices, soft and hard read, truth-table tasks, direct descent as the floor | science | 25 min | none | in review |
| 2 | 1 | Signals I: the read mode as an axis of the combinatorial test (soft, straight-through on the hard tables); what a local update may read (nothing, task identity, error) | science | 30 min | 1 | queued |
| 3 | 1 | Signals II: how error travels, the uniform adjoint and the relay through each gate's own table; the check that the relay keeps the gradient's sign where the uniform adjoint loses it | science | 30 min | 2 | queued |

Decided in review of step 0 (2026-09-03): step 2's inner update is the smallest rule, Δ = −η·signal, with η and the initial tables meta-learned (a sanity check that the workshop re-learns descent under the right signal); step 3 grows it into g(logit, signal). The read mode (soft, straight-through) is an axis from step 1; softjax enters at step 6 with selection. `fit` already takes a `window`: the online regime is a stream of case windows, W = all is the batched floor, W = 1 fully online. Tasks gain an output mask, not zero padding, when several share a fabric.
