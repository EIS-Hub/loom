# Queue — the next chunks, in order

One live file, rewritten in place. Gabriel pulls from the top; no chunk opens that is not next.
A chunk is at most ~150 hand-written lines plus its test, read in full in one sitting.

| # | step | chunk | kind | reading | prerequisite | status |
|---|---|---|---|---|---|---|
| 1 | 0 | One tile computes: LUT tables as a pytree, wiring as indices, soft and hard read, truth-table tasks, direct descent as the floor | science | 25 min | none | in review |
| 2 | 1 | Signals I: what a local update may read (nothing, task identity, error); the uniform adjoint and the relay through each gate's own table | science | 30 min | 1 | queued |
| 3 | 1 | Signals II: straight-through on the soft and hard read; the check that the relay keeps the gradient's sign where the uniform adjoint loses it | science | 30 min | 2 | queued |

Notes carried forward from review: step 2's regime is one function, a stream of case windows of width W (batched = all cases, online = 1); descent and the rule both consume windows. Step 1 adds the straight-through read (hard forward, soft gradient). softjax enters at step 6 with selection. Tasks gain an output mask, not zero padding, when several share a fabric.
