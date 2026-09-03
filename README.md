# loom

**The machine that weaves the fabric.** The core of *self-constructing machines*: one small local
rule that compiles a computation onto a substrate, heals it when the substrate breaks, and keeps
learning, built here from first principles as a progression of steps, each small enough to be read
in full and owned.

The programme map (the two axes, the three steps, where every repo sits) is the meta repo's
`VISION.md`; the decision that started this repo is its `decisions/2026-09-03-fresh-start-rebuild-the-core.md`.

## Shape

- **A substrate is data plus one function.** A config pytree and one pure forward, with a soft read
  for gradients and a hard read for deploy. Anything that passes the same two functions is a
  substrate; the differentiable Mosaic core (EIS-Hub) is the second one, wrapped in its own repo.
- **A signal, a rule and a regime are pure functions** on local arrays. Composition is plain
  arguments. Locality is structural: what is not an argument cannot be read.
- **One combinatorial test** runs every substrate × optimiser × signal combination the code claims
  to support. **One check per step**, asserted in CI. Numbers live in the checks, nowhere else.
- **No ladder, no rungs.** The step number is the landmark and never moves.

## The steps

| step | the piece | the check |
|---|---|---|
| 0 | One tile computes: LUT tables, wiring as indices, soft and hard read, truth-table tasks, direct descent as the floor | descent reaches the target on the hard read; soft ≡ hard at deploy |
| 1 | Signals: what a local update may read and how error travels (uniform adjoint, relay through the table); the read mode as an axis, soft or straight-through on the hard tables | the relay keeps the true gradient's sign where the uniform adjoint loses it; whether training the deployed circuit directly closes the soft/hard gap by construction |
| 2 | The training workshop: pool-based meta-learning with the **smallest rule as the inner update**, Δ = −η·signal, where only η is learned by the outer loop through truncated BPTT; the pool holds tile states of every age (a fraction re-seeded each outer step), the tables are never meta-learned; a minimal train/held-out task split; the online window | from a non-functional η the outer loop recovers a working one under a sign-consistent signal and cannot under a sign-flipped one; the tuned rule adapts a fresh tile to a held-out task |
| 3 | The rule: one small function of (logit, relayed error) applied at every logit, **replacing −η·signal as the inner update in the same workshop** | with the relayed error the rule discovers held-out tasks and the blind rule does not; its margin over descent on the same signal is measured, either sign a finding |
| 4 | Damage and heal; the basin over degenerate solutions | function recovered in a different configuration |
| 5 | Evaluation discipline, re-lifted from blastema's design: {train, held-out} wiring × {train, held-out} task, the input-case split as a third axis, the memorisation gap, paired seeds, the deploy window as a visible axis | every check runs paired, at the deploy window; the tile is owned |
| 6 | Wiring as configuration: per-port selection, a lost core routed around | frozen mis-wiring fails; route-around recovers what in-tile repair cannot |
| 7 | The second substrate: the Mosaic core through the same two functions | it passes every check the LUT fabric passes |
| 8 | The maze: arbitrary I/O; a second task superposed | the dataflow grows between arbitrary points; the second task reuses the first |

*The steps are a plan, not a contract: they change when a check teaches us, and the change is recorded in the PR that makes it. Steps 6 to 8 are placeholders, each to be split into several chunks when reached; `QUEUE.md` only ever holds the next three.*

## Not here, by decision

Mosaic hardware specifics (they stay in EIS-Hub repos that import loom). Demo code (demo-kit attaches
later, tag-pinned). Ladders, registries, result tables. Mesh, engram, context, until the tile is owned.

## Working here

```bash
pip install -e ".[dev]" && pre-commit install
pytest                # the fast checks; -m slow for the long ones
pyright --pythonpath "$(which python)"   # so it sees the env's jax
```

Work lands as **chunks**: one question, at most ~150 hand-written lines plus a test, one page in the PR
body (`.github/pull_request_template.md`), read in full. The order is `QUEUE.md`. Code re-lifted from the
frozen reference implementation (`blastema`) carries a one-line provenance comment.

Every module has its didactic page under `docs/`, written in the same chunk as the code it explains: the
PR page is the changelog, the doc is the explanation. Numbers live in the checks; a doc may carry a
worked example, never a result.
