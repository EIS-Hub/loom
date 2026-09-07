# Recipes, claims, notes and probes: how we know things

A test is mechanistic: the read is exact on XOR, the hard read emits bits, a signal has the
logits' shape. Deterministic, no seed, no step size, seconds; a red one means the code does not do
what it says. The moment a loop trains under a condition, an assertion becomes an empirical claim,
and a red one can mean a bug, a wrong condition or a wrong claim. That ambiguity is what a ladder
hidden inside tests produces, so the boundary is a directory rather than a habit.

| object | home | what it is | what it is not |
|---|---|---|---|
| **recipe** | `src/loom/recipes.py` | a named, frozen training condition: signal, step size, budget, window, hidden widths, arity, init scale | never a literal in a claim or a note; never overridden in place |
| **test** | `tests/` | mechanics of the code; the merge gate | never a training loop |
| **claim** | `claims/`, one file per note | a qualitative statement on the smallest instance under a named recipe, with the seeds that make it a claim | not a sweep, a table or a figure; not the science record |
| **note** | `notes/YYYY-MM-DD-slug.md` | the finding: the question as X against its floor, the recipes by name, the numbers, the figure, the claims it left behind, what would change it | never edited; a later note supersedes by pointer; no status, no tiers |
| **probe** | `probes/YYYY-MM-DD-slug.py` | the script behind a note when a few lines are not enough; one file, imports only loom | never imported; no shared probe utilities; deletable once its note is superseded |

## How they interact

`recipes.run(recipe, task, seed)` is the one place a condition meets a task: it draws the task,
builds the tile to the task's width and fits it. Claims call it with a named recipe and assert on
the result; nothing else in a claim may train. A recipe owns *how* we train; a claim owns *what*
is claimed, *on which task*, with *how many seeds*, and points at its note. Changing a step size
means editing or adding a recipe, a one-file diff next to the claim that depends on it, so the same
claim under another rate cannot happen silently. A probe may build unnamed recipe variants: exploring
the condition space is its job, and the note records which conditions it visited.

The lifecycle: a question → a probe → a note with the numbers → if the note leaves a statement that
must keep holding, a claim under a named recipe → the programme map points at the note.

## What runs when

`pytest` runs the tests: the merge gate. `pytest claims` runs the claims as a second, separately
named CI job on every PR and push: a red there is a finding, answered in the PR by a note or a
fix, never by a quiet turn of a recipe. When a claim earns a `slow` marker it moves to a nightly
run. Two gates keep the boundary: no training condition may appear under `tests/`, and every
module has its page under `docs/`.

## What is deliberately absent

No registry of claims, no status vocabulary, no tiers, no generated tables, no byte-pinned
records. Reproducibility is live: a claim re-runs on every CI run, an expensive finding is
reproduced by its probe and recipe on demand. The day any of the above needs a generator or a
schema, it has become the ladder again and should be cut back, not extended.
