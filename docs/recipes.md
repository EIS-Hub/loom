# Recipes and findings: how we know things

A test is mechanistic: the read is exact on XOR, the hard read emits bits, a signal has the
logits' shape. Deterministic, no seed, no step size, seconds; a red one means the code does not do
what it says. The moment a loop trains under a condition, an assertion becomes an empirical claim,
and a red one can mean a bug, a wrong condition or a wrong claim. That ambiguity is what a ladder
hidden inside tests produces, so the boundary is a directory rather than a habit.

| object | home | what it is | what it is not |
|---|---|---|---|
| **recipe** | `src/loom/recipes.py` | a named, frozen training condition: signal, step size, budget, window, hidden widths, arity, init scale | never a literal in a claim or a note; never overridden in place |
| **test** | `tests/` | mechanics of the code; the merge gate | never a training loop |
| **finding** | `findings/YYYY-MM-DD-slug/` | one dated directory per finding: `note.md`, the probes that printed its tables, and `test_claim.py`, the statement that must keep holding | never edited once it lands; a later finding supersedes it by pointer; no status, no tiers |
| **note** | `findings/…/note.md` | the finding and the path to it (per-seed numbers stay in the tables, with a bold mean ± sd beside them): the question as X against its floor, how the floor was chosen, the recipes by name, the numbers, the hypotheses that fell and the one that held, the claim it left behind and which cells of its tables that claim pins, what would change it | never a claim's only home |
| **probe** | `findings/…/YYYY-MM-DD-slug.py` | the dated script behind a note's tables, run by hand, its printed output pasted into the note; one file, imports only loom; where it trains a tile under a signal and a rule it does so through `recipes.run`, on a named recipe or its unnamed `_replace` variants, so the numbers come from the module the claim asserts on and from the recipe seeds | never imported; never re-implements what a module provides; deletable once its finding is superseded |
| **claim** | `findings/…/test_claim.py` | a qualitative statement on the smallest instance under named recipes, with the seeds that make it a claim; the cells of the note's tables that must keep holding, re-run on every push | not a sweep, a table or a figure; not the science record |

## How they interact

`recipes.run(recipe, task, seed)` is the one place a condition meets a task: it draws the task,
builds the tile to the task's width and fits it. A recipe is named after the loop it runs:
`DescentRecipe` configures `descent.fit`; the meta-learning condition arrives with step 2 and is named
after its loop too. Claims call it with a named recipe and assert on
the result; nothing else in a claim may train. A recipe owns *how* we train; a claim owns *what*
is claimed, *on which task*, with *how many seeds*, and points at its note. Changing a step size
means editing or adding a recipe, a one-file diff next to the claim that depends on it, so the same
claim under another rate cannot happen silently. One swap is allowed in a claim, the recipe's
*signal* (`DEEP_FLOOR._replace(signal=…)`): the signal is what a signals claim is about, everything
else is the condition it holds under; a signal that needs its own condition to be fair, as the bits
do with a smaller step, gets its own recipe. A probe may build unnamed recipe variants: exploring
the condition space is its job, and the note records which conditions it visited.

A finding's three kinds of file are one run of the code, read three ways. The probe sweeps the
condition space through `recipes.run`, so its printed tables come from the module the claim asserts
on and from the recipe seeds; the note pastes those tables and says which cells the claim pins; the
claim names those cells as recipes and re-runs them on every push. A probe that re-implements a
rule, or seeds its own tiles, breaks that link: the note's numbers then belong to nothing that runs
again, and the claim beside them is a different experiment. A probe that measures something the
loop cannot express (a signal at initialisation, a hand-made perturbation) says so in its docstring.
The findings of 2026-09-03 to 2026-09-09 predate this contract and ran their own loops; they are
recorded as they were run.

The lifecycle: a question → a probe → a note with the numbers → if the note leaves a statement that
must keep holding, `test_claim.py` under named recipes → the programme map points at the finding.

## What runs when

`pytest` runs the tests: the merge gate. `pytest findings` runs every finding's claim as a second,
separately named CI job on every PR and push: a red there is a finding, answered in the PR by a note
or a fix, never by a quiet turn of a recipe. When a claim earns a `slow` marker it moves to a nightly
run. Three gates keep the boundary: no training condition may appear under `tests/`, every module
has its page under `docs/`, and every finding is a note that cites every probe beside it.

## What is deliberately absent

No registry of claims, no status vocabulary, no tiers, no generated tables, no byte-pinned
records. Reproducibility is live: a claim re-runs on every CI run, an expensive finding is
reproduced by its probe and recipe on demand. The day any of the above needs a generator or a
schema, it has become the ladder again and should be cut back, not extended.
