"""Gates: tests stay mechanics (no training condition under tests/); every module has its doc;
every finding is a note that cites every probe beside it; every probe pins the CPU before importing
jax (a shared machine: nothing takes a GPU on its own)."""

import fnmatch
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
# a condition belongs to a recipe, not a test (Meta is meta-learning's condition, step 2)
TRAINING = re.compile(r"\b(recipes|lr=|DescentRecipe\(|MetaRecipe\()")
LOOPS = re.compile(r"\b(descend|fit|trajectory)\(|recipes\.run\(")  # tested where defined


def test_no_training_condition_lives_under_tests():
    for f in (ROOT / "tests").glob("test_*.py"):
        text = f.read_text()
        if f.name != "test_gates.py":
            assert not TRAINING.search(text), f"{f.name}: a training condition belongs in recipes"
        if f.name not in ("test_gates.py", "test_descent.py"):
            assert not LOOPS.search(text), f"{f.name}: a training loop is a claim, not a test"


def test_every_module_has_its_doc():
    for m in (ROOT / "src" / "loom").glob("*.py"):
        if m.name != "__init__.py":
            assert (ROOT / "docs" / f"{m.stem}.md").exists(), f"docs/{m.stem}.md is missing"


FINDING = re.compile(r"\d{4}-\d{2}-\d{2}-[a-z0-9-]+$")
CITED = re.compile(r"`([^`\s]*\d{4}-\d{2}-\d{2}-[^`\s]*\.py)`")  # a dated probe named in a note


def expand(pattern):
    """`a-{x,y}.py` → `a-x.py`, `a-y.py`: a note's brace form for a family of probes."""
    m = re.search(r"\{([^}]*)\}", pattern)
    if not m:
        return [pattern]
    head, tail = pattern[: m.start()], pattern[m.end() :]
    return [q for alt in m.group(1).split(",") for q in expand(head + alt + tail)]


def probes(finding):
    return [f for f in finding.glob("*.py") if f.name != "test_claim.py"]


def test_every_finding_is_a_note_that_cites_every_probe_beside_it():
    for finding in sorted((ROOT / "findings").iterdir()):
        if not finding.is_dir():
            continue
        assert FINDING.match(finding.name), f"{finding.name}: a finding is dated and slugged"
        note = finding / "note.md"
        assert note.exists(), f"{finding.name}: no note.md"
        cited = [q for c in CITED.findall(note.read_text()) for q in expand(c.rsplit("/", 1)[-1])]
        for f in probes(finding):
            assert FINDING.match(f.stem + "-x"), f"{f}: a probe is dated; helpers are not allowed"
            assert any(fnmatch.fnmatch(f.name, q) for q in cited), f"{f} is not cited by its note"
        for c in CITED.findall(note.read_text()):  # and every citation resolves
            base = ROOT / c.rsplit("/", 1)[0] if "/" in c else finding
            hits = [h for q in expand(c.rsplit("/", 1)[-1]) for h in base.glob(q)]
            assert hits, f"{finding.name}: `{c}` names no file"


def test_every_probe_pins_the_cpu_before_importing_jax():
    for finding in (ROOT / "findings").iterdir():
        for f in probes(finding) if finding.is_dir() else []:
            text = f.read_text()
            pin = text.find('os.environ.setdefault("JAX_PLATFORMS", "cpu")')
            imports = [i for i in (text.find("\nimport jax"), text.find("\nfrom loom")) if i >= 0]
            assert imports and 0 <= pin < min(imports), f"{f.name}: pin the CPU before jax or loom"
