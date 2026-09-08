"""Gates: tests stay mechanics (no training condition under tests/); every module has its doc;
every probe pins the CPU before importing jax (a shared machine: nothing takes a GPU on its own)."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
# a condition belongs to a recipe, not a test (Meta is meta-learning's condition, step 2)
TRAINING = re.compile(r"\b(recipes|lr=|Descent\(|Meta\()")
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


def test_every_probe_pins_the_cpu_before_importing_jax():
    for f in (ROOT / "probes").glob("*.py"):
        text = f.read_text()
        pin = text.find('os.environ.setdefault("JAX_PLATFORMS", "cpu")')
        imports = [i for i in (text.find("\nimport jax"), text.find("\nfrom loom")) if i >= 0]
        assert imports and 0 <= pin < min(imports), f"{f.name}: pin the CPU before jax or loom"
