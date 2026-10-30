"""The browser engine (docs/periorisk.js) must agree with the Python package.

Runs docs/periorisk.js under Node.js; skipped when Node is not installed.
"""
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

import periorisk
from test_portable_model import load_rows

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
MODEL = ROOT / "periorisk" / "data" / "model.json"
RUNNER = ROOT / "tests" / "js" / "run_engine.js"


def run(mode, rows, tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    out = subprocess.run([NODE, str(RUNNER), mode, str(MODEL), str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_browser_engine_matches_scikit_learn_for_all_participants(tmp_path):
    rows = load_rows()
    probs = run("features", [x for x, _, _ in rows], tmp_path)
    worst = max(abs(p - expected) for p, (_, expected, _) in zip(probs, rows))
    assert worst < 1e-12, worst


def random_answers(rng):
    a = {"natural_teeth": "yes", "age": rng.randint(30, 95)}
    choices = {
        "sex": ["female", "male"], "education": ["less_than_high_school", "high_school", "more_than_high_school"],
        "smoking": ["never", "former", "current"], "diabetes": ["no", "borderline", "yes"],
        "gum_health": ["excellent", "very_good", "good", "fair", "poor"],
        "gum_treatment": ["yes", "no", "not_sure"], "loose_teeth": ["yes", "no", "not_sure"],
        "bone_loss": ["yes", "no", "not_sure"], "tooth_doesnt_look_right": ["yes", "no", "not_sure"],
    }
    for k, opts in choices.items():
        if rng.random() < 0.9:
            a[k] = rng.choice(opts)
    for k in ("floss_days", "mouthwash_days"):
        if rng.random() < 0.9:
            a[k] = rng.randint(0, 7)
    if rng.random() < 0.7:
        a["height_cm"] = round(rng.uniform(145, 200), 1)
        a["weight_kg"] = round(rng.uniform(45, 140), 1)
    return a


def test_browser_engine_matches_python_for_answer_sets(tmp_path):
    rng = random.Random(2026)
    answers = [random_answers(rng) for _ in range(400)]
    js = run("answers", answers, tmp_path)
    for a, j in zip(answers, js):
        r = periorisk.predict(a)
        assert abs(j["probability"] - r.probability) < 1e-12
        assert j["band"] == r.band
        assert j["screen_positive"] == r.screen_positive
        assert sorted(j["unanswered"]) == sorted(r.unanswered)
        py = {c.question: c.change for c in r.contributions}
        assert set(py) == {c["key"] for c in j["contributions"]}
        for c in j["contributions"]:
            assert abs(c["change"] - py[c["key"]]) < 1e-12


@pytest.mark.parametrize("answers, code", [
    ({"age": 29}, "out_of_scope"),
    ({"age": 50, "natural_teeth": "no"}, "out_of_scope"),
    ({"age": 50, "height_cm": 100, "weight_kg": 300}, "implausible_bmi"),
    ({"age": 50, "smoking": "sometimes"}, "invalid"),
    ({"age": 50, "floss_days": 9}, "invalid"),
    ({"age": 50, "height_cm": 170}, "invalid"),
])
def test_browser_engine_refuses_like_python(answers, code, tmp_path):
    (j,) = run("answers", [answers], tmp_path)
    assert j.get("error") == code
    with pytest.raises(periorisk.PerioRiskError):
        periorisk.predict(answers)


def test_model_js_loads_in_a_browser_like_context():
    out = subprocess.run([NODE, str(RUNNER), "modeljs", str(ROOT / "docs" / "model.js")],
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == json.loads(MODEL.read_text())


@pytest.mark.parametrize("answers, python_ok", [
    ({"age": 50, "floss_days": "  "}, True),       # blank = not answered in both
    ({"age": "  50 "}, True),
    ({"age": "0x32"}, False),
    ({"age": "5_0"}, False),
    ({"age": 50, "weight": 70}, False),             # unknown question
    ({"age": 50, "natural_teeth": "banana"}, False),
    ({"age": 50, "floss_days": True}, False),
    ({"age": 50, "mouthwash_days": "1e400"}, False),
])
def test_browser_and_python_validate_alike(answers, python_ok, tmp_path):
    (j,) = run("answers", [answers], tmp_path)
    if python_ok:
        assert abs(j["probability"] - periorisk.predict(answers).probability) < 1e-12
    else:
        assert "error" in j
        with pytest.raises(periorisk.PerioRiskError):
            periorisk.predict(answers)
