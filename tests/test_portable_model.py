"""The pure-Python engine must reproduce scikit-learn's predictions for every NHANES participant.

tests/fixtures/sklearn_probabilities.csv is written by the training pipeline from the fitted
scikit-learn model (CalibratedClassifierCV), so this test needs neither scikit-learn nor numpy.
"""
import csv
import json
from pathlib import Path

import periorisk
from periorisk._model import PortableModel

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = {"education", "smoking", "diabetes"}


def load_rows():
    with open(ROOT / "data" / "processed" / "analytic_dataset.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(ROOT / "tests" / "fixtures" / "sklearn_probabilities.csv", newline="") as fh:
        expected = {r["SEQN"]: float(r["probability"]) for r in csv.DictReader(fh)}
    feats = periorisk.load_model().features
    out = []
    for r in rows:
        x = {f: (None if r[f] == "" else (r[f] if f in CATEGORICAL else float(r[f]))) for f in feats}
        out.append((x, expected[str(int(float(r["SEQN"])))], r))
    return out


def test_matches_scikit_learn_for_all_participants():
    m = periorisk.load_model()
    rows = load_rows()
    assert len(rows) == 10_660
    worst = max(abs(m.probability(x) - p) for x, p, _ in rows)
    assert worst < 1e-12, worst


def test_rows_with_missing_answers_are_covered():
    # imputation paths must be exercised by the parity test above
    rows = load_rows()
    assert sum(any(v is None for v in x.values()) for x, _, _ in rows) > 100


def test_test_set_auc_reproduces_the_report():
    m = periorisk.load_model()
    pairs = [(m.probability(x), int(r["perio_mod_severe"])) for x, _, r in load_rows() if r["cycle"] == "2013-2014"]
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    # Mann-Whitney AUC with ties counted as one half
    neg_sorted = sorted(neg)
    import bisect
    wins = sum(bisect.bisect_left(neg_sorted, p) + 0.5 * (bisect.bisect_right(neg_sorted, p) - bisect.bisect_left(neg_sorted, p))
               for p in pos)
    auc = wins / (len(pos) * len(neg))
    spec = m.spec["performance"]
    assert len(pairs) == spec["n_test"] == 3610
    assert abs(auc - spec["test_auc"]) < 1e-9


def test_model_file_is_well_formed():
    spec = json.loads((ROOT / "periorisk" / "data" / "model.json").read_text())
    m = PortableModel(spec)
    assert m.version == "1.0.0"
    assert len(m.folds) == 5
    assert all(len(f.trees) == 300 for f in m.folds)
    assert [b["band"] for b in m.risk_bands] == ["low", "moderate", "high"]
    assert 0.4 < m.threshold < 0.5
    for f in m.folds:
        n_inputs = len(m.numeric) + len(m.binary) + sum(len(c) for c in m.categories.values())
        for feat, *_ in f.trees:
            assert max(feat) < n_inputs


def test_web_model_js_is_the_same_model():
    text = (ROOT / "docs" / "model.js").read_text()
    prefix = "window.PERIORISK_MODEL = "
    body = text[text.index(prefix) + len(prefix):].rstrip().rstrip(";")
    assert json.loads(body) == json.loads((ROOT / "periorisk" / "data" / "model.json").read_text())
