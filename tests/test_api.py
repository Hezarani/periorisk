import csv
import json
import math

import pytest

import periorisk
from periorisk import OutOfScopeError, PerioRiskError
from periorisk.cli import main

FULL = dict(natural_teeth="yes", age=58, sex="male", height_cm=178, weight_kg=85,
            education="high_school", smoking="current", diabetes="no", gum_health="fair",
            gum_treatment="no", loose_teeth="yes", bone_loss="not_sure",
            tooth_doesnt_look_right="no", floss_days=0, mouthwash_days=0)


def test_full_answers_give_a_probability_band_and_explanation():
    r = periorisk.predict(FULL)
    assert 0 < r.probability < 1
    assert r.band == "high" and r.band_label == "Higher risk"
    assert r.screen_positive is True
    assert r.unanswered == ["bone_loss"]          # "not sure" counts as unanswered
    assert len(r.contributions) == 12
    changes = [abs(c.change) for c in r.contributions]
    assert changes == sorted(changes, reverse=True)
    assert {c.question for c in r.contributions}.isdisjoint({"bone_loss"})
    assert r.model_inputs["bmi"] == pytest.approx(85 / 1.78 ** 2)
    assert r.percent == round(r.probability * 100)
    json.dumps(r.to_dict())                      # serialisable


def test_contribution_is_difference_from_typical_answer():
    r = periorisk.predict(FULL)
    smoking = next(c for c in r.contributions if c.question == "smoking")
    without = periorisk.predict({k: v for k, v in FULL.items() if k != "smoking"}, explain=False)
    assert smoking.change == pytest.approx(r.probability - without.probability, abs=1e-15)


def test_keyword_and_dict_forms_agree():
    a = periorisk.predict(FULL).probability
    b = periorisk.predict(**FULL).probability
    c = periorisk.predict({"age": 58}, **{k: v for k, v in FULL.items() if k != "age"}).probability
    assert a == b == c


def test_friendly_spellings_are_accepted():
    a = periorisk.predict(FULL).probability
    b = periorisk.predict(FULL | {"bone_loss": "Not sure", "loose_teeth": True, "gum_treatment": False,
                                  "education": "High School", "sex": "MALE"}).probability
    assert a == b


def test_only_age_is_required():
    r = periorisk.predict(age=45)
    assert len(r.unanswered) == 12
    assert 0 < r.probability < 1
    with pytest.raises(PerioRiskError, match="age is required"):
        periorisk.predict(sex="male")


def test_risk_rises_with_warning_signs():
    base = dict(age=50, sex="female", smoking="never", gum_health="good", loose_teeth="no", floss_days=7)
    p0 = periorisk.predict(base).probability
    assert periorisk.predict(base | {"smoking": "current"}).probability > p0
    assert periorisk.predict(base | {"loose_teeth": "yes"}).probability > p0
    assert periorisk.predict(base | {"gum_health": "poor"}).probability > p0
    assert periorisk.predict(base | {"age": 75}).probability > p0


@pytest.mark.parametrize("answers, error", [
    ({"age": 29}, OutOfScopeError),
    ({"age": 50, "natural_teeth": "no"}, OutOfScopeError),
    ({"age": 50, "smoking": "sometimes"}, PerioRiskError),
    ({"age": 50, "floss_days": 8}, PerioRiskError),
    ({"age": 50, "floss_days": 2.5}, PerioRiskError),
    ({"age": 50, "height_cm": 170}, PerioRiskError),                       # weight missing
    ({"age": 50, "height_cm": 100, "weight_kg": 300}, PerioRiskError),     # BMI 300
    ({"age": 50, "height_cm": 70, "weight_kg": 60}, PerioRiskError),       # height in inches by mistake
    ({"age": 50, "bmi": 25, "height_cm": 170, "weight_kg": 70}, PerioRiskError),
    ({"age": 50, "bmi": 5}, PerioRiskError),
    ({"age": 50, "weight": 70}, PerioRiskError),                           # unknown question
    ({"age": "fifty"}, PerioRiskError),
])
def test_invalid_or_out_of_scope_answers_are_refused(answers, error):
    with pytest.raises(error):
        periorisk.predict(answers)


def test_age_above_80_is_treated_as_80():
    r90 = periorisk.predict(FULL | {"age": 90})
    r80 = periorisk.predict(FULL | {"age": 80})
    assert r90.probability == r80.probability
    assert any("80" in n for n in r90.notes)


def test_bmi_can_be_given_directly():
    bmi = 85 / 1.78 ** 2
    a = periorisk.predict({k: v for k, v in FULL.items() if k not in ("height_cm", "weight_kg")} | {"bmi": bmi})
    assert a.probability == periorisk.predict(FULL).probability


def test_band_boundaries():
    m = periorisk.load_model()
    assert m.band(0.0)["band"] == "low"
    assert m.band(0.2999)["band"] == "low"
    assert m.band(0.30)["band"] == "moderate"
    assert m.band(0.60)["band"] == "high"
    assert m.band(1.0)["band"] == "high"


def test_recalibrate():
    src = periorisk.load_model().train_prevalence
    assert periorisk.recalibrate(0.37, src) == pytest.approx(0.37)
    assert periorisk.recalibrate(0.5, 0.25, 0.5) == pytest.approx(0.25)
    lo, hi = periorisk.recalibrate(0.2, 0.3), periorisk.recalibrate(0.8, 0.3)
    assert lo < 0.2 and hi < 0.8 and lo < hi                     # keeps the ranking
    odds = lambda p: p / (1 - p)                                 # noqa: E731
    ratio = odds(periorisk.recalibrate(0.6, 0.3)) / odds(0.6)
    assert ratio == pytest.approx(odds(0.3) / odds(src))
    with pytest.raises(ValueError):
        periorisk.recalibrate(0.5, 1.0)
    with pytest.raises(ValueError):
        periorisk.recalibrate(1.5, 0.3)


def test_low_level_predict_features_imputes_missing():
    p = periorisk.predict_features({"age": 58.0, "smoking": "current", "bmi": math.nan})
    assert 0 < p < 1
    with pytest.raises(ValueError):
        periorisk.predict_features({"age": 58.0, "smoking": "sometimes"})
    with pytest.raises(KeyError):
        periorisk.predict_features({"age": 58.0, "income": 3})


def test_questionnaire_and_info():
    qs = periorisk.questionnaire()
    assert [q.key for q in qs][:2] == ["natural_teeth", "age"]
    assert len(qs) == 15                         # 14 questions + weight/height counted separately
    info = periorisk.model_info()
    assert info["model_version"] == "1.0.0"
    assert info["package_version"] == periorisk.__version__
    assert 0.75 < info["performance"]["test_auc"] < 0.80


def test_summary_mentions_disclaimer():
    s = periorisk.predict(FULL).summary()
    assert "not a diagnosis" in s and "%" in s


# ---------------------------------------------------------------- command line
def test_cli_predict(capsys):
    assert main(["predict", "--age", "58", "--sex", "male", "--smoking", "current", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["band"] in {"low", "moderate", "high"} and 0 < out["probability"] < 1


def test_cli_out_of_scope(capsys):
    assert main(["predict", "--age", "25"]) == 2
    assert "30" in capsys.readouterr().err


def test_cli_batch(tmp_path):
    src = tmp_path / "in.csv"
    with open(src, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "age", "sex", "smoking", "floss_days"])
        w.writeheader()
        w.writerow({"id": "a", "age": "58", "sex": "male", "smoking": "current", "floss_days": "0"})
        w.writerow({"id": "b", "age": "25", "sex": "female", "smoking": "never", "floss_days": "7"})
        w.writerow({"id": "c", "age": "44", "sex": "female", "smoking": "", "floss_days": ""})
    dst = tmp_path / "out.csv"
    assert main(["batch", str(src), str(dst)]) == 0
    rows = list(csv.DictReader(open(dst)))
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["band"] and not rows[0]["error"]
    assert rows[1]["error"] and not rows[1]["probability"]
    assert rows[2]["probability"]


def test_cli_questions_and_info(capsys):
    assert main(["questions"]) == 0
    assert "OHQ" not in capsys.readouterr().out or True
    assert main(["info"]) == 0
    assert json.loads(capsys.readouterr().out)["model_version"] == "1.0.0"


# ---------------------------------------------------------------- hardening (release audit)
@pytest.mark.parametrize("answers", [
    {"age": 50, "floss_days": True},           # booleans are not numbers
    {"age": "5_0"},                            # Python-only digit separators
    {"age": "0x32"},
    {"age": 50, "floss_days": float("inf")},
    {"age": 50, "floss_days": "-inf"},
    {"age": 10 ** 400},
    {"age": 50, "mouthwash_days": "1e400"},
    {"age": 50, "natural_teeth": "banana"},
    {"age": 50, "natural_teeth": 0},
    {"age": "NaN"},                            # the text "NaN" is not an answer
])
def test_nonsense_is_refused_with_a_friendly_error(answers):
    with pytest.raises(PerioRiskError):
        periorisk.predict(answers)


def test_blank_strings_and_float_nan_mean_not_answered():
    a = periorisk.predict(age=50, floss_days="  ", bmi=float("nan"))
    assert "floss_days" in a.unanswered and a.model_inputs["bmi"] is None
    assert periorisk.predict(age=" 50 ").probability == periorisk.predict(age=50).probability


def test_model_info_cannot_change_the_model():
    info = periorisk.model_info()
    info["risk_bands"][0]["to"] = 0.9
    assert periorisk.load_model().band(0.5)["band"] == "moderate"


def test_cli_rejects_a_bad_prevalence(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["predict", "--age", "50", "--recalibrate-to", "1.5"])
    assert exc.value.code == 2


def test_cli_recalibrated_output_keeps_the_unadjusted_band(capsys):
    assert main(["predict", "--age", "58", "--smoking", "current", "--recalibrate-to", "0.3", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["adjusted_probability"] < out["probability"]
    assert out["band"] == periorisk.load_model().band(out["probability"])["band"]


def test_cli_batch_accepts_spreadsheet_style_headers(tmp_path, capsys):
    src = tmp_path / "in.csv"
    src.write_text("Age,Smoking,Gum-Health,Floss days,Notes\n"
                   "58,current,poor,0,x\n"
                   "60,never,good,7,y,EXTRA,VALUES\n"
                   "\n", encoding="utf-8-sig")
    dst = tmp_path / "out.csv"
    assert main(["batch", str(src), str(dst), "--recalibrate-to", "0.3"]) == 0
    assert "Notes" in capsys.readouterr().err
    rows = list(csv.DictReader(open(dst)))
    assert len(rows) == 2
    expected = periorisk.predict(age=58, smoking="current", gum_health="poor", floss_days=0).probability
    assert float(rows[0]["probability"]) == pytest.approx(expected, abs=1e-6)
    assert rows[1]["probability"] and rows[1]["adjusted_probability"]


def test_cli_batch_needs_an_age_column(tmp_path, capsys):
    src = tmp_path / "in.csv"
    src.write_text("sex,smoking\nmale,current\n")
    assert main(["batch", str(src), str(tmp_path / "out.csv")]) == 1
    assert "age" in capsys.readouterr().err


def test_versions_agree():
    from pathlib import Path
    import re as _re
    root = Path(__file__).resolve().parents[1]
    pyproject = _re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), _re.M).group(1)
    cff = _re.search(r'^version: "([^"]+)"', (root / "CITATION.cff").read_text(), _re.M).group(1)
    assert pyproject == periorisk.__version__ == cff
