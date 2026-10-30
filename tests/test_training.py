"""Checks of the training pipeline's data handling. Needs the training requirements
(pip install -r requirements-training.txt); skipped otherwise."""
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from training.clean import MISSING_CODES, build_features, classify  # noqa: E402
from training.evaluate import calibration_stats, prior_shift, risk_band  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_TEETH, N_SITES = 28, 4


def mouth(cal_sites=(), pd_sites=()):
    """One person with healthy readings (CAL 1 mm, PD 2 mm) except the given
    (tooth, site, value) entries."""
    cal = np.full((1, N_TEETH, N_SITES), 1.0)
    pdp = np.full((1, N_TEETH, N_SITES), 2.0)
    for t, s, v in cal_sites:
        cal[0, t, s] = v
    for t, s, v in pd_sites:
        pdp[0, t, s] = v
    return cal, pdp


@pytest.mark.parametrize("cal_sites, pd_sites, expected", [
    ((), (), "none"),
    # severe: CAL >= 6 on two different teeth and one PD >= 5
    (((0, 0, 6), (5, 2, 7)), ((9, 1, 5),), "severe"),
    # two sites with CAL >= 6 but on the SAME tooth -> not severe; CAL >= 4 on one tooth only -> not moderate
    (((0, 0, 6), (0, 1, 6)), ((9, 1, 5),), "mild"),
    # moderate via CAL >= 4 on two teeth (no deep pocket)
    (((0, 0, 4), (3, 3, 4)), (), "moderate"),
    # moderate via PD >= 5 on two teeth
    ((), ((0, 0, 5), (1, 0, 5)), "moderate"),
    # PD >= 5 on two sites of the same tooth -> only mild
    ((), ((0, 0, 5), (0, 1, 5)), "mild"),
    # mild: CAL >= 3 on two teeth AND PD >= 4 on two teeth
    (((0, 0, 3), (1, 0, 3)), ((2, 0, 4), (3, 0, 4)), "mild"),
    # CAL >= 3 on two teeth but PD >= 4 on one tooth only -> none
    (((0, 0, 3), (1, 0, 3)), ((2, 0, 4),), "none"),
])
def test_cdc_aap_case_definition(cal_sites, pd_sites, expected):
    cal, pdp = mouth(cal_sites, pd_sites)
    assert classify(cal, pdp)[0] == expected


def test_unmeasured_sites_are_ignored():
    cal, pdp = mouth(((0, 0, 6), (5, 2, 7)), ((9, 1, 5),))
    cal[0, 5, 2] = np.nan       # one of the two deep-attachment sites not measured
    assert classify(cal, pdp)[0] == "mild"


def covs(**cols):
    base = dict(SEQN=[1.0], cycle=["2013-2014"], RIDAGEYR=[50.0], RIAGENDR=[1.0], BMXBMI=[27.0],
                DMDEDUC2=[3.0], SMQ020=[1.0], SMQ040=[3.0], DIQ010=[2.0], OHQ835=[2.0], OHQ845=[3.0], OHQ850=[2.0],
                OHQ855=[1.0], OHQ860=[9.0], OHQ865=[2.0], OHQ870=[77.0], OHQ875=[7.0],
                RIDRETH1=[3.0], WTMEC2YR=[1000.0], SDMVPSU=[1.0], SDMVSTRA=[104.0])
    base |= {k: [v] for k, v in cols.items()}
    return pd.DataFrame(base)


def test_feature_coding():
    f = build_features(covs()).iloc[0]
    assert f["male"] == 1 and f["education"] == "high_school" and f["smoking"] == "former"
    assert f["diabetes"] == "no" and f["gum_health_rating"] == 3
    assert f["ever_treated_gum_disease"] == 0 and f["ever_loose_teeth"] == 1
    assert pd.isna(f["told_bone_loss"])               # 9 = don't know
    assert pd.isna(f["floss_days"])                   # 77 = refused
    assert f["mouthwash_days"] == 7


@pytest.mark.parametrize("ever, now, expected", [(2, None, "never"), (1, 1, "current"), (1, 2, "current"),
                                                 (1, 3, "former"), (1, None, None), (9, None, None)])
def test_smoking_coding(ever, now, expected):
    f = build_features(covs(SMQ020=ever, SMQ040=now)).iloc[0]
    assert (pd.isna(f["smoking"]) and expected is None) or f["smoking"] == expected


@pytest.mark.parametrize("code, expected", [(1, "less_than_high_school"), (2, "less_than_high_school"),
                                            (3, "high_school"), (4, "more_than_high_school"),
                                            (5, "more_than_high_school"), (7, None), (9, None)])
def test_education_coding(code, expected):
    f = build_features(covs(DMDEDUC2=code)).iloc[0]
    assert (pd.isna(f["education"]) and expected is None) or f["education"] == expected


def test_missing_codes():
    assert {7, 9, 77, 99} <= MISSING_CODES


def test_calibration_statistics_on_perfectly_calibrated_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    s = calibration_stats(y, p)
    assert s["o_e_ratio"] == pytest.approx(1, abs=0.01)
    assert s["calibration_intercept"] == pytest.approx(0, abs=0.02)
    assert s["calibration_slope"] == pytest.approx(1, abs=0.02)


def test_prior_shift_matches_package():
    import periorisk
    assert prior_shift(np.array([0.4]), 0.487, 0.3)[0] == pytest.approx(periorisk.recalibrate(0.4, 0.3, 0.487))


def test_risk_bands():
    assert [risk_band(p) for p in (0, 0.299, 0.3, 0.599, 0.6, 1.0)] == ["low", "low", "moderate", "moderate", "high", "high"]


def test_reported_numbers_are_consistent():
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    model = json.loads((ROOT / "periorisk" / "data" / "model.json").read_text())
    assert meta["threshold"] == model["threshold"]
    assert meta["test_metrics"]["auc"] == model["performance"]["test_auc"]
    assert meta["n_train"] == 7050 and meta["n_test"] == 3610


def test_sklearn_model_matches_fixture_when_versions_match():
    import sklearn
    if sklearn.__version__ != "1.8.0":
        pytest.skip("the saved model needs scikit-learn 1.8.0")
    import joblib
    from training.config import FEATURES
    model = joblib.load(ROOT / "models" / "model.joblib")
    df = pd.read_csv(ROOT / "data" / "processed" / "analytic_dataset.csv")
    fixture = pd.read_csv(ROOT / "tests" / "fixtures" / "sklearn_probabilities.csv")
    p = model.predict_proba(df[FEATURES])[:, 1]
    assert np.max(np.abs(p - fixture["probability"].to_numpy())) < 1e-15
