"""Step 2 - Clean the raw NHANES data, apply the CDC/AAP periodontitis case definition, and build
the model features.

Case definition (Eke et al., J Periodontol 2012; Page & Eke, J Periodontol 2007), using
interproximal sites of all teeth except third molars:
  severe   : >=2 interproximal sites with CAL >= 6 mm (not on the same tooth)
             AND >=1 interproximal site with PD >= 5 mm
  moderate : >=2 interproximal sites with CAL >= 4 mm (not on the same tooth)
             OR >=2 interproximal sites with PD >= 5 mm (not on the same tooth)
  mild     : (>=2 interproximal sites with CAL >= 3 mm AND >=2 interproximal sites
             with PD >= 4 mm, not on the same tooth) OR one site with PD >= 5 mm
The model predicts moderate OR severe periodontitis (yes / no).
"""
import numpy as np
import pandas as pd

from .config import CYCLES, FILES, INTERPROXIMAL_SITES, RAW_DIR, TARGET, TEETH

MISSING_CODES = {7, 9, 77, 99, 777, 999}  # NHANES "refused" / "don't know"


def _read(file: str, suffix: str) -> pd.DataFrame:
    df = pd.read_sas(RAW_DIR / f"{file}_{suffix}.xpt", format="xport")
    cols = FILES[file]
    return df if cols is None else df[cols]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (periodontal exam, covariates), one row per participant."""
    perio, covs = [], []
    for suffix in CYCLES:
        p = _read("OHXPER", suffix)
        p.insert(1, "cycle", CYCLES[suffix][1])
        perio.append(p)
        c = _read("DEMO", suffix)
        for f in ["OHQ", "SMQ", "DIQ", "BMX"]:
            c = c.merge(_read(f, suffix), on="SEQN", how="left")
        c.insert(1, "cycle", CYCLES[suffix][1])
        covs.append(c)
    perio = pd.concat(perio, ignore_index=True)
    covs = pd.concat(covs, ignore_index=True)
    # SAS transport files store 0 as a tiny float (5e-79); rounding fixes that.
    num = covs.select_dtypes("number").columns.drop(["WTMEC2YR", "BMXBMI"])
    covs[num] = covs[num].round(0)
    return perio, covs


def _site_matrix(perio: pd.DataFrame, measure: str) -> np.ndarray:
    """Array of shape (people, teeth, sites) for measure 'LA' (CAL) or 'PC' (PD)."""
    arr = np.stack([
        np.stack([perio[f"OHX{t}{measure}{s}"].to_numpy(float) for s in INTERPROXIMAL_SITES], axis=1)
        for t in TEETH
    ], axis=1)
    arr = np.round(arr, 0)
    arr[arr >= 99] = np.nan  # 99 = could not be assessed
    return arr


def classify(cal: np.ndarray, pd_: np.ndarray) -> np.ndarray:
    """CDC/AAP category for each person from (people, teeth, sites) arrays of interproximal
    clinical attachment loss (cal) and probing depth (pd_) in mm. NaN = not measured."""
    def teeth_with(arr, thr):  # number of teeth with at least one site >= thr
        return np.sum(np.any(arr >= thr, axis=2), axis=1)

    def sites_with(arr, thr):
        return np.sum(arr >= thr, axis=(1, 2))

    with np.errstate(invalid="ignore"):  # NaN >= x is simply False
        severe = (teeth_with(cal, 6) >= 2) & (sites_with(pd_, 5) >= 1)
        moderate = (teeth_with(cal, 4) >= 2) | (teeth_with(pd_, 5) >= 2)
        mild = ((teeth_with(cal, 3) >= 2) & (teeth_with(pd_, 4) >= 2)) | (sites_with(pd_, 5) >= 1)
    return np.select([severe, moderate, mild], ["severe", "moderate", "mild"], "none")


def case_definition(perio: pd.DataFrame) -> pd.DataFrame:
    cal = _site_matrix(perio, "LA")
    pd_ = _site_matrix(perio, "PC")
    out = pd.DataFrame({"SEQN": perio["SEQN"].to_numpy()})
    out["n_teeth_measured"] = np.sum(np.any(~np.isnan(cal), axis=2), axis=1)
    out["max_cal_mm"] = np.max(np.where(np.isnan(cal), -1, cal), axis=(1, 2))
    out["max_pd_mm"] = np.max(np.where(np.isnan(pd_), -1, pd_), axis=(1, 2))
    out["perio_category"] = classify(cal, pd_)
    out[TARGET] = out["perio_category"].isin(["moderate", "severe"]).astype(int)
    return out


def _clean_code(s: pd.Series) -> pd.Series:
    return s.where(~s.isin(MISSING_CODES))


def build_features(covs: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({"SEQN": covs["SEQN"], "cycle": covs["cycle"]})
    f["age"] = covs["RIDAGEYR"]  # NHANES top-codes age at 80
    f["male"] = (covs["RIAGENDR"] == 1).astype(int)
    f["bmi"] = covs["BMXBMI"]    # measured in the exam centre (the app asks for height and weight)

    edu = _clean_code(covs["DMDEDUC2"])
    f["education"] = edu.map({1: "less_than_high_school", 2: "less_than_high_school",
                              3: "high_school", 4: "more_than_high_school",
                              5: "more_than_high_school"})

    # SMQ020: smoked at least 100 cigarettes in life? SMQ040: smoke now (every day / some days / not at all)
    ever = _clean_code(covs["SMQ020"])
    now = _clean_code(covs["SMQ040"])
    f["smoking"] = np.select([ever == 2, now.isin([1, 2]), now == 3],
                             ["never", "current", "former"], default=None)
    f["smoking"] = f["smoking"].replace({None: np.nan})

    f["diabetes"] = _clean_code(covs["DIQ010"]).map({1: "yes", 2: "no", 3: "borderline"})

    # OHQ845: 1 = excellent ... 5 = poor
    f["gum_health_rating"] = _clean_code(covs["OHQ845"])
    for col, name in [("OHQ850", "ever_treated_gum_disease"), ("OHQ855", "ever_loose_teeth"),
                      ("OHQ860", "told_bone_loss"), ("OHQ865", "teeth_dont_look_right")]:
        f[name] = _clean_code(covs[col]).map({1: 1, 2: 0})
    for col, name in [("OHQ870", "floss_days"), ("OHQ875", "mouthwash_days")]:
        f[name] = covs[col].where(covs[col].between(0, 7))

    # Kept for fairness checks, survey-weighted evaluation and a sensitivity analysis - NOT model inputs
    f["think_gum_disease"] = _clean_code(covs["OHQ835"]).map({1: 1, 2: 0})
    f["race_ethnicity"] = covs["RIDRETH1"].map({1: "Mexican American", 2: "Other Hispanic",
                                                3: "Non-Hispanic White", 4: "Non-Hispanic Black",
                                                5: "Other / multiracial"})
    f["exam_weight"] = covs["WTMEC2YR"]
    f["psu"] = covs["SDMVPSU"]
    f["stratum"] = covs["SDMVSTRA"]
    return f


def build_cohort(perio: pd.DataFrame, covs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply eligibility rules. Returns (clean cohort, features table, flow counts)."""
    flow = {"periodontal_file_rows": len(perio)}
    complete = perio[perio["OHDPDSTS"] == 1]  # 1 = complete exam (2 = partial, 3 = not done)
    flow["exam_complete"] = len(complete)
    flow["excluded_exam_partial"] = int((perio["OHDPDSTS"] == 2).sum())
    flow["excluded_exam_not_done"] = int((perio["OHDPDSTS"] == 3).sum())

    labels = case_definition(complete)
    feats = build_features(covs)
    df = labels.merge(feats, on="SEQN", how="inner")
    df = df[df["age"] >= 30]
    flow["age_30_plus"] = len(df)
    flow["excluded_no_teeth_measured"] = int((df["n_teeth_measured"] == 0).sum())
    flow["excluded_one_tooth_measured"] = int((df["n_teeth_measured"] == 1).sum())
    df = df[df["n_teeth_measured"] >= 2]
    flow["two_or_more_teeth_measured"] = len(df)

    clean_cohort = df[["SEQN", "cycle", "n_teeth_measured", "max_cal_mm", "max_pd_mm",
                       "perio_category", TARGET]].reset_index(drop=True)
    features = df.drop(columns=["n_teeth_measured", "max_cal_mm", "max_pd_mm",
                                "perio_category"]).reset_index(drop=True)
    return clean_cohort, features, flow
