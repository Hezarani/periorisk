"""Central settings for the training pipeline. Change things here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
PACKAGE_DATA_DIR = ROOT / "periorisk" / "data"   # portable model used by the Python package
DOCS_DIR = ROOT / "docs"                          # GitHub Pages web app
FIXTURE_DIR = ROOT / "tests" / "fixtures"

MODEL_VERSION = "1.0.0"
MODEL_DATE = "2026-09-29"   # date model 1.0.0 was first trained; fixed so that re-runs are byte-identical

# NHANES cycles with the full-mouth periodontal exam (adults 30+).
# suffix -> (start year used in the CDC download URL, label)
CYCLES = {
    "F": ("2009", "2009-2010"),
    "G": ("2011", "2011-2012"),
    "H": ("2013", "2013-2014"),
}
TRAIN_CYCLES = ["F", "G"]  # 2009-2012 are used to build the model
TEST_CYCLES = ["H"]        # 2013-2014 is held out (temporal validation)

URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file}_{suffix}.xpt"

# NHANES files and the columns kept from each one.
# SDMVPSU / SDMVSTRA (masked variance units and strata) are used only for design-based
# (survey-weighted) confidence intervals in the evaluation, never as model inputs.
# OHQ835 ("Do you think you might have gum disease?") is used only in a sensitivity analysis.
FILES = {
    "DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "WTMEC2YR",
             "SDMVPSU", "SDMVSTRA"],
    "OHQ": ["SEQN", "OHQ835", "OHQ845", "OHQ850", "OHQ855", "OHQ860", "OHQ865", "OHQ870", "OHQ875"],
    "SMQ": ["SEQN", "SMQ020", "SMQ040"],
    "DIQ": ["SEQN", "DIQ010"],
    "BMX": ["SEQN", "BMXBMI"],
    "OHXPER": None,  # keep every column of the periodontal exam
}

# Periodontal exam: sites between the teeth (interproximal) used by the CDC/AAP definition.
# D = disto-facial, S = mesio-facial, P = disto-lingual, A = mesio-lingual
INTERPROXIMAL_SITES = ["D", "S", "P", "A"]
TEETH = [f"{t:02d}" for t in list(range(2, 16)) + list(range(18, 32))]  # third molars excluded

# Predictors a person can answer about themselves (no clinical exam needed).
NUMERIC_FEATURES = ["age", "bmi", "gum_health_rating", "floss_days", "mouthwash_days"]
BINARY_FEATURES = ["male", "ever_treated_gum_disease", "ever_loose_teeth",
                   "told_bone_loss", "teeth_dont_look_right"]
CATEGORICAL_FEATURES = ["education", "smoking", "diabetes"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET = "perio_mod_severe"

# Risk bands shown in the web app (predicted probability).
RISK_BANDS = [(0.0, 0.30, "low"), (0.30, 0.60, "moderate"), (0.60, 1.01, "high")]

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
