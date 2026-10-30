"""PerioRisk - periodontitis risk screening from 14 self-reported questions.

    >>> import periorisk
    >>> r = periorisk.predict(age=58, sex="male", smoking="current", gum_health="fair",
    ...                       loose_teeth="yes", floss_days=0)
    >>> r.band
    'high'

The model estimates the probability of moderate or severe periodontitis (CDC/AAP surveillance
case definition) and was trained on NHANES 2009-2012 and validated on NHANES 2013-2014. It is an
educational screening tool, not a diagnosis. Unanswered questions are filled with a typical answer
from the training data (the most common answer, or the median for numbers).
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional

from ._model import PortableModel, load_model, load_model_file, recalibrate
from .questions import (AGE_MIN, AGE_TOP, BMI, BY_KEY, GUM_HEALTH_SCORE, HEIGHT_CM, QUESTIONS,
                        WEIGHT_KG, Question)

__version__ = "1.0.0"
__all__ = ["predict", "predict_features", "recalibrate", "model_info", "questionnaire",
           "Result", "Contribution", "PerioRiskError", "OutOfScopeError", "QUESTIONS", "Question",
           "load_model", "load_model_file", "PortableModel", "__version__"]


class PerioRiskError(ValueError):
    """An answer is missing, invalid or implausible."""


class OutOfScopeError(PerioRiskError):
    """The person is outside the population the model was developed for."""


LABELS = {
    "age": "Age", "sex": "Sex", "bmi": "Height and weight", "education": "Education",
    "smoking": "Smoking", "diabetes": "Diabetes", "gum_health": "Self-rated teeth and gums",
    "gum_treatment": "Past treatment for gum disease", "loose_teeth": "A tooth became loose on its own",
    "bone_loss": "Told of bone loss around teeth", "tooth_doesnt_look_right": "A tooth that doesn't look right",
    "floss_days": "Days cleaning between teeth", "mouthwash_days": "Days using mouthwash for dental problems",
}
# question key -> model input
_FEATURE_OF = {"age": "age", "sex": "male", "bmi": "bmi", "education": "education", "smoking": "smoking",
               "diabetes": "diabetes", "gum_health": "gum_health_rating",
               "gum_treatment": "ever_treated_gum_disease", "loose_teeth": "ever_loose_teeth",
               "bone_loss": "told_bone_loss", "tooth_doesnt_look_right": "teeth_dont_look_right",
               "floss_days": "floss_days", "mouthwash_days": "mouthwash_days"}
BAND_LABELS = {"low": "Lower risk", "moderate": "Moderate risk", "high": "Higher risk"}


@dataclass
class Contribution:
    """How one answer moved the estimate, compared with a typical answer in the training data (the
    most common answer, or the median for numbers such as age). This describes the model, not cause
    and effect."""
    question: str
    label: str
    answer: object
    change: float           # probability with this answer minus probability with the typical answer


@dataclass
class Result:
    probability: float
    band: str                          # "low" | "moderate" | "high"
    band_label: str
    band_observed_rate: float          # share of 2013-2014 test participants in this band who had the disease
    screen_positive: bool              # probability >= the screening threshold (Youden's J, training data)
    threshold: float
    answers: dict                      # normalised answers
    model_inputs: dict                 # the 13 values the model saw (None = filled with the typical answer)
    unanswered: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    contributions: list = field(default_factory=list)
    model_version: str = ""

    @property
    def percent(self) -> int:
        return int(round(self.probability * 100))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["percent"] = self.percent
        return d

    def summary(self) -> str:
        rng = {"low": "below 30%", "moderate": "30-60%", "high": "60% or more"}[self.band]
        lines = [f"Estimated chance of moderate or severe periodontitis: {self.percent}% ({self.band_label})",
                 f"In the 2013-2014 validation survey, {self.band_observed_rate:.0%} of the people whose estimate "
                 f"fell in this band ({rng}) had moderate or severe periodontitis."]
        if self.contributions:
            lines.append("What moved the estimate most (percentage points, compared with a typical answer):")
            for c in self.contributions[:5]:
                lines.append(f"  {c.change * 100:+5.1f} points  {c.label}: {c.answer}")
        lines += [f"Note: {n}" for n in self.notes]
        lines.append("Educational screening estimate only - not a diagnosis. "
                     "A dental examination with a periodontal probe is needed to confirm gum disease.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
def _norm(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_").replace("'", "")
    return s or None


def _choice(key: str, value) -> Optional[str]:
    v = _norm(value)
    if v is None:
        return None
    q = BY_KEY[key]
    if v not in q.choices:
        raise PerioRiskError(f"{key} must be one of {sorted(q.choices)}, got {value!r}")
    return v


_NUMBER = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


def _number(key: str, value, lo: float, hi: float, integer: bool = False) -> Optional[float]:
    """A finite number in [lo, hi]; None/blank/NaN = not answered. Same rules as docs/periorisk.js."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise PerioRiskError(f"{key} must be a number, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if not _NUMBER.fullmatch(text):
            raise PerioRiskError(f"{key} must be a number, got {value!r}")
        value = text
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        raise PerioRiskError(f"{key} must be a number, got {value!r}") from None
    if math.isnan(x):
        return None
    if not lo <= x <= hi:   # also rejects infinity
        raise PerioRiskError(f"{key} must be between {lo:g} and {hi:g}, got {value!r}")
    if integer and x != math.floor(x):
        raise PerioRiskError(f"{key} must be a whole number, got {value!r}")
    return x


def _normalise(answers: Mapping) -> tuple[dict, dict, list]:
    unknown = set(answers) - set(BY_KEY) - {"bmi"}
    if unknown:
        raise PerioRiskError(f"unknown question(s): {', '.join(sorted(unknown))}. "
                             f"See periorisk.questionnaire().")
    notes = []
    if _choice("natural_teeth", answers.get("natural_teeth")) == "no":
        raise OutOfScopeError("PerioRisk is for people with at least two natural teeth; "
                              "the model learned only from people who had them.")
    age = _number("age", answers.get("age"), 0, 130)
    if age is None:
        raise PerioRiskError("age is required")
    if age < AGE_MIN:
        raise OutOfScopeError("PerioRisk is for adults aged 30 or older (the NHANES periodontal "
                              "examination was done only from age 30).")
    if age > AGE_TOP:
        notes.append("Ages above 80 are treated as 80, as in NHANES.")
    a = {"age": age}
    for key in ("sex", "education", "smoking", "diabetes", "gum_health", "gum_treatment",
                "loose_teeth", "bone_loss", "tooth_doesnt_look_right"):
        a[key] = _choice(key, answers.get(key))
    for key in ("floss_days", "mouthwash_days"):
        a[key] = _number(key, answers.get(key), 0, 7, integer=True)

    bmi = _number("bmi", answers.get("bmi"), *BMI)
    h = _number("height_cm", answers.get("height_cm"), *HEIGHT_CM)
    w = _number("weight_kg", answers.get("weight_kg"), *WEIGHT_KG)
    if bmi is not None and (h is not None or w is not None):
        raise PerioRiskError("give either bmi or height_cm and weight_kg, not both")
    if bmi is None and (h is None) != (w is None):
        raise PerioRiskError("height_cm and weight_kg must be given together")
    if bmi is None and h is not None:
        bmi = w / (h / 100.0) ** 2
        if not BMI[0] <= bmi <= BMI[1]:
            raise PerioRiskError(f"height and weight give a body-mass index of {bmi:.0f}, which is "
                                 "outside the plausible range; please check them")
        a["height_cm"], a["weight_kg"] = h, w
    a["bmi"] = bmi

    x = {
        "age": min(age, AGE_TOP),
        "bmi": bmi,
        "gum_health_rating": None if a["gum_health"] is None else float(GUM_HEALTH_SCORE[a["gum_health"]]),
        "floss_days": a["floss_days"],
        "mouthwash_days": a["mouthwash_days"],
        "male": None if a["sex"] is None else (1.0 if a["sex"] == "male" else 0.0),
        "ever_treated_gum_disease": _yn(a["gum_treatment"]),
        "ever_loose_teeth": _yn(a["loose_teeth"]),
        "told_bone_loss": _yn(a["bone_loss"]),
        "teeth_dont_look_right": _yn(a["tooth_doesnt_look_right"]),
        "education": a["education"],
        "smoking": a["smoking"],
        "diabetes": a["diabetes"],
    }
    return a, x, notes


def _yn(v: Optional[str]) -> Optional[float]:
    return None if v in (None, "not_sure") else (1.0 if v == "yes" else 0.0)


def _display(key: str, a: dict):
    if key == "bmi":
        return f"BMI {a['bmi']:.1f}"
    v = a[key]
    q = BY_KEY.get(key)
    if q is not None and q.choices and v in q.choices:
        return q.choices[v]
    return int(v) if isinstance(v, float) and v.is_integer() else v


# ---------------------------------------------------------------------------------------------
def predict(answers: Optional[Mapping] = None, /, *, explain: bool = True,
            model: Optional[PortableModel] = None, **kwargs) -> Result:
    """Estimate the probability of moderate or severe periodontitis from questionnaire answers.

    Answers can be passed as a dict, as keyword arguments, or both. Only ``age`` is required;
    any other question left out (or answered "not_sure") is filled with a typical answer from the
    training data (most common answer, or median for numbers), which makes the estimate less personal. See :func:`questionnaire`
    for every question, its NHANES wording and the allowed answers.
    """
    merged = dict(answers or {}) | kwargs
    m = model or load_model()
    a, x, notes = _normalise(merged)
    p = m.probability(x)
    band = m.band(p)
    unanswered = [k for k, f in _FEATURE_OF.items() if x[f] is None]
    if unanswered:
        notes.append(f"{len(unanswered)} question(s) not answered; a typical answer was used instead.")
    contributions = []
    if explain:
        for key, feat in _FEATURE_OF.items():
            if x[feat] is None:
                continue
            ref = m.probability(x | {feat: None})
            contributions.append(Contribution(key, LABELS[key], _display(key, a), p - ref))
        contributions.sort(key=lambda c: -abs(c.change))
    return Result(probability=p, band=band["band"], band_label=BAND_LABELS[band["band"]],
                  band_observed_rate=band["test_observed"], screen_positive=p >= m.threshold,
                  threshold=m.threshold, answers=a, model_inputs=x, unanswered=unanswered,
                  notes=notes, contributions=contributions, model_version=m.version)


def predict_features(features: Mapping, model: Optional[PortableModel] = None) -> float:
    """Low-level: probability from the 13 model inputs exactly as coded in the training data
    (e.g. ``gum_health_rating`` 1-5, ``male`` 0/1, ``smoking`` "never"/"former"/"current").
    Missing inputs (None/NaN) are imputed."""
    return (model or load_model()).probability(dict(features))


def questionnaire() -> list[Question]:
    """The questions, their NHANES source variables and the allowed answers."""
    return list(QUESTIONS)


def model_info() -> dict:
    """Version, outcome, population, threshold, risk bands and headline performance."""
    s = load_model().spec
    keep = ("model_version", "outcome", "population", "threshold", "train_prevalence",
            "risk_bands", "performance", "exported_from", "model_date", "age_range")
    return copy.deepcopy({k: s[k] for k in keep if k in s}) | {"package_version": __version__}
