"""The PerioRisk questionnaire.

Question wording follows NHANES 2009-2014 (CDC/NCHS questionnaire files DEMO, SMQ, DIQ and OHQ),
because the model learned from people who answered these questions. Two NHANES smoking questions
(SMQ020, SMQ040) are combined into one, the five NHANES education levels are grouped into three,
and height and weight replace the body-mass index that NHANES measured in its examination centre.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

YES_NO = {"yes": "Yes", "no": "No", "not_sure": "Not sure"}


@dataclass(frozen=True)
class Question:
    key: str                       # argument name in periorisk.predict()
    text: str                      # wording shown to people
    feature: Optional[str]         # model input it feeds (None = eligibility only)
    choices: dict = field(default_factory=dict)   # answer code -> label ({} = number)
    source: str = ""               # NHANES variable(s) the wording comes from
    note: str = ""


QUESTIONS: list[Question] = [
    Question("natural_teeth",
             "Do you still have at least two of your own natural teeth, not counting wisdom teeth? "
             "Teeth with fillings or crowns count; implants, bridges and dentures don't.",
             None, {"yes": "Yes", "no": "No"},
             note="Eligibility: the model learned only from people with two or more natural teeth."),
    Question("age", "How old are you?", "age", source="RIDAGEYR",
             note="30 or older. NHANES records everyone aged 80+ as 80."),
    Question("sex", "Sex", "male", {"female": "Female", "male": "Male"}, source="RIAGENDR"),
    Question("height_cm", "Height in centimetres (optional)", "bmi", source="BMXBMI",
             note="Used with weight to compute body-mass index."),
    Question("weight_kg", "Weight in kilograms (optional)", "bmi", source="BMXBMI"),
    Question("education",
             "What is the highest grade or level of school you have completed, or the highest degree "
             "you have received?", "education",
             {"less_than_high_school": "Less than high school (did not finish)",
              "high_school": "High school graduate or equivalent (e.g. GED)",
              "more_than_high_school": "Some college, vocational or university education, or a degree"},
             source="DMDEDUC2"),
    Question("smoking", "Which describes you best? (Cigarettes only)", "smoking",
             {"never": "I have smoked fewer than 100 cigarettes in my whole life",
              "former": "I have smoked 100 or more cigarettes, but I don't smoke now",
              "current": "I smoke cigarettes now, every day or on some days"},
             source="SMQ020, SMQ040"),
    Question("diabetes",
             "Other than during pregnancy, have you ever been told by a doctor or health professional "
             "that you have diabetes or sugar diabetes?", "diabetes",
             {"no": "No", "borderline": "Borderline", "yes": "Yes"}, source="DIQ010"),
    Question("gum_health", "Overall, how would you rate the health of your teeth and gums?",
             "gum_health_rating",
             {"excellent": "Excellent", "very_good": "Very good", "good": "Good", "fair": "Fair", "poor": "Poor"},
             source="OHQ845"),
    Question("gum_treatment",
             "Have you ever had treatment for gum disease such as scaling and root planing, sometimes "
             "called \"deep cleaning\"?", "ever_treated_gum_disease", YES_NO, source="OHQ850"),
    Question("loose_teeth", "Have you ever had any teeth become loose on their own, without an injury?",
             "ever_loose_teeth", YES_NO, source="OHQ855"),
    Question("bone_loss", "Have you ever been told by a dental professional that you lost bone around your teeth?",
             "told_bone_loss", YES_NO, source="OHQ860"),
    Question("tooth_doesnt_look_right", "During the past three months, have you noticed a tooth that doesn't look right?",
             "teeth_dont_look_right", YES_NO, source="OHQ865"),
    Question("floss_days",
             "Aside from brushing your teeth with a toothbrush, in the last seven days, how many days did "
             "you use dental floss or any other device to clean between your teeth?", "floss_days",
             source="OHQ870", note="0 to 7"),
    Question("mouthwash_days",
             "Aside from brushing your teeth with a toothbrush, in the last seven days, how many days did "
             "you use mouthwash or other dental rinse product that you use to treat dental disease or "
             "dental problems?", "mouthwash_days", source="OHQ875", note="0 to 7"),
]

BY_KEY = {q.key: q for q in QUESTIONS}

GUM_HEALTH_SCORE = {"excellent": 1, "very_good": 2, "good": 3, "fair": 4, "poor": 5}

# Plausible ranges; values outside them are almost certainly typing mistakes.
HEIGHT_CM = (100.0, 250.0)
WEIGHT_KG = (25.0, 350.0)
BMI = (12.0, 90.0)
AGE_MIN, AGE_TOP = 30, 80
