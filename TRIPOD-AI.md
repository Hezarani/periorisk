# TRIPOD+AI checklist — PerioRisk v1.0.0

Reporting checklist for prediction model development and evaluation (Collins et al., *BMJ*
2024;385:e078378, [doi:10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)). Item
topics are paraphrased; see the statement for the full wording. This study both **develops** (D)
the model and **evaluates** (E) it on a later survey cycle (temporal validation).

Where: **R** = [README.md](README.md) · **P** = [paper.md](paper.md) · **MC** = [MODEL_CARD.md](MODEL_CARD.md) ·
**RES** = [reports/results.md](reports/results.md) · code paths are relative to the repository root.

| Item | Topic | Where / how it is addressed |
|---|---|---|
| 1 | Title identifies development + evaluation, population, outcome | P (title); R (first paragraph) |
| 2 | Abstract (TRIPOD+AI for Abstracts) | P (Summary: objective, data, design, main results); CITATION.cff abstract |
| 3a | Healthcare context, rationale, existing models | R "Why this exists", "Closest prior and related work"; P "Statement of need" |
| 3b | Target population, intended purpose, intended users | MC "Intended use"; R; web app "About" |
| 3c | Known health inequalities | P "Statement of need" (higher prevalence in men, older adults, Mexican American and non-Hispanic Black adults and people with lower incomes, Eke et al. 2018); MC "Fairness" |
| 4 | Objectives (development and evaluation) | P "Statement of need"; R |
| 5a | Data sources, rationale, representativeness | R "Data, outcome and predictors"; MC "Data" — NHANES, a nationally representative survey |
| 5b | Dates of data collection | NHANES 2009–2010, 2011–2012 (development), 2013–2014 (evaluation) |
| 6a | Setting, centres | NHANES mobile examination centres across the US (CDC/NCHS) |
| 6b | Eligibility criteria | `training/clean.py` `build_cohort`: complete periodontal exam (OHDPDSTS = 1), age ≥ 30, ≥ 2 teeth measured; RES "Study population" |
| 6c | Treatments received | Not applicable (cross-sectional survey; past gum treatment is a self-reported predictor) |
| 7 | Data preparation, quality checks | `training/clean.py` (missing codes 7/9/77/99, site codes ≥ 99 = not assessable, SAS float rounding); case-definition code checked against CDC published prevalence (RES) |
| 8a | Outcome definition, assessment, rationale | CDC/AAP moderate or severe periodontitis from interproximal CAL and PD, third molars excluded (`training/clean.py` `classify`; tests in `tests/test_training.py`); current disease (no time horizon) |
| 8b | Outcome assessors | Calibrated NHANES dental examiners (CDC/NCHS protocol) |
| 8c | Blinding of outcome assessment | Examiners recorded measurements without a prediction model; questionnaire answers were collected separately |
| 9a | How predictors were chosen | Pre-specified: CDC/AAP self-report items (Eke & Dye 2009) plus established risk factors available to lay users; no data-driven selection |
| 9b | Predictor definitions and timing | `training/clean.py` `build_features`; `periorisk/questions.py` (NHANES wording and source variables); collected at the same NHANES visit as the examination |
| 9c | Predictor assessors | Self-report by participants (interviewer-administered in NHANES); BMI measured in NHANES, self-reported in the app |
| 10 | Study size | All eligible participants (7,050 development with 3,430 events; 3,610 evaluation with 1,547 events). No formal sample-size calculation; 3,430 events for 16 predictor degrees of freedom (≈ 214 per parameter) — a rough guide only for boosted trees. The evaluation set's 1,547 events exceed common recommendations (≥ 100–200 events) for validation studies |
| 11 | Missing data | Single imputation inside the model pipeline (median for numeric, most frequent for binary/categorical), fitted on training folds only. Missing per predictor: at most 1.8% (development) and 0.5% (evaluation); 3.2% and 1.5% of participants had any missing answer (RES "Missing answers") |
| 12a | Data use and partitioning | Temporal split by survey cycle (`training/train.py` `split`); 5-fold stratified cross-validation within development data |
| 12b | Predictor handling | Numeric: standardised; binary as 0/1; categorical one-hot (`training/train.py` `make_preprocessor`); age top-coded at 80 by NHANES and coding of every answer in `training/clean.py` `build_features` |
| 12c | Model type, tuning, internal validation | Logistic regression, random forest, histogram gradient boosting; grid search by 5-fold CV AUC; selection rule pre-specified; sigmoid calibration (`training/train.py`) |
| 12d | Heterogeneity across clusters | Survey strata/PSUs used for design-based CIs (`training/evaluate.py` `survey_metrics`); no centre-level clustering in NHANES public data |
| 12e | Performance measures and plots | AUC, sensitivity, specificity, PPV, NPV, Brier, scaled Brier, O/E, calibration intercept and slope, calibration plot, decision curve, permutation importance (`training/evaluate.py`) |
| 12f | Model updating | None to the deployed model; prior-shift recalibration demonstrated (RES "Transportability") and offered as `periorisk.recalibrate` |
| 12g | How predictions were calculated for evaluation | Fitted pipeline applied once to 2013–2014 data (`training/run.py`); the portable export reproduces these predictions (`tests/test_portable_model.py`) |
| 13 | Class imbalance | Not applicable — outcome prevalence 43–49%; no resampling or re-weighting |
| 14 | Fairness approach | Race/ethnicity excluded as a predictor; AUC, prevalence, mean prediction and O/E reported by sex, age group, race/ethnicity, education and diabetes (RES) |
| 15 | Model output and thresholds | Calibrated probability; threshold 0.470 by Youden's J on out-of-fold training predictions; app risk bands < 30%, 30–60%, ≥ 60% with observed rates per band (MC) |
| 16 | Differences between development and evaluation data | Later survey cycle; prevalence 48.7% vs 42.9% (RES; metadata `prevalence_train`, `prevalence_test`) |
| 17 | Ethics | NHANES protocols approved by the NCHS Research Ethics Review Board with written informed consent; this is a secondary analysis of public, de-identified data |
| 18a | Funding | None |
| 18b | Conflicts of interest | None declared |
| 18c | Protocol | No separate protocol; the full analysis is the versioned code in this repository |
| 18d | Registration | Not registered |
| 18e | Data availability | NHANES public data (downloaded by `training/download.py`); analytic dataset in `data/processed/` |
| 18f | Code availability | This repository (MIT), archived on Zenodo |
| 19 | Patient and public involvement | None in development; the questionnaire wording is the public NHANES wording |
| 20a | Participant flow, outcome counts | RES "Study population": 14,071 → 11,753 complete exams (210 partial, 2,108 not done) → 10,660 (1,070 edentulous, 23 with one tooth measured); 3,430 and 1,547 events |
| 20b | Participant characteristics | RES "Participant characteristics" table; subgroup table by sex, age, race/ethnicity, education, diabetes |
| 20c | Development vs evaluation distributions | RES "Participant characteristics" (development and evaluation side by side) and "Missing answers" |
| 21 | Participants and events per analysis | RES (n and prevalence per subgroup and band) |
| 22 | Full model specification, access | `periorisk/data/model.json` (every parameter); `models/model.joblib`; no restrictions (MIT) |
| 23a | Performance with CIs, including subgroups | RES "Final model on the test set", "Population-representative performance", "Performance in subgroups" |
| 23b | Heterogeneity across clusters | Survey-weighted, design-based estimates (RES) |
| 24 | Model updating results | No update to the deployed model. RES "Transportability" demonstrates prior-shift recalibration: to the test cycle's own prevalence O/E 0.903 → 0.992 (by construction), and to the survey-weighted prevalence weighted O/E 0.80 → 1.06 (over-correction) |
| 25 | Interpretation | P "Results and interpretation" |
| 26 | Limitations | R "Limitations"; MC "Known limitations"; web app "Limits you should know" |
| 27a | Poor-quality or missing inputs | Unanswered or "not sure" answers are imputed and listed; implausible height/weight/BMI and out-of-scope users (age < 30, < 2 natural teeth) are refused (`periorisk/__init__.py`, `docs/periorisk.js`) |
| 27b | User interaction and expertise | Lay users answer 14 questions; no clinical expertise needed; result presented as band, observed rate and next step |
| 27c | Future research | P "Future work": external validation outside the US and in dental clinics, validated translations, prospective evaluation |
