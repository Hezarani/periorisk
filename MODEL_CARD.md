# Model card — PerioRisk v1.0.0

## Model details
- **What it predicts:** the probability that a person aged 30+ with at least two natural teeth currently has **moderate or severe
  periodontitis** by the CDC/AAP surveillance case definition (Page & Eke 2007; Eke et al. 2012).
- **Inputs:** 13 values from 14 self-reported questions (age, sex, body-mass index from height and
  weight, education, cigarette smoking, diabetes, self-rated teeth and gums, past gum treatment,
  loose teeth, told of bone loss, a tooth that doesn't look right, days of interdental cleaning,
  days of mouthwash use for dental problems). Wording follows NHANES 2009–2014.
- **Algorithm:** histogram gradient boosting (scikit-learn 1.8.0; 300 iterations, learning rate
  0.03, 7 leaves) inside a pipeline with median / most-frequent imputation, standardisation and
  one-hot encoding, calibrated by 5-fold sigmoid (Platt) scaling — i.e. five calibrated ensembles
  whose probabilities are averaged. Selected over logistic regression and random forest by 5-fold
  cross-validated AUC on the training data (0.803 vs 0.800 and 0.800).
- **Screening threshold:** 0.470 (Youden's J on out-of-fold training predictions). The app shows
  three risk bands instead: below 30%, 30–60%, 60% or more.
- **Portable form:** `periorisk/data/model.json` holds every imputation value, scaling constant,
  tree node and calibration parameter; the Python package and the web app evaluate it without
  scikit-learn and reproduce scikit-learn's probabilities to within 1e-12.
- **Author:** Hossein Boustani Hezarani, DDS. MIT licence. Version 1.0.0 (model and package).

## Intended use
- Education, health literacy and self-screening: prompting adults aged 30+ to have a periodontal
  examination.
- Research: estimating periodontitis risk in surveys or cohorts that have the same questions but
  no periodontal examination; teaching and benchmarking of clinical prediction modelling.

## Not intended for
- Diagnosis, treatment decisions, or replacing a clinical periodontal examination.
- People under 30 or with fewer than two natural teeth (outside the development population).
- Use as a certified medical device, or for insurance, employment or other decisions about individuals.

## Data
NHANES 2009–2014 (CDC/NCHS, public domain). Adults 30+ with a complete full-mouth periodontal
examination and at least two teeth measured: n = 10,660. **Development:** 2009–2012 (n = 7,050,
48.7% with moderate/severe periodontitis). **Temporal validation:** 2013–2014 (n = 3,610, 42.9%).
Cohort flow, coding of every variable and missing-data handling: `training/clean.py` and
`reports/results.md`.

## Performance (2013–2014 temporal test set; 95% CIs by 1,000 bootstrap resamples)
| Measure | Estimate (95% CI) |
|---|---|
| AUC | 0.781 (0.766–0.796) |
| Sensitivity / specificity at 0.470 | 0.734 (0.712–0.755) / 0.680 (0.658–0.702) |
| PPV / NPV | 0.633 (0.609–0.657) / 0.773 (0.753–0.792) |
| Brier score (scaled Brier) | 0.190 (0.222) |
| Calibration slope | 0.909 (0.839–0.979) |
| Calibration-in-the-large | −0.254 (−0.332 to −0.171) |
| Observed / expected ratio | 0.903 (0.874–0.935) |
| Band rule ≥ 30%: sensitivity / specificity | 0.873 / 0.466 |
| Band rule ≥ 60%: sensitivity / specificity | 0.604 / 0.804 |

Bootstrap CIs above ignore the survey's cluster design; the design-based CIs below are wider.

**Population-representative (survey-weighted, design-based CIs):** AUC 0.785 (0.760–0.807);
sensitivity 0.700; specificity 0.725; observed/expected 0.799 (0.723–0.882); calibration slope
0.972 (0.866–1.074).

## Fairness
Race/ethnicity is not an input. Test AUC by subgroup: sex 0.76–0.78; age 30–44 0.75, 45–64 0.80,
65+ 0.73; race/ethnicity 0.75–0.79; education 0.71–0.77; diabetes 0.72–0.78. The model
over-estimates risk most for non-Hispanic White participants (O/E 0.76) and people aged 65+
(O/E 0.83), and slightly under-estimates for ages 30–44 (O/E 1.09) and non-Hispanic Black and
Mexican American participants (O/E 1.06 and 1.04). Full table: `reports/results.md`.

## Known limitations
- US population only; transport to other countries is unverified. If the local prevalence of
  moderate/severe periodontitis (CDC/AAP) is known, `periorisk.recalibrate` adjusts calibration-in-the-large.
- Cross-sectional data: it estimates current disease, not incidence or progression.
- Prevalence fell between survey cycles, so probabilities run about 11% high on average in the test
  cycle (mean predicted 47.4% vs observed 42.9%) and about 25% high for the weighted US population
  (43.0% vs 34.3%). A prevalence adjustment alone over-corrects the latter (O/E 0.80 → 1.06) because
  that gap comes mostly from NHANES's sampling design. The app therefore leads with the risk band and
  the observed rate in that band.
- Self-reported predictors; BMI was measured in NHANES but is self-reported in the app.
- Development analyses are unweighted; participants with partial periodontal examinations were excluded.
- The eighth CDC/AAP self-report item ("Do you think you might have gum disease?", OHQ835) was not
  pre-specified. In a sensitivity analysis on the development data (5-fold cross-validation) it raised
  AUC slightly (0.803 → 0.807); v1.0.0 keeps the pre-specified set and the item is a candidate for a
  future version.
- Answers to the 14 questions were missing for 3.2% of development and 1.5% of evaluation
  participants (at most 1.8% for any single question) and were imputed.
- Not evaluated prospectively, in dental clinics, or with translated questionnaires.

## Privacy
The web app runs entirely in the browser: no answers are uploaded, stored, logged or tracked, and
the page's Content-Security-Policy blocks all network connections. The Python package does no I/O
beyond reading its bundled model file.
