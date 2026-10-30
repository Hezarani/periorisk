# Changelog

## 1.0.0 — 2026-10-29

First public release. The trained model is unchanged from the author's original pipeline (identical
predictions for all 10,660 participants); this release reviews, fixes and extends it.

### Added
- **Private web app** on GitHub Pages (`docs/`): the model runs in the browser, no answers leave the
  device, no cookies or analytics; per-answer explanation ("what shaped your estimate"), next steps,
  printable summary for the dentist, metric/US units, dark mode, keyboard and screen-reader support.
- **Portable model** (`periorisk/data/model.json`) exported from scikit-learn and verified on every
  participant; pure-Python and JavaScript engines that reproduce scikit-learn to within 1e-12.
- **`periorisk` Python package** (no dependencies): `predict()`, `predict_features()`,
  `recalibrate()` (prior-shift adjustment to a known prevalence), `questionnaire()`, `model_info()`,
  and a command-line tool (`periorisk predict | batch | questions | info`).
- **Evaluation**: calibration slope, calibration-in-the-large and observed/expected ratio with
  bootstrap CIs; NPV; scaled Brier; survey-weighted performance with design-based (Rao–Wu) CIs;
  decision-curve table; subgroup calibration (mean predicted, O/E) and education subgroup;
  transportability demonstration (and its limits); each candidate model scored at its own
  threshold; participant characteristics and missing-data tables; exclusion reasons in the cohort
  flow; sensitivity/specificity of the app's band cut-offs; a sensitivity analysis of the eighth
  CDC/AAP self-report item (OHQ835) on the development data.
- TRIPOD+AI checklist, JOSS-style paper, Zenodo metadata, ORCID in CITATION.cff, test suite (97
  checks) and GitHub Actions (tests on Python 3.10–3.14; PyPI Trusted Publishing on release).

### Fixed
- Every answer was pre-filled in the old app, so a result appeared without answering anything; now
  every question must be answered (or "Not sure" / "Prefer not to say").
- No check for natural teeth: the model only applies to people with at least two; people without
  natural teeth, and people under 30, now get an explanation instead of a number.
- The smoking question now matches the NHANES definition (100 cigarettes in a lifetime).
- Implausible height/weight (e.g. BMI 300) was accepted; now refused with a message.
- Privacy text did not say that results were logged even without consent; there is now no logging at all.
- README results table mixed calibrated and uncalibrated numbers and scored all candidates at the
  selected model's threshold.
- Placeholder repository URL in the app and missing ORCID in the citation file.
- The saved model only loaded with one exact scikit-learn version; the app and package no longer need scikit-learn.
- Download size in the docs (≈95 MB, not 60 MB); downloads are now written atomically.

### Removed
- Streamlit app and prediction-log database (replaced by the private static web app).
