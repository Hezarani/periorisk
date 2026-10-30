# Contributing to PerioRisk

Thank you for helping. Bug reports, validation results from other populations and questionnaire
translations are especially welcome.

- **Report a problem:** open an issue at https://github.com/Hezarani/periorisk/issues with what you
  did, what you expected and what happened. Please never include anyone's health information.
- **External validation:** if you evaluate PerioRisk on another dataset, please open an issue — even
  poor results are useful. `periorisk batch` scores a CSV file; `periorisk.recalibrate` adjusts to a
  known prevalence.
- **Code changes:** run `python -m pytest` before opening a pull request. If you change the model or
  the training pipeline, re-run `python -m training` (Python 3.12 with `requirements-training.txt`)
  so that `periorisk/data/model.json`, `docs/model.js`, the reports and the test fixture are
  regenerated together, and bump the model version.
- **Web app:** `docs/periorisk.js` must stay a line-by-line port of `periorisk/_model.py` and
  `periorisk/__init__.py`; `tests/test_web_engine.py` checks that they agree.

By contributing you agree that your contribution is licensed under the MIT licence.
