---
title: 'PerioRisk: open, temporally validated periodontitis risk screening from self-reported questions'
tags:
  - Python
  - JavaScript
  - periodontitis
  - dentistry
  - clinical prediction model
  - NHANES
  - dental public health
authors:
  - name:
      given-names: Hossein
      surname: Boustani Hezarani
    orcid: 0009-0007-8481-0498
    affiliation: 1
affiliations:
  - name: Independent researcher, Ankara, Türkiye
    index: 1
date: 29 October 2026
bibliography: paper.bib
---

# Summary

PerioRisk estimates the probability that an adult currently has moderate or severe periodontitis,
as defined by the CDC/AAP surveillance case definition [@page2007; @eke2012], from 14 questions
that anyone can answer without a dental examination: age, sex, height and weight, education,
smoking, diabetes, and the self-reported oral-health items developed for periodontitis surveillance
[@eke2009; @eke2013]. A calibrated gradient-boosting model was developed on the 2009–2012 cycles of
the US National Health and Nutrition Examination Survey (NHANES) and tested once on the later
2013–2014 cycle (n = 3,610), where it reached an AUC of 0.781 (95% CI 0.766–0.796) with a
calibration slope of 0.91 and a survey-weighted AUC of 0.785. PerioRisk is a dependency-free Python
package and command-line tool, a private web application that runs entirely in the browser, and a
single-command pipeline that rebuilds the published model exactly from public CDC data.

# Statement of need

Periodontitis affects about 42% of dentate US adults aged 30 and over, 7.8% severely, with higher
prevalence in men, older adults, Mexican American and non-Hispanic Black adults, and people with
lower incomes [@eke2018]. It is often painless until late, and the reference standard —
full-mouth probing — needs a dental visit. Self-reported measures can prompt people to seek that examination and let
researchers estimate periodontitis where no examination is possible [@eke2013; @eke2016]. The CDC
programme has published a self-report scoring algorithm developed and validated on NHANES 2009–2012
[@eke2025], while a 2025 meta-analysis concluded that self-report questionnaires show potential for
monitoring periodontitis but that individual items vary widely in accuracy (pooled sensitivity
17–82%) and that cross-cultural validity is limited [@bi2025].

The objectives were to develop a self-report model of moderate/severe periodontitis, evaluate it on
a later survey cycle with full reporting of discrimination, calibration, clinical usefulness and
fairness, and make it usable — by the public privately, and by researchers reproducibly.

# State of the field

Machine-learning models of periodontitis built on NHANES are common [@montero2019; @bashir2022;
@vu2025], and screening tools from other countries have been externally validated in medical and
dental settings [@verhulst2019; @viana2025]. Many NHANES models are evaluated on a random split of
the same data, some include laboratory variables that a lay user cannot supply, discrimination is
often the only performance measure reported, and fitted models are rarely released in a usable form.
PerioRisk does not aim to out-perform existing self-report models such as the CDC scoring algorithm
[@eke2025]; its discrimination is in the same range. Clinical prediction research has repeatedly
shown that machine learning seldom outperforms logistic regression on tabular data
[@christodoulou2019] and that calibration is as important as discrimination [@vancalster2019]. PerioRisk therefore focuses on transparent temporal
validation on a later survey cycle, calibration (including survey-weighted, population-level
calibration), and an exactly reproducible, reusable model rather than on a new algorithm, and
reports following TRIPOD+AI [@collins2024].

# Software design

The **training pipeline** (`python -m training`) downloads 18 NHANES files from the CDC, applies the
CDC/AAP case definition to every interproximal site of every tooth except third molars, builds the
13 model inputs, stores each step in a SQLite database, compares logistic regression, random forest
[@breiman2001] and histogram gradient boosting [@friedman2001] by five-fold cross-validated AUC with
scikit-learn [@pedregosa2011], calibrates the selected model with five-fold sigmoid scaling, picks a
screening threshold by Youden's index [@youden1950] on out-of-fold training predictions, and
evaluates the model once on 2013–2014. Re-running it reproduces the published model byte for byte.

The **portable model** removes the usual dependence of a saved model on one scikit-learn and Python
version. The pipeline exports every imputation value, scaling constant, tree node and calibration
parameter of the five calibrated ensembles to a 0.6 MB JSON file and refuses to write it unless an
independent re-implementation reproduces scikit-learn's probabilities for all 10,660 participants.
The same file is evaluated by a pure-Python engine (the `periorisk` package, no dependencies) and a
JavaScript engine (the web app); the test suite checks both against scikit-learn's predictions to
within $10^{-12}$ on every commit, together with the API, the command line and the case definition.

The **package** (`periorisk.predict`) validates answers, refuses people outside the development
population (under 30, fewer than two natural teeth, implausible body measurements), imputes
unanswered questions, and returns the probability, risk band, the observed disease rate in that band
in the test cycle, and a per-answer explanation: how much the estimate changes when one answer is
replaced by a typical answer (the most common answer, or the median for numbers).
`periorisk.recalibrate` applies a prior-shift adjustment [@saerens2002] for populations with a
known, different prevalence. The **web app** asks the questions in the NHANES wording (two smoking
questions are combined and education levels grouped), shows the estimate, the band, the
explanation and next steps, prints a summary for the dentist, and makes no network connections.

# Results and interpretation

Of 14,071 participants in the periodontal examination files, 10,660 adults aged 30+ with a
complete examination and at least two teeth formed the analytic sample (7,050 development, 3,610
test). The case-definition code reproduces the CDC's published survey-weighted prevalence (42.3% vs
42.2% total; 7.8% vs 7.8% severe) [@eke2018]. The three model types performed almost identically on
the test cycle (AUC 0.778–0.782). The selected calibrated gradient-boosting model reached an AUC of
0.781 (95% CI 0.766–0.796), sensitivity 0.734 and specificity 0.680 at the pre-specified threshold,
calibration slope 0.91 (0.84–0.98) and observed/expected ratio 0.90 (0.87–0.93), with positive
net benefit over referring everyone or no one for threshold probabilities between about 10% and
80% [@vickers2006]. Survey-weighted evaluation with design-based confidence intervals [@rao1992] gave an AUC of 0.785 (0.760–0.807) and
an observed/expected ratio of 0.80 (0.72–0.88) for the US population: the model over-estimates risk
because prevalence fell between cycles and NHANES over-samples higher-risk groups. A prior-shift
adjustment to the test cycle's own prevalence restored calibration-in-the-large (by construction),
but adjusting to the survey-weighted population prevalence over-corrected (O/E 1.06), showing that a
prevalence update alone cannot correct miscalibration caused by a different case mix. Used as
screening rules, estimates of 30% or more flagged 87% of people with the disease (specificity 47%).
Test AUC ranged from 0.71 to 0.80 across sex, age, race/ethnicity, education and diabetes subgroups; race/ethnicity
is not a model input. The tool is meant to prompt an examination, not to replace one.

# Future work

External validation outside the United States and in dental clinics, validated translations of the
questionnaire, prospective evaluation of whether the app leads people to have a periodontal
examination, a head-to-head comparison with the CDC self-report scoring algorithm [@eke2025] on the
same data, and adding the eighth CDC/AAP self-report item ("Do you think you might have gum
disease?"), which raised cross-validated AUC slightly (0.803 to 0.807) in a sensitivity analysis.

# Research impact statement

PerioRisk gives dental public-health researchers a reproducible, citable reference model with a
complete TRIPOD+AI checklist, an analytic dataset and a portable model format that other
prediction-model projects can reuse; and it gives the public a private, plain-language way to judge
whether to ask for a periodontal examination.

# AI usage disclosure

Generative AI was used extensively. Anthropic's Claude (Opus model family, in the Cowork mode of the
Claude desktop application) reviewed, refactored and extended the code, tests and documentation
under the author's direction, searched and summarised the literature, and ran automated audits of
the software. The author conceived the project, defined its scope and requirements, made the design
decisions, reviewed the outputs and is responsible for the content. Correctness was checked by an
automated test suite (including scikit-learn–Python–JavaScript parity tests), by separate AI-run
audits whose findings were fixed, and by verifying every DOI against its registry record.

# Acknowledgements

PerioRisk uses data collected by the CDC National Center for Health Statistics and builds on
scikit-learn, NumPy, pandas and matplotlib. This work received no specific funding. The author
declares no competing interests.

# References
