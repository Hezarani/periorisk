"""Run the whole project from raw data to a saved, exported model:

    python -m training

Ingestion -> Cleaning -> Case definition -> Feature engineering -> Database -> Training ->
Evaluation -> Saving the model, the report and the portable model used by the package and app.
"""
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import sklearn
from sqlalchemy import text

from . import evaluate as ev
from . import export
from .clean import build_cohort, load_raw
from .config import (DOCS_DIR, FEATURES, FIG_DIR, FIXTURE_DIR, MODEL_DATE, MODEL_DIR, MODEL_VERSION,
                     PACKAGE_DATA_DIR, PROCESSED_DIR, REPORT_DIR, RISK_BANDS, ROOT, TARGET)
from .db import get_engine, run_sql_file
from .download import download_all
from .train import added_predictor_check, finalize, own_thresholds, split, train_models


def weighted_prevalence(df, col_mask):
    return float(np.average(col_mask.astype(float), weights=df["exam_weight"]))


def main():
    for d in (PROCESSED_DIR, MODEL_DIR, FIG_DIR, PACKAGE_DATA_DIR, FIXTURE_DIR, DOCS_DIR / "img"):
        d.mkdir(parents=True, exist_ok=True)

    print("1/7  Downloading NHANES 2009-2014 from the CDC (skipped if already done)")
    download_all()

    print("2/7  Cleaning data and applying the CDC/AAP case definition")
    perio, covs = load_raw()
    clean_cohort, features, flow = build_cohort(perio, covs)
    print(f"     {flow}")

    print("3/7  Saving raw, clean and feature tables to the database")
    engine = get_engine()
    with engine.begin() as conn:  # views must go before their tables are replaced
        conn.execute(text("DROP VIEW IF EXISTS v_prevalence_by_cycle"))
        conn.execute(text("DROP VIEW IF EXISTS v_case_definition_counts"))
    perio.to_sql("raw_periodontal", engine, if_exists="replace", index=False, chunksize=500)
    covs.to_sql("raw_covariates", engine, if_exists="replace", index=False, chunksize=2000)
    clean_cohort.to_sql("clean_cohort", engine, if_exists="replace", index=False)
    features.to_sql("features", engine, if_exists="replace", index=False)
    run_sql_file(engine, ROOT / "sql" / "views.sql")
    features.to_csv(PROCESSED_DIR / "analytic_dataset.csv", index=False)

    cats = clean_cohort.merge(features[["SEQN", "exam_weight"]], on="SEQN")
    prevalence = {k: weighted_prevalence(cats, cats["perio_category"].isin(v)) for k, v in {
        "total_periodontitis": ["mild", "moderate", "severe"],
        "moderate_or_severe": ["moderate", "severe"],
        "severe": ["severe"]}.items()}
    print(f"     survey-weighted prevalence: { {k: round(v * 100, 1) for k, v in prevalence.items()} }")

    print("4/7  Training models on 2009-2012 (5-fold cross-validation)")
    train, test = split(features)
    fitted = train_models(train)
    best_name = max(fitted, key=lambda k: fitted[k]["cv_auc"])
    print(f"     best model by cross-validated AUC: {best_name}")
    thresholds = own_thresholds(fitted, train)
    final, threshold, _ = finalize(best_name, fitted[best_name]["estimator"], train)

    print("5/7  Evaluating on the untouched 2013-2014 data")
    y = test[TARGET]
    probs = {n: f["estimator"].predict_proba(test[FEATURES])[:, 1] for n, f in fitted.items()}
    p_final = final.predict_proba(test[FEATURES])[:, 1]
    all_models = {n: ev.classification_metrics(y, p, thresholds[n]) | {
        "cv_auc": fitted[n]["cv_auc"], "best_params": fitted[n]["best_params"]} for n, p in probs.items()}
    final_metrics = ev.classification_metrics(y, p_final, threshold)
    final_ci = ev.bootstrap_ci(y, p_final, threshold)
    survey = ev.survey_metrics(y, p_final, threshold, test["exam_weight"], test["psu"], test["stratum"])
    subgroups = ev.subgroup_table(test, p_final)
    bands = ev.band_table(y, p_final)
    train_prev = float(train[TARGET].mean())
    shifted = ev.prior_shift(p_final, train_prev, float(y.mean()))
    transport = {"source_prevalence": train_prev, "target_prevalence": float(y.mean()),
                 "before": ev.calibration_stats(y, p_final) | {"auc": float(ev.roc_auc_score(y, p_final))},
                 "after": ev.calibration_stats(y, shifted) | {"auc": float(ev.roc_auc_score(y, shifted))}}
    pop_prev = survey["prevalence"]["estimate"]
    w = test["exam_weight"].to_numpy()
    transport["population"] = {
        "target_prevalence": pop_prev,
        "weighted_before": ev.calibration_stats(y, p_final, w),
        "weighted_after": ev.calibration_stats(y, ev.prior_shift(p_final, train_prev, pop_prev), w)}
    band_cuts = {f"{int(round(c * 100))}": {k: float(v) for k, v in ev.threshold_metrics(y, p_final, c).items()}
                 for c in (0.30, 0.60)}
    flow["events_development"] = int(train[TARGET].sum())
    flow["events_evaluation"] = int(y.sum())
    table1 = ev.characteristics(train, test)
    missing = ev.missingness(train, test)
    print("     sensitivity analysis: adding OHQ835 (cross-validation on the training data only)")
    added = added_predictor_check(fitted[best_name]["estimator"], train, "think_gum_disease")
    nb_t = [0.2, 0.3, 0.4, 0.5]
    nb_model, nb_all = ev.net_benefit(y.to_numpy(), p_final, nb_t)
    decision = [{"threshold": t, "model": float(m), "refer_everyone": float(a)}
                for t, m, a in zip(nb_t, nb_model, nb_all)]

    ev.plot_roc(y, probs | {"Final model": p_final}, FIG_DIR / "roc_curves.png")
    ev.plot_calibration(y, p_final, FIG_DIR / "calibration.png")
    ev.plot_confusion(final_metrics, FIG_DIR / "confusion_matrix.png")
    ev.plot_decision_curve(y, p_final, FIG_DIR / "decision_curve.png")
    importance = ev.plot_importance(final, test, FIG_DIR / "feature_importance.png")

    results = pd.DataFrame([{"model": n, **{k: v for k, v in m.items() if k != "best_params"}}
                            for n, m in all_models.items()]
                           + [{"model": "Final model (calibrated)", **final_metrics}])
    results.to_sql("model_results", engine, if_exists="replace", index=False)

    print("6/7  Saving model and report")
    joblib.dump(final, MODEL_DIR / "model.joblib", compress=3)
    band_info = [{"band": b, "from": lo, "to": min(hi, 1.0),
                  "test_n": int(r["n"]), "test_observed": float(r["observed_prevalence"]),
                  "test_mean_predicted": float(r["mean_predicted"])}
                 for (lo, hi, b), (_, r) in zip(RISK_BANDS, bands.iterrows())]
    meta = {
        "model_version": MODEL_VERSION,
        "model_date": MODEL_DATE,
        "selected_model": best_name,
        "calibrated": best_name != "Logistic Regression",
        "threshold": threshold,
        "risk_bands": band_info,
        "features": FEATURES,
        "target": "Moderate or severe periodontitis (CDC/AAP case definition)",
        "train_data": "NHANES 2009-2010 and 2011-2012",
        "test_data": "NHANES 2013-2014 (temporal hold-out)",
        "n_train": int(len(train)), "n_test": int(len(test)),
        "prevalence_train": train_prev, "prevalence_test": float(y.mean()),
        "scikit_learn_version": sklearn.__version__,
        "test_metrics": final_metrics,
        "test_metrics_ci": final_ci,
        "survey_weighted_test_metrics": survey,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (REPORT_DIR / "metrics.json").write_text(json.dumps({
        "cohort_flow": flow, "weighted_prevalence": prevalence, "final_model": best_name,
        "threshold": threshold, "final_test_metrics": final_metrics, "final_test_metrics_ci": final_ci,
        "survey_weighted_test_metrics": survey, "all_models": all_models,
        "risk_bands_test": bands.to_dict("records"), "subgroups": subgroups.to_dict("records"),
        "transportability_prior_shift": transport, "net_benefit": decision, "band_cutoffs": band_cuts,
        "characteristics": table1, "missing_data_pct": missing, "sensitivity_added_predictor": added,
        "permutation_importance": importance.to_dict("records")}, indent=2, default=float) + "\n")
    write_report(flow, prevalence, all_models, best_name, final_metrics, final_ci, survey, threshold,
                 subgroups, bands, importance, transport, decision, len(train), len(test),
                 band_cuts, table1, missing, added)

    print("7/7  Exporting the portable model for the Python package and the web app")
    portable = export.export_model(final, {
        "model_version": MODEL_VERSION,
        "model_date": MODEL_DATE,
        "exported_from": f"scikit-learn {sklearn.__version__}",
        "outcome": meta["target"],
        "population": "US adults aged 30+ with at least two natural teeth (NHANES 2009-2014)",
        "threshold": threshold,
        "train_prevalence": train_prev,
        "age_range": [30, 80],
        "risk_bands": band_info,
        "performance": {
            "test_auc": final_metrics["auc"], "test_auc_ci": [final_metrics["auc_ci_low"], final_metrics["auc_ci_high"]],
            "sensitivity": final_metrics["recall_sensitivity"], "specificity": final_metrics["specificity"],
            "calibration_slope": final_metrics["calibration_slope"], "o_e_ratio": final_metrics["o_e_ratio"],
            "ppv": final_metrics["precision_ppv"], "npv": final_metrics["npv"],
            "survey_weighted_auc": survey["auc"]["estimate"], "survey_weighted_o_e_ratio": survey["o_e_ratio"]["estimate"],
            "sensitivity_at_30": band_cuts["30"]["recall_sensitivity"], "specificity_at_30": band_cuts["30"]["specificity"],
            "sensitivity_at_60": band_cuts["60"]["recall_sensitivity"], "specificity_at_60": band_cuts["60"]["specificity"],
            "n_train": int(len(train)), "n_test": int(len(test)),
        },
    })
    gap = export.verify(portable, final, features)
    print(f"     largest difference between the portable model and scikit-learn: {gap:.2e}")
    if gap > 1e-12:
        raise RuntimeError("the portable model does not reproduce scikit-learn")
    export.write(portable, PACKAGE_DATA_DIR / "model.json", DOCS_DIR / "model.js")
    every = final.predict_proba(features[FEATURES])[:, 1]
    pd.DataFrame({"SEQN": features["SEQN"].astype(int), "probability": [repr(float(v)) for v in every]}) \
        .to_csv(FIXTURE_DIR / "sklearn_probabilities.csv", index=False)
    for name in ("calibration.png", "roc_curves.png", "decision_curve.png"):
        shutil.copy(FIG_DIR / name, DOCS_DIR / "img" / name)
    print(f"Done. Test AUC = {final_metrics['auc']:.3f} "
          f"(95% CI {final_metrics['auc_ci_low']:.3f}-{final_metrics['auc_ci_high']:.3f})")


def _pct(x):
    return f"{x * 100:.1f}%"


def _ci(pair, pct=False, digits=3):
    f = (lambda v: f"{v * 100:.1f}") if pct else (lambda v: f"{v:.{digits}f}")
    return f"{f(pair[0])}-{f(pair[1])}"


def write_report(flow, prev, all_models, best, m, ci, survey, thr, subgroups, bands, imp,
                 transport, decision, n_train, n_test, band_cuts, table1, missing, added):
    lines = [
        "# Results report", "",
        "_This file is written automatically by `python -m training`. Do not edit by hand._", "",
        "## Study population", "",
        f"- Participants in the periodontal exam files (aged 30+): {flow['periodontal_file_rows']:,}",
        f"- Excluded, exam partial: {flow['excluded_exam_partial']:,}; exam not done: {flow['excluded_exam_not_done']:,}",
        f"- With a complete periodontal exam: {flow['exam_complete']:,} (all aged 30 or older: {flow['age_30_plus']:,})",
        f"- Excluded, no teeth measured (edentulous): {flow['excluded_no_teeth_measured']:,}; "
        f"only one tooth measured: {flow['excluded_one_tooth_measured']:,}",
        f"- With at least 2 teeth measured (analytic sample): {flow['two_or_more_teeth_measured']:,}",
        f"- Development set (2009-2012): {n_train:,}, of whom {flow['events_development']:,} had moderate or severe "
        f"periodontitis / Evaluation set (2013-2014): {n_test:,}, of whom {flow['events_evaluation']:,}", "",
        "Survey-weighted prevalence in the analytic sample (check of the case-definition code against "
        "the CDC's published NHANES 2009-2014 estimates of 42.2% and 7.8%, Eke et al. 2018):", "",
        f"- Any periodontitis: {_pct(prev['total_periodontitis'])}",
        f"- Moderate or severe: {_pct(prev['moderate_or_severe'])}",
        f"- Severe: {_pct(prev['severe'])}", "",
        "## Participant characteristics (unweighted)", "",
        "| Characteristic | Development 2009-2012 | Evaluation 2013-2014 |", "|---|---|---|",
    ] + [f"| {r['characteristic']} | {r['development']} | {r['evaluation']} |" for r in table1] + [
        "", "### Missing answers (filled by the model's imputation)", "",
        "| Model input | Development, % missing | Evaluation, % missing |", "|---|---|---|",
    ] + [f"| {r['feature'].replace('_', ' ')} | {r['development_pct']:.2f} | {r['evaluation_pct']:.2f} |" for r in missing] + [
        "",
        "## Model comparison (test set 2013-2014)", "",
        "Each candidate is scored at its own screening threshold (Youden's J on its out-of-fold "
        "training predictions). The candidates are the uncalibrated models refit on all training data.", "",
        "| Model | CV AUC (train) | Test AUC (95% CI) | Threshold | Sensitivity | Specificity | Brier |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in all_models.items():
        lines.append(f"| {name} | {r['cv_auc']:.3f} | {r['auc']:.3f} ({r['auc_ci_low']:.3f}-"
                     f"{r['auc_ci_high']:.3f}) | {r['threshold']:.3f} | {r['recall_sensitivity']:.3f} | "
                     f"{r['specificity']:.3f} | {r['brier']:.3f} |")
    lines += [
        f"| **Final model: {best} + sigmoid calibration** | - | **{m['auc']:.3f} ({m['auc_ci_low']:.3f}-"
        f"{m['auc_ci_high']:.3f})** | {thr:.3f} | {m['recall_sensitivity']:.3f} | {m['specificity']:.3f} | "
        f"{m['brier']:.3f} |", "",
        f"Selected model: **{best}** (highest cross-validated AUC on the training data)"
        + (", calibrated with 5-fold sigmoid (Platt) scaling." if best != "Logistic Regression" else "."),
        f"Screening threshold chosen on training data only (Youden's J): **{thr:.3f}**", "",
        "## Final model on the test set", "",
        "95% confidence intervals: 1,000 participant-level bootstrap resamples. These ignore the NHANES "
        "cluster design and are therefore narrower than design-based intervals (compare the survey-weighted "
        "section below).", "",
        "| Measure | Estimate | 95% CI |", "|---|---|---|",
        f"| AUC (discrimination) | {m['auc']:.3f} | {m['auc_ci_low']:.3f}-{m['auc_ci_high']:.3f} |",
        f"| Sensitivity | {m['recall_sensitivity']:.3f} | {_ci(ci['recall_sensitivity'])} |",
        f"| Specificity | {m['specificity']:.3f} | {_ci(ci['specificity'])} |",
        f"| Positive predictive value | {m['precision_ppv']:.3f} | {_ci(ci['precision_ppv'])} |",
        f"| Negative predictive value | {m['npv']:.3f} | {_ci(ci['npv'])} |",
        f"| Brier score (lower is better) | {m['brier']:.3f} | {_ci(ci['brier'])} |",
        f"| Scaled Brier score | {m['scaled_brier']:.3f} | |",
        f"| Observed / expected ratio (ideal 1) | {m['o_e_ratio']:.3f} | {_ci(ci['o_e_ratio'])} |",
        f"| Calibration-in-the-large (ideal 0) | {m['calibration_intercept']:.3f} | {_ci(ci['calibration_intercept'])} |",
        f"| Calibration slope (ideal 1) | {m['calibration_slope']:.3f} | {_ci(ci['calibration_slope'])} |",
        f"| Accuracy | {m['accuracy']:.3f} | |",
        f"| F1 score | {m['f1']:.3f} | |", "",
        "## Population-representative performance (survey-weighted)", "",
        "Weighted with the NHANES examination weights, so the estimates describe the US adult "
        "population aged 30+ with teeth rather than the (over-sampled) survey sample. 95% CIs: "
        f"Rao-Wu rescaling bootstrap over {survey['n_strata']} strata ({survey['n_replicates']} replicates).", "",
        "| Measure | Estimate | 95% CI |", "|---|---|---|",
    ]
    names = {"prevalence": "Prevalence of moderate/severe periodontitis", "auc": "AUC",
             "recall_sensitivity": "Sensitivity", "specificity": "Specificity",
             "precision_ppv": "Positive predictive value", "npv": "Negative predictive value",
             "accuracy": "Accuracy", "brier": "Brier score", "o_e_ratio": "Observed / expected ratio",
             "calibration_intercept": "Calibration-in-the-large", "calibration_slope": "Calibration slope"}
    for k, label in names.items():
        s = survey[k]
        lines.append(f"| {label} | {s['estimate']:.3f} | {_ci(s['ci'])} |")
    lines += [
        "", "## Risk bands used in the app", "",
        "| Band | Predicted probability | People in test set | Mean predicted | Actually had periodontitis |",
        "|---|---|---|---|---|",
    ]
    ranges = {"low": "below 30%", "moderate": "30% to 60%", "high": "60% or more"}
    for _, r in bands.iterrows():
        lines.append(f"| {r['band']} | {ranges[r['band']]} | {int(r['n']):,} | {_pct(r['mean_predicted'])} | "
                     f"{_pct(r['observed_prevalence'])} |")
    c30, c60 = band_cuts["30"], band_cuts["60"]
    lines += ["", "Using the band cut-offs as screening rules: an estimate of 30% or more flagged "
              f"{_pct(c30['recall_sensitivity'])} of people with moderate or severe periodontitis "
              f"(specificity {_pct(c30['specificity'])}); an estimate of 60% or more flagged "
              f"{_pct(c60['recall_sensitivity'])} (specificity {_pct(c60['specificity'])})."]
    lines += ["", "## Clinical usefulness (decision curve)", "",
              "Net benefit = true positives found per person screened, minus false positives weighted by "
              "the odds of the threshold (Vickers & Elkin 2006).", "",
              "| Threshold probability | Model | Refer everyone |", "|---|---|---|"]
    for d in decision:
        lines.append(f"| {d['threshold']:.0%} | {d['model']:.3f} | {d['refer_everyone']:.3f} |")
    b, a = transport["before"], transport["after"]
    lines += ["", "## Transportability: adjusting to a population's prevalence", "",
              f"Prevalence fell from {_pct(transport['source_prevalence'])} (2009-2012) to "
              f"{_pct(transport['target_prevalence'])} (2013-2014). Re-weighting the predicted odds to a known "
              "prevalence (prior-shift adjustment, Saerens et al. 2002; available in the package as "
              "`periorisk.recalibrate`) shows what a user with a reliable local prevalence estimate can expect. "
              "Here the test-set prevalence itself is used, so this is a demonstration of the mechanism, "
              "not an independent validation.", "",
              "| | Observed / expected | Calibration-in-the-large | Calibration slope | AUC |", "|---|---|---|---|---|",
              f"| As trained | {b['o_e_ratio']:.3f} | {b['calibration_intercept']:.3f} | {b['calibration_slope']:.3f} | {b['auc']:.3f} |",
              f"| Adjusted to {_pct(transport['target_prevalence'])} | {a['o_e_ratio']:.3f} | "
              f"{a['calibration_intercept']:.3f} | {a['calibration_slope']:.3f} | {a['auc']:.3f} |"]
    pb, pa = transport["population"]["weighted_before"], transport["population"]["weighted_after"]
    lines += ["", "A prevalence adjustment cannot fix every kind of miscalibration. Adjusting to the survey-weighted "
              f"US prevalence ({_pct(transport['population']['target_prevalence'])}) moves the survey-weighted "
              f"observed/expected ratio from {pb['o_e_ratio']:.2f} to {pa['o_e_ratio']:.2f} (an over-correction), because "
              "the population-level miscalibration comes mainly from NHANES's over-sampling of higher-risk groups "
              "(a change in who is surveyed), not from a change in prevalence alone."]
    lines += ["", "## Performance in subgroups (fairness check)", "",
              "Observed / expected below 1 means the model over-estimates risk in that group; above 1, "
              "under-estimates. Race/ethnicity is not a model input.", "",
              "| Group | Level | n | Prevalence | Mean predicted | Observed / expected | AUC |",
              "|---|---|---|---|---|---|---|"]
    for _, r in subgroups.iterrows():
        lines.append(f"| {r['group'].replace('_', ' ')} | {r['level'].replace('_', ' ')} | {int(r['n']):,} | "
                     f"{_pct(r['prevalence'])} | {_pct(r['mean_predicted'])} | {r['o_e_ratio']:.2f} | {r['auc']:.3f} |")
    keys = list(added)
    lines += ["", "## Sensitivity analysis: an eighth CDC/AAP self-report item", "",
              "The CDC/AAP self-report set also includes \"Do you think you might have gum disease?\" (OHQ835). "
              "Adding it to the selected model configuration was checked by 5-fold cross-validation on the "
              f"development data only: AUC {added[keys[0]]:.4f} without it vs {added[keys[1]]:.4f} with it. "
              "The gain is small; version 1.0.0 keeps the pre-specified 14-question set, and the item is a "
              "candidate for a future version."]
    lines += ["", "## Most influential answers (permutation importance)", "",
              "| Answer | Drop in AUC when shuffled |", "|---|---|"]
    for _, r in imp.iterrows():
        lines.append(f"| {r['feature'].replace('_', ' ')} | {r['auc_drop']:.4f} |")
    lines += ["", "## Figures", "",
              "![ROC](figures/roc_curves.png)", "![Calibration](figures/calibration.png)",
              "![Decision curve](figures/decision_curve.png)",
              "![Confusion matrix](figures/confusion_matrix.png)",
              "![Importance](figures/feature_importance.png)", ""]
    (REPORT_DIR / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
