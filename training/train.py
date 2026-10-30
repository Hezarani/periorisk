"""Step 3 - Train and compare three models, pick the best one by cross-validated AUC.

Models: Logistic Regression, Random Forest, Histogram Gradient Boosting
(scikit-learn's built-in gradient boosting, same family as XGBoost/LightGBM, no extra install).
Everything (imputation, scaling, encoding, model) lives in one scikit-learn Pipeline, so the
exact same steps run in training and in the exported portable model.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (BINARY_FEATURES, CATEGORICAL_FEATURES, CYCLES, FEATURES, NUMERIC_FEATURES,
                     RANDOM_STATE, TARGET, TEST_CYCLES, TRAIN_CYCLES)


def make_preprocessor(extra_binary=()) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES + list(extra_binary)),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])


CANDIDATES = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000),
        {"model__C": [0.1, 1.0, 10.0]},
    ),
    "Random Forest": (
        # n_jobs=1: a multi-threaded forest adds its trees' votes in a random order, which changes the
        # last digits of its predictions from run to run (the grid search below is still parallel)
        RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=RANDOM_STATE),
        {"model__min_samples_leaf": [10, 25, 50], "model__max_features": ["sqrt", 0.5]},
    ),
    "Gradient Boosting": (
        HistGradientBoostingClassifier(max_iter=300, early_stopping=False, random_state=RANDOM_STATE),
        {"model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [7, 15],
         "model__l2_regularization": [0.0, 1.0]},
    ),
}


def _cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def split(features: pd.DataFrame):
    train_labels = [CYCLES[c][1] for c in TRAIN_CYCLES]
    test_labels = [CYCLES[c][1] for c in TEST_CYCLES]
    train = features[features["cycle"].isin(train_labels)].reset_index(drop=True)
    test = features[features["cycle"].isin(test_labels)].reset_index(drop=True)
    return train, test


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def train_models(train: pd.DataFrame, verbose: bool = True) -> dict:
    X, y = train[FEATURES], train[TARGET]
    fitted = {}
    for name, (model, grid) in CANDIDATES.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=_cv(), n_jobs=-1)
        search.fit(X, y)
        fitted[name] = {"estimator": search.best_estimator_,
                        "cv_auc": float(search.best_score_),
                        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()}}
        if verbose:
            print(f"     {name:<20} cross-validated AUC = {search.best_score_:.3f}  {fitted[name]['best_params']}")
    return fitted


def own_thresholds(fitted: dict, train: pd.DataFrame) -> dict:
    """Each candidate's own screening threshold (Youden's J on its out-of-fold training
    predictions), so the model comparison compares like with like."""
    X, y = train[FEATURES], train[TARGET]
    out = {}
    for name, f in fitted.items():
        oof = cross_val_predict(f["estimator"], X, y, cv=_cv(), method="predict_proba")[:, 1]
        out[name] = youden_threshold(y, oof)
    return out


def finalize(best_name: str, best_estimator, train: pd.DataFrame):
    """Calibrate tree models, and choose a screening threshold on training data only."""
    X, y = train[FEATURES], train[TARGET]
    if best_name == "Logistic Regression":
        final = best_estimator  # already well calibrated by design
    else:
        final = CalibratedClassifierCV(best_estimator, method="sigmoid", cv=_cv())
    oof = cross_val_predict(final, X, y, cv=_cv(), method="predict_proba")[:, 1]
    threshold = youden_threshold(y, oof)
    final.fit(X, y)
    return final, threshold, oof


def added_predictor_check(best_estimator, train: pd.DataFrame, extra: str) -> dict:
    """Sensitivity analysis on the TRAINING data only: 5-fold cross-validated AUC of the selected
    model configuration with and without one extra binary predictor."""
    out = {}
    for label, cols in (("pre-specified", []), (f"plus {extra}", [extra])):
        pipe = Pipeline([("prep", make_preprocessor(cols)), ("model", clone(best_estimator.named_steps["model"]))])
        scores = cross_val_score(pipe, train[FEATURES + cols], train[TARGET], cv=_cv(), scoring="roc_auc")
        out[label] = float(scores.mean())
    return out
