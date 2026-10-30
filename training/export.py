"""Step 5 - Export the trained scikit-learn model to a portable, dependency-free JSON file.

The fitted model (5 calibrated gradient-boosting ensembles, each with its own imputation and
scaling) is written out as plain numbers: imputation values, scaling constants, every tree node
and the sigmoid calibration parameters. The Python package (periorisk/) and the web app (docs/)
both evaluate this same file, so neither needs scikit-learn or a server - and neither depends on
a particular scikit-learn or Python version.

The export is verified against scikit-learn on every participant before it is written.
"""
import json

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from .config import (BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES)

FORMAT = "periorisk.portable-model"
FORMAT_VERSION = 1


def _tree(nodes) -> dict:
    leaf = nodes["is_leaf"].astype(bool)
    if np.any(nodes["is_categorical"][~leaf]):
        raise ValueError("categorical splits are not supported by the portable format")
    return {
        "f": [-1 if lf else int(fi) for lf, fi in zip(leaf, nodes["feature_idx"])],
        "t": [0 if lf else float(t) for lf, t in zip(leaf, nodes["num_threshold"])],
        "l": [0 if lf else int(x) for lf, x in zip(leaf, nodes["left"])],
        "r": [0 if lf else int(x) for lf, x in zip(leaf, nodes["right"])],
        "v": [float(v) if lf else 0 for lf, v in zip(leaf, nodes["value"])],
    }


def _fold(calibrated) -> dict:
    pipe = calibrated.estimator
    prep, hgb = pipe.named_steps["prep"], pipe.named_steps["model"]
    num = prep.named_transformers_["num"]
    cat = prep.named_transformers_["cat"]
    if hgb.n_trees_per_iteration_ != 1 or len(calibrated.calibrators) != 1:
        raise ValueError("only binary classifiers are supported")
    # the column order the trees see: numeric (scaled), binary, one-hot categorical
    expected = ([f"num__{c}" for c in NUMERIC_FEATURES] + [f"bin__{c}" for c in BINARY_FEATURES]
                + [f"cat__{c}_{v}" for c, cats in zip(CATEGORICAL_FEATURES, cat.named_steps["onehot"].categories_)
                   for v in cats])
    if list(prep.get_feature_names_out()) != expected:
        raise ValueError("unexpected preprocessing layout")
    sig = calibrated.calibrators[0]
    return {
        "numeric_impute": [float(v) for v in num.named_steps["impute"].statistics_],
        "numeric_mean": [float(v) for v in num.named_steps["scale"].mean_],
        "numeric_scale": [float(v) for v in num.named_steps["scale"].scale_],
        "binary_impute": [float(v) for v in prep.named_transformers_["bin"].statistics_],
        "categorical_impute": [str(v) for v in cat.named_steps["impute"].statistics_],
        "categories": [[str(v) for v in c] for c in cat.named_steps["onehot"].categories_],
        "baseline": float(hgb._baseline_prediction.ravel()[0]),
        "sigmoid_a": float(sig.a_),
        "sigmoid_b": float(sig.b_),
        "trees": [_tree(p[0].nodes) for p in hgb._predictors],
    }


def export_model(final, extra: dict) -> dict:
    if not isinstance(final, CalibratedClassifierCV):
        raise ValueError("the portable export expects the calibrated gradient-boosting model")
    folds = [_fold(c) for c in final.calibrated_classifiers_]
    cats = folds[0]["categories"]
    if any(f["categories"] != cats for f in folds):
        raise ValueError("folds disagree on the categories")
    for f in folds:
        del f["categories"]
    return {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "features": {"numeric": NUMERIC_FEATURES, "binary": BINARY_FEATURES,
                     "categorical": CATEGORICAL_FEATURES, "order": FEATURES},
        "categories": dict(zip(CATEGORICAL_FEATURES, cats)),
        **extra,
        "folds": folds,
    }


# ---- a straightforward re-implementation, used here only to verify the export ----

def portable_predict(model: dict, rows: list[dict]) -> np.ndarray:
    feats = model["features"]
    cats = model["categories"]
    out = np.zeros(len(rows))
    for i, x in enumerate(rows):
        total = 0.0
        for fold in model["folds"]:
            v = []
            for j, name in enumerate(feats["numeric"]):
                val = x.get(name)
                val = fold["numeric_impute"][j] if val is None or val != val else float(val)
                v.append((val - fold["numeric_mean"][j]) / fold["numeric_scale"][j])
            for j, name in enumerate(feats["binary"]):
                val = x.get(name)
                v.append(fold["binary_impute"][j] if val is None or val != val else float(val))
            for j, name in enumerate(feats["categorical"]):
                val = x.get(name)
                val = fold["categorical_impute"][j] if val is None or val != val else val
                v.extend(1.0 if val == c else 0.0 for c in cats[name])
            raw = fold["baseline"]
            for t in fold["trees"]:
                n = 0
                while t["f"][n] >= 0:
                    n = t["l"][n] if v[t["f"][n]] <= t["t"][n] else t["r"][n]
                raw += t["v"][n]
            total += 1.0 / (1.0 + np.exp(fold["sigmoid_a"] * raw + fold["sigmoid_b"]))
        out[i] = total / len(model["folds"])
    return out


def verify(model: dict, final, frame) -> float:
    rows = frame[FEATURES].astype(object).where(frame[FEATURES].notna(), None).to_dict("records")
    ours = portable_predict(model, rows)
    theirs = final.predict_proba(frame[FEATURES])[:, 1]
    return float(np.max(np.abs(ours - theirs)))


def write(model: dict, json_path, js_path) -> None:
    text = json.dumps(model, separators=(",", ":"), allow_nan=False)
    json_path.write_text(text + "\n")
    js_path.write_text("/* PerioRisk portable model - generated by `python -m training`; do not edit. */\n"
                       f"window.PERIORISK_MODEL = {text};\n")
