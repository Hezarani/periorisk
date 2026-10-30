"""Evaluate the portable PerioRisk model (periorisk/data/model.json) in pure Python.

The JSON file holds the complete fitted model exported from scikit-learn by the training pipeline:
five cross-fitted gradient-boosting ensembles, each with its own imputation values, scaling
constants, 300 regression trees and sigmoid (Platt) calibration. The final probability is the
mean of the five calibrated probabilities - exactly what scikit-learn's CalibratedClassifierCV
computes. The test suite checks this against scikit-learn's own output for all 10,660 NHANES
participants.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources
from typing import Mapping, Optional

FORMAT = "periorisk.portable-model"


class _Fold:
    __slots__ = ("num_impute", "num_mean", "num_scale", "bin_impute", "cat_impute",
                 "baseline", "a", "b", "trees")

    def __init__(self, d: dict):
        self.num_impute = d["numeric_impute"]
        self.num_mean = d["numeric_mean"]
        self.num_scale = d["numeric_scale"]
        self.bin_impute = d["binary_impute"]
        self.cat_impute = d["categorical_impute"]
        self.baseline = d["baseline"]
        self.a = d["sigmoid_a"]
        self.b = d["sigmoid_b"]
        self.trees = [(t["f"], t["t"], t["l"], t["r"], t["v"]) for t in d["trees"]]


class PortableModel:
    """A loaded portable model. Use :func:`load_model` to get the bundled one."""

    def __init__(self, spec: dict):
        if spec.get("format") != FORMAT or spec.get("format_version") != 1:
            raise ValueError("not a PerioRisk portable model (format version 1)")
        self.spec = spec
        feats = spec["features"]
        self.numeric = list(feats["numeric"])
        self.binary = list(feats["binary"])
        self.categorical = list(feats["categorical"])
        self.features = list(feats["order"])
        self.categories = {k: list(v) for k, v in spec["categories"].items()}
        self.folds = [_Fold(f) for f in spec["folds"]]
        self.threshold = float(spec["threshold"])
        self.train_prevalence = float(spec["train_prevalence"])
        self.version = spec["model_version"]
        self.risk_bands = spec["risk_bands"]

    # -- core ---------------------------------------------------------------------------
    def _vector(self, fold: _Fold, x: Mapping[str, object]) -> list:
        v = []
        for j, name in enumerate(self.numeric):
            val = x.get(name)
            val = fold.num_impute[j] if _missing(val) else float(val)
            v.append((val - fold.num_mean[j]) / fold.num_scale[j])
        for j, name in enumerate(self.binary):
            val = x.get(name)
            v.append(fold.bin_impute[j] if _missing(val) else float(val))
        for j, name in enumerate(self.categorical):
            val = x.get(name)
            val = fold.cat_impute[j] if _missing(val) else val
            v.extend(1.0 if val == c else 0.0 for c in self.categories[name])
        return v

    def probability(self, x: Mapping[str, object]) -> float:
        """Calibrated probability of moderate/severe periodontitis for one person, given the
        13 model inputs (any of them may be None = not answered)."""
        self._check(x)
        total = 0.0
        for fold in self.folds:
            v = self._vector(fold, x)
            raw = fold.baseline
            for f, t, left, right, value in fold.trees:
                n = 0
                while f[n] >= 0:
                    n = left[n] if v[f[n]] <= t[n] else right[n]
                raw += value[n]
            total += _expit(-(fold.a * raw + fold.b))
        return total / len(self.folds)

    def band(self, p: float) -> dict:
        for b in self.risk_bands:
            if b["from"] <= p < b["to"]:
                return b
        return self.risk_bands[-1]

    def _check(self, x: Mapping[str, object]) -> None:
        unknown = set(x) - set(self.features)
        if unknown:
            raise KeyError(f"unknown model input(s): {', '.join(sorted(unknown))}")
        for name in self.categorical:
            val = x.get(name)
            if not _missing(val) and val not in self.categories[name]:
                raise ValueError(f"{name} must be one of {self.categories[name]}, got {val!r}")


def _missing(val) -> bool:
    return val is None or (isinstance(val, float) and math.isnan(val))


def _expit(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@lru_cache(maxsize=1)
def load_model() -> PortableModel:
    """The model bundled with this package (cached after the first call)."""
    text = resources.files("periorisk").joinpath("data/model.json").read_text(encoding="utf-8")
    return PortableModel(json.loads(text))


def load_model_file(path) -> PortableModel:
    """Load a portable model from another file (e.g. a re-trained model)."""
    with open(path, encoding="utf-8") as fh:
        return PortableModel(json.load(fh))


def recalibrate(probability: float, target_prevalence: float,
                source_prevalence: Optional[float] = None) -> float:
    """Adjust a predicted probability to a population with a different prevalence of
    moderate/severe periodontitis (prior-shift adjustment; Saerens, Latinne & Decaestecker 2002).

    ``source_prevalence`` defaults to the prevalence in the model's training data (48.7%).
    The adjustment assumes that, within people with and without the disease, answers are
    distributed as in NHANES. Use it only with a reliable prevalence estimate for the same
    CDC/AAP case definition; it changes calibration, never the ranking of people (AUC).
    """
    if source_prevalence is None:
        source_prevalence = load_model().train_prevalence
    for name, val in (("probability", probability), ("target_prevalence", target_prevalence),
                      ("source_prevalence", source_prevalence)):
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0.0 < target_prevalence < 1.0 or not 0.0 < source_prevalence < 1.0:
        raise ValueError("prevalences must be strictly between 0 and 1")
    a = target_prevalence / source_prevalence
    b = (1.0 - target_prevalence) / (1.0 - source_prevalence)
    return a * probability / (a * probability + b * (1.0 - probability))
