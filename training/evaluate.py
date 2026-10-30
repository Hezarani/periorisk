"""Step 4 - Evaluate models on the held-out 2013-2014 data and draw the figures.

Everything here follows the TRIPOD+AI reporting items for model performance: discrimination
(AUC), calibration (calibration-in-the-large, slope, observed/expected ratio, plot), clinical
usefulness (decision curve) and performance in subgroups - each with a 95% confidence interval.
Population-representative (survey-weighted) estimates use the NHANES exam weights, and their
confidence intervals respect the survey design (Rao-Wu rescaling bootstrap over the masked
variance units within strata).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score,  # noqa: E402
                             roc_auc_score, roc_curve)

from .config import FEATURES, N_BOOTSTRAP, RANDOM_STATE, RISK_BANDS, TARGET  # noqa: E402

COLORS = {"Logistic Regression": "#2E6F95", "Random Forest": "#7A9E7E",
          "Gradient Boosting": "#B8841F", "Final model": "#9B2D4B"}

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titleweight": "bold"})


# ---------- helpers ----------

def risk_band(p: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "high"


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def prior_shift(p, source_prevalence: float, target_prevalence: float):
    """Adjust probabilities to a population with a different prevalence (Saerens et al. 2002).
    Assumes answers given disease status are distributed as in NHANES (label shift)."""
    p = np.asarray(p, float)
    a = target_prevalence / source_prevalence
    b = (1 - target_prevalence) / (1 - source_prevalence)
    return a * p / (a * p + b * (1 - p))


def _logistic(X, y, w, offset=None, iters=100):
    """Weighted logistic regression by Newton-Raphson (no penalty). Returns coefficients."""
    X = np.asarray(X, float)
    beta = np.zeros(X.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ beta + off)))
        grad = X.T @ (w * (y - mu))
        hess = (X * (w * mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration_stats(y, p, w=None) -> dict:
    """Observed/expected ratio, calibration-in-the-large (intercept with the model's
    log-odds as offset; ideal 0) and calibration slope (ideal 1)."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    lp = logit(p)
    citl = _logistic(np.ones((len(y), 1)), y, w, offset=lp)[0]
    slope = _logistic(np.column_stack([np.ones_like(lp), lp]), y, w)[1]
    return {"o_e_ratio": float(np.sum(w * y) / np.sum(w * np.asarray(p))),
            "calibration_intercept": float(citl), "calibration_slope": float(slope)}


def threshold_metrics(y, p, threshold, w=None) -> dict:
    y = np.asarray(y)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    pred = np.asarray(p) >= threshold
    tp = np.sum(w * (pred & (y == 1))); fp = np.sum(w * (pred & (y == 0)))
    tn = np.sum(w * (~pred & (y == 0))); fn = np.sum(w * (~pred & (y == 1)))
    return {"recall_sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "precision_ppv": tp / (tp + fp), "npv": tn / (tn + fn),
            "accuracy": (tp + tn) / (tp + tn + fp + fn)}


def bootstrap_auc(y, p, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def classification_metrics(y, p, threshold) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    lo, hi = bootstrap_auc(y, p)
    out = {"threshold": float(threshold), "auc": roc_auc_score(y, p), "auc_ci_low": lo, "auc_ci_high": hi}
    out |= threshold_metrics(y, p, threshold)
    brier = brier_score_loss(y, p)
    out |= {"f1": f1_score(y, pred), "brier": brier,
            "scaled_brier": 1 - brier / (y.mean() * (1 - y.mean())),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    out |= calibration_stats(y, p)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


CI_KEYS = ["recall_sensitivity", "specificity", "precision_ppv", "npv", "brier",
           "o_e_ratio", "calibration_intercept", "calibration_slope"]


def bootstrap_ci(y, p, threshold, n=N_BOOTSTRAP, seed=RANDOM_STATE) -> dict:
    """Percentile 95% CIs (participant-level bootstrap) for threshold and calibration metrics."""
    rng = np.random.default_rng(seed + 1)
    y, p = np.asarray(y), np.asarray(p)
    draws = {k: [] for k in CI_KEYS}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        m = threshold_metrics(yb, pb, threshold) | calibration_stats(yb, pb)
        m["brier"] = brier_score_loss(yb, pb)
        for k in CI_KEYS:
            draws[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def _survey_point(y, p, threshold, w) -> dict:
    out = {"prevalence": float(np.sum(w * y) / np.sum(w)),
           "auc": float(roc_auc_score(y, p, sample_weight=w)),
           "brier": float(np.sum(w * (p - y) ** 2) / np.sum(w))}
    out |= {k: float(v) for k, v in threshold_metrics(y, p, threshold, w).items()}
    out |= calibration_stats(y, p, w)
    return out


def survey_metrics(y, p, threshold, weight, psu, stratum, n=N_BOOTSTRAP, seed=RANDOM_STATE) -> dict:
    """Population-representative performance (NHANES exam weights) with design-based 95% CIs
    from the Rao-Wu rescaling bootstrap (Rao, Wu & Yue 1992): in each stratum, n_h - 1 PSUs are
    drawn with replacement and their weights multiplied by n_h / (n_h - 1) x times drawn."""
    y, p, w = np.asarray(y), np.asarray(p, float), np.asarray(weight, float)
    psu, stratum = np.asarray(psu), np.asarray(stratum)
    point = _survey_point(y, p, threshold, w)
    rng = np.random.default_rng(seed + 2)
    groups = {}
    for h in np.unique(stratum):
        units = np.unique(psu[stratum == h])
        groups[h] = [(u, (stratum == h) & (psu == u)) for u in units]
    draws = {k: [] for k in point}
    for _ in range(n):
        mult = np.zeros(len(y))
        for h, units in groups.items():
            nh = len(units)
            if nh < 2:  # a stratum with a single PSU cannot be resampled; keep it as is
                mult[units[0][1]] = 1.0
                continue
            picks = rng.integers(0, nh, nh - 1)
            counts = np.bincount(picks, minlength=nh)
            for (u, mask), c in zip(units, counts):
                mult[mask] = c * nh / (nh - 1)
        wb = w * mult
        keep = wb > 0
        if len(np.unique(y[keep])) < 2:
            continue
        m = _survey_point(y[keep], p[keep], threshold, wb[keep])
        for k in point:
            draws[k].append(m[k])
    return {k: {"estimate": v, "ci": [float(np.percentile(draws[k], 2.5)),
                                      float(np.percentile(draws[k], 97.5))]}
            for k, v in point.items()} | {"n_replicates": len(draws["auc"]),
                                          "n_strata": len(groups)}


def subgroup_table(test: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    df = test.assign(prob=p)
    df["age_group"] = pd.cut(df["age"], [29, 44, 64, 200], labels=["30-44", "45-64", "65+"])
    df["sex"] = np.where(df["male"] == 1, "Male", "Female")
    rows = []
    for col in ["sex", "age_group", "race_ethnicity", "education", "diabetes"]:
        for level, g in df.groupby(col, observed=True):
            if g[TARGET].nunique() == 2 and len(g) >= 50:
                rows.append({"group": col, "level": str(level), "n": len(g),
                             "prevalence": g[TARGET].mean(),
                             "mean_predicted": g["prob"].mean(),
                             "o_e_ratio": g[TARGET].mean() / g["prob"].mean(),
                             "auc": roc_auc_score(g[TARGET], g["prob"])})
    return pd.DataFrame(rows)


def band_table(y, p) -> pd.DataFrame:
    bands = pd.Series([risk_band(x) for x in p], name="band")
    df = pd.DataFrame({"band": bands, "y": np.asarray(y), "p": np.asarray(p)})
    out = df.groupby("band").agg(n=("y", "size"), observed_prevalence=("y", "mean"),
                                 mean_predicted=("p", "mean"))
    return out.reindex(["low", "moderate", "high"]).reset_index()


def net_benefit(y, p, thresholds):
    y = np.asarray(y); n = len(y); prev = y.mean()
    model, treat_all = [], []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        model.append(tp / n - fp / n * t / (1 - t))
        treat_all.append(prev - (1 - prev) * t / (1 - t))
    return np.array(model), np.array(treat_all)


# ---------- figures ----------

def plot_roc(y, probs: dict, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in probs.items():
        final = name == "Final model"
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=COLORS.get(name, "grey"), lw=2.4 if final else 1.4,
                ls="-" if final else (0, (4, 2)), zorder=3 if final else 2,
                label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="#999", lw=1)
    ax.set(xlabel="1 - specificity", ylabel="Sensitivity", title="ROC curves, 2013-2014 test set")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_calibration(y, p, path):
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [5, 1], "hspace": 0.05})
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1, label="Perfect calibration")
    ax.plot(mean, frac, marker="o", color=COLORS["Final model"], lw=2, label="Final model (deciles)")
    ax.set(ylabel="Observed proportion with periodontitis", title="Calibration, 2013-2014 test set",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    axh.hist(p, bins=40, range=(0, 1), color=COLORS["Final model"], alpha=0.55)
    axh.set(xlabel="Predicted probability", yticks=[])
    axh.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.1)
    fig.savefig(path); plt.close(fig)


def plot_confusion(m: dict, path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="RdPu")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Predicted no", "Predicted yes"])
    ax.set_yticks([0, 1], ["Actual no", "Actual yes"])
    ax.set_title(f"Confusion matrix, test set (threshold {m['threshold']:.2f})")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_decision_curve(y, p, path):
    t = np.linspace(0.05, 0.80, 76)
    model, all_ = net_benefit(y, p, t)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(t, model, color=COLORS["Final model"], lw=2, label="Final model")
    ax.plot(t, all_, color="#2E6F95", lw=1.5, label="Refer everyone")
    ax.axhline(0, color="#555", lw=1, label="Refer no one")
    ax.set(xlabel="Threshold probability", ylabel="Net benefit",
           title="Decision curve, test set", ylim=(-0.05, max(model.max(), all_.max()) + 0.05))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_importance(model, test: pd.DataFrame, path) -> pd.DataFrame:
    r = permutation_importance(model, test[FEATURES], test[TARGET], scoring="roc_auc",
                               n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURES, "auc_drop": r.importances_mean,
                        "sd": r.importances_std}).sort_values("auc_drop")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp["feature"].str.replace("_", " "), imp["auc_drop"], xerr=imp["sd"],
            color=COLORS["Final model"], alpha=0.85)
    ax.set(xlabel="Drop in AUC when the answer is shuffled", title="Which answers matter most")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return imp.sort_values("auc_drop", ascending=False)


# ---------- descriptive tables ----------

def _pct(series, value):
    s = series.dropna()
    return 100 * float((s == value).mean()) if len(s) else float("nan")


def characteristics(train: pd.DataFrame, test: pd.DataFrame) -> list:
    """Participant characteristics (unweighted) of the development and evaluation data."""
    def rows(d):
        return [
            ("Participants, n", f"{len(d):,}"),
            ("Moderate or severe periodontitis, n (%)", f"{int(d[TARGET].sum()):,} ({100 * d[TARGET].mean():.1f}%)"),
            ("Age, years, mean (SD)", f"{d['age'].mean():.1f} ({d['age'].std():.1f})"),
            ("Male, %", f"{100 * d['male'].mean():.1f}"),
            ("Body-mass index, mean (SD)", f"{d['bmi'].mean():.1f} ({d['bmi'].std():.1f})"),
            ("Education: less than high school / high school / more, %",
             " / ".join(f"{_pct(d['education'], v):.1f}" for v in
                        ("less_than_high_school", "high_school", "more_than_high_school"))),
            ("Smoking: never / former / current, %",
             " / ".join(f"{_pct(d['smoking'], v):.1f}" for v in ("never", "former", "current"))),
            ("Diabetes: no / borderline / yes, %",
             " / ".join(f"{_pct(d['diabetes'], v):.1f}" for v in ("no", "borderline", "yes"))),
            ("Teeth and gums rated fair or poor, %",
             f"{100 * float((d['gum_health_rating'].dropna() >= 4).mean()):.1f}"),
            ("Ever treated for gum disease, %", f"{_pct(d['ever_treated_gum_disease'], 1):.1f}"),
            ("Ever had a tooth loosen on its own, %", f"{_pct(d['ever_loose_teeth'], 1):.1f}"),
            ("Told of bone loss around teeth, %", f"{_pct(d['told_bone_loss'], 1):.1f}"),
            ("Noticed a tooth that doesn't look right, %", f"{_pct(d['teeth_dont_look_right'], 1):.1f}"),
            ("Days cleaning between teeth (0-7), mean (SD)", f"{d['floss_days'].mean():.1f} ({d['floss_days'].std():.1f})"),
            ("Days using mouthwash for dental problems (0-7), mean (SD)",
             f"{d['mouthwash_days'].mean():.1f} ({d['mouthwash_days'].std():.1f})"),
        ]
    dev, ev = rows(train), rows(test)
    return [{"characteristic": a[0], "development": a[1], "evaluation": b[1]} for a, b in zip(dev, ev)]


def missingness(train: pd.DataFrame, test: pd.DataFrame) -> list:
    out = [{"feature": f, "development_pct": 100 * float(train[f].isna().mean()),
            "evaluation_pct": 100 * float(test[f].isna().mean())} for f in FEATURES]
    out.append({"feature": "any of the above", "development_pct": 100 * float(train[FEATURES].isna().any(axis=1).mean()),
                "evaluation_pct": 100 * float(test[FEATURES].isna().any(axis=1).mean())})
    return out
