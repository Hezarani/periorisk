/* PerioRisk engine - evaluates the portable model (model.js / periorisk/data/model.json) in the
 * browser. A line-by-line port of periorisk/_model.py and periorisk/__init__.py; the test suite
 * checks that both give the same probabilities for all 10,660 NHANES participants.
 * Nothing here touches the network: answers never leave the device. MIT licence. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.PerioRisk = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var AGE_MIN = 30, AGE_TOP = 80;
  var HEIGHT_CM = [100, 250], WEIGHT_KG = [25, 350], BMI = [12, 90];
  var GUM_HEALTH_SCORE = { excellent: 1, very_good: 2, good: 3, fair: 4, poor: 5 };
  var CHOICES = {
    sex: ["female", "male"],
    education: ["less_than_high_school", "high_school", "more_than_high_school"],
    smoking: ["never", "former", "current"],
    diabetes: ["no", "borderline", "yes"],
    gum_health: ["excellent", "very_good", "good", "fair", "poor"],
    gum_treatment: ["yes", "no", "not_sure"],
    loose_teeth: ["yes", "no", "not_sure"],
    bone_loss: ["yes", "no", "not_sure"],
    tooth_doesnt_look_right: ["yes", "no", "not_sure"]
  };
  // question key -> model input (same order as periorisk._FEATURE_OF)
  var FEATURE_OF = {
    age: "age", sex: "male", bmi: "bmi", education: "education", smoking: "smoking",
    diabetes: "diabetes", gum_health: "gum_health_rating", gum_treatment: "ever_treated_gum_disease",
    loose_teeth: "ever_loose_teeth", bone_loss: "told_bone_loss",
    tooth_doesnt_look_right: "teeth_dont_look_right", floss_days: "floss_days", mouthwash_days: "mouthwash_days"
  };

  function PerioRiskError(message, code) {
    var e = new Error(message);
    e.name = "PerioRiskError";
    e.code = code || "invalid";
    return e;
  }

  function missing(v) { return v === null || v === undefined || (typeof v === "number" && isNaN(v)); }

  function expit(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    var e = Math.exp(z);
    return e / (1 + e);
  }

  /* ---------- the model ---------- */
  function vector(model, fold, x) {
    var f = model.features, v = [], j, val, name, cats, k;
    for (j = 0; j < f.numeric.length; j++) {
      val = x[f.numeric[j]];
      val = missing(val) ? fold.numeric_impute[j] : Number(val);
      v.push((val - fold.numeric_mean[j]) / fold.numeric_scale[j]);
    }
    for (j = 0; j < f.binary.length; j++) {
      val = x[f.binary[j]];
      v.push(missing(val) ? fold.binary_impute[j] : Number(val));
    }
    for (j = 0; j < f.categorical.length; j++) {
      name = f.categorical[j];
      val = x[name];
      if (missing(val)) val = fold.categorical_impute[j];
      cats = model.categories[name];
      for (k = 0; k < cats.length; k++) v.push(val === cats[k] ? 1 : 0);
    }
    return v;
  }

  function probability(model, x) {
    var cats = model.categories, name;
    for (name in cats) {
      if (!missing(x[name]) && cats[name].indexOf(x[name]) < 0) {
        throw PerioRiskError(name + " must be one of " + cats[name].join(", "));
      }
    }
    var total = 0, i, t, n, raw, fold, v, trees;
    for (i = 0; i < model.folds.length; i++) {
      fold = model.folds[i];
      v = vector(model, fold, x);
      raw = fold.baseline;
      trees = fold.trees;
      for (t = 0; t < trees.length; t++) {
        var tr = trees[t], ff = tr.f;
        n = 0;
        while (ff[n] >= 0) n = v[ff[n]] <= tr.t[n] ? tr.l[n] : tr.r[n];
        raw += tr.v[n];
      }
      total += expit(-(fold.sigmoid_a * raw + fold.sigmoid_b));
    }
    return total / model.folds.length;
  }

  function band(model, p) {
    var bands = model.risk_bands;
    for (var i = 0; i < bands.length; i++) if (bands[i].from <= p && p < bands[i].to) return bands[i];
    return bands[bands.length - 1];
  }

  function recalibrate(p, target, source) {
    var a = target / source, b = (1 - target) / (1 - source);
    return a * p / (a * p + b * (1 - p));
  }

  /* ---------- answers -> model inputs (same rules as periorisk/__init__.py) ---------- */
  var NUMBER = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;
  var KEYS = ["natural_teeth", "age", "sex", "height_cm", "weight_kg", "bmi", "education", "smoking", "diabetes",
              "gum_health", "gum_treatment", "loose_teeth", "bone_loss", "tooth_doesnt_look_right",
              "floss_days", "mouthwash_days"];

  function num(key, value, lo, hi, integer) {
    if (value === null || value === undefined) return null;
    if (typeof value === "boolean") throw PerioRiskError(key + " must be a number");
    if (typeof value === "string") {
      var text = value.trim();
      if (text === "") return null;
      if (!NUMBER.test(text)) throw PerioRiskError(key + " must be a number");
      value = text;
    } else if (typeof value !== "number") {
      throw PerioRiskError(key + " must be a number");
    }
    var x = Number(value);
    if (isNaN(x)) return null;
    if (!(x >= lo && x <= hi)) throw PerioRiskError(key + " must be between " + lo + " and " + hi);
    if (integer && x !== Math.floor(x)) throw PerioRiskError(key + " must be a whole number");
    return x;
  }

  function choice(key, value) {
    if (missing(value) || value === "") return null;
    if (CHOICES[key].indexOf(value) < 0) throw PerioRiskError(key + " must be one of " + CHOICES[key].join(", "));
    return value;
  }

  function yn(v) { return v === null || v === "not_sure" ? null : (v === "yes" ? 1 : 0); }

  function fromAnswers(answers) {
    var notes = [], key;
    for (key in answers) {
      if (Object.prototype.hasOwnProperty.call(answers, key) && KEYS.indexOf(key) < 0) {
        throw PerioRiskError("unknown question: " + key);
      }
    }
    if (!missing(answers.natural_teeth) && answers.natural_teeth !== "yes" && answers.natural_teeth !== "no") {
      throw PerioRiskError("natural_teeth must be yes or no");
    }
    if (answers.natural_teeth === "no") {
      throw PerioRiskError("PerioRisk is for people with at least two natural teeth.", "out_of_scope");
    }
    var age = num("age", answers.age, 0, 130, false);
    if (age === null) throw PerioRiskError("age is required");
    if (age < AGE_MIN) throw PerioRiskError("PerioRisk is for adults aged 30 or older.", "out_of_scope");
    if (age > AGE_TOP) notes.push("Ages above 80 are treated as 80, as in NHANES.");
    var a = { age: age }, k;
    for (k in CHOICES) a[k] = choice(k, answers[k]);
    a.floss_days = num("floss_days", answers.floss_days, 0, 7, true);
    a.mouthwash_days = num("mouthwash_days", answers.mouthwash_days, 0, 7, true);
    var h = num("height_cm", answers.height_cm, HEIGHT_CM[0], HEIGHT_CM[1], false);
    var w = num("weight_kg", answers.weight_kg, WEIGHT_KG[0], WEIGHT_KG[1], false);
    var bmi = num("bmi", answers.bmi, BMI[0], BMI[1], false);
    if (bmi !== null && (h !== null || w !== null)) throw PerioRiskError("give either bmi or height and weight, not both");
    if (bmi === null && (h === null) !== (w === null)) throw PerioRiskError("height and weight must be given together");
    if (bmi === null && h !== null) {
      bmi = w / Math.pow(h / 100, 2);
      if (bmi < BMI[0] || bmi > BMI[1]) {
        throw PerioRiskError("Height and weight give a body-mass index of " + Math.round(bmi) +
          ", which is outside the plausible range. Please check them.", "implausible_bmi");
      }
    }
    a.bmi = bmi;
    var x = {
      age: Math.min(age, AGE_TOP),
      bmi: bmi,
      gum_health_rating: a.gum_health === null ? null : GUM_HEALTH_SCORE[a.gum_health],
      floss_days: a.floss_days,
      mouthwash_days: a.mouthwash_days,
      male: a.sex === null ? null : (a.sex === "male" ? 1 : 0),
      ever_treated_gum_disease: yn(a.gum_treatment),
      ever_loose_teeth: yn(a.loose_teeth),
      told_bone_loss: yn(a.bone_loss),
      teeth_dont_look_right: yn(a.tooth_doesnt_look_right),
      education: a.education,
      smoking: a.smoking,
      diabetes: a.diabetes
    };
    return { answers: a, x: x, notes: notes };
  }

  function assess(model, answers) {
    var norm = fromAnswers(answers), x = norm.x;
    var p = probability(model, x), b = band(model, p);
    var contributions = [], unanswered = [], key, feat;
    for (key in FEATURE_OF) {
      feat = FEATURE_OF[key];
      if (missing(x[feat])) { unanswered.push(key); continue; }
      var ref = {}, f;
      for (f in x) ref[f] = x[f];
      ref[feat] = null;
      contributions.push({ key: key, change: p - probability(model, ref) });
    }
    contributions.sort(function (u, v) { return Math.abs(v.change) - Math.abs(u.change); });
    return {
      probability: p, band: b.band, bandObserved: b.test_observed,
      screenPositive: p >= model.threshold, threshold: model.threshold,
      contributions: contributions, unanswered: unanswered, notes: norm.notes,
      answers: norm.answers, inputs: x
    };
  }

  return {
    probability: probability, band: band, recalibrate: recalibrate, fromAnswers: fromAnswers,
    assess: assess, FEATURE_OF: FEATURE_OF, CHOICES: CHOICES,
    limits: { age: [AGE_MIN, AGE_TOP], height_cm: HEIGHT_CM, weight_kg: WEIGHT_KG, bmi: BMI }
  };
});
