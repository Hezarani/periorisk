/* PerioRisk web app - questionnaire, result and explanation. Everything runs on this device:
 * no network requests, no storage, no cookies. MIT licence. */
(function () {
  "use strict";
  var MODEL = window.PERIORISK_MODEL, PR = window.PerioRisk;
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var YNS = [["yes", "Yes"], ["no", "No"], ["not_sure", "Not sure"]];
  // Wording follows the NHANES 2009-2014 questionnaires (see periorisk/questions.py).
  var Q = {
    age: { type: "age", text: "How old are you?", help: "PerioRisk is for adults aged 30 and over." },
    natural_teeth: { type: "seg", text: "Do you still have at least two of your own natural teeth, not counting wisdom teeth?",
      help: "Teeth with fillings or crowns count; implants, bridges and dentures don’t.", choices: [["yes", "Yes"], ["no", "No"]] },
    sex: { type: "seg", text: "Sex", choices: [["female", "Female"], ["male", "Male"]] },
    body: { type: "body", text: "Height and weight", tag: "optional",
      help: "Used to work out body-mass index. It has only a small effect on the estimate." },
    education: { type: "list", text: "What is the highest grade or level of school you have completed, or the highest degree you have received?",
      choices: [["less_than_high_school", "Less than high school (did not finish)"],
                ["high_school", "High school graduate or equivalent (e.g. GED)"],
                ["more_than_high_school", "Some college, vocational or university education, or a degree"]] },
    smoking: { type: "list", text: "Which describes you best?", help: "Cigarettes only, as in the survey.",
      choices: [["never", "I have smoked fewer than 100 cigarettes in my whole life"],
                ["former", "I have smoked 100 or more cigarettes, but I don’t smoke now"],
                ["current", "I smoke cigarettes now, every day or on some days"]] },
    diabetes: { type: "seg", text: "Other than during pregnancy, have you ever been told by a doctor or health professional that you have diabetes or sugar diabetes?",
      choices: [["no", "No"], ["borderline", "Borderline"], ["yes", "Yes"]] },
    gum_health: { type: "seg", text: "Overall, how would you rate the health of your teeth and gums?",
      choices: [["excellent", "Excellent"], ["very_good", "Very good"], ["good", "Good"], ["fair", "Fair"], ["poor", "Poor"]] },
    gum_treatment: { type: "seg", text: "Have you ever had treatment for gum disease such as scaling and root planing, sometimes called “deep cleaning”?", choices: YNS },
    loose_teeth: { type: "seg", text: "Have you ever had any teeth become loose on their own, without an injury?", choices: YNS },
    bone_loss: { type: "seg", text: "Have you ever been told by a dental professional that you lost bone around your teeth?", choices: YNS },
    tooth_doesnt_look_right: { type: "seg", text: "During the past three months, have you noticed a tooth that doesn’t look right?", choices: YNS },
    floss_days: { type: "days", text: "Aside from brushing your teeth with a toothbrush, in the last seven days, how many days did you use dental floss or any other device to clean between your teeth?" },
    mouthwash_days: { type: "days", text: "Aside from brushing your teeth with a toothbrush, in the last seven days, how many days did you use mouthwash or other dental rinse product that you use to treat dental disease or dental problems?" }
  };
  var ORDER = ["age", "natural_teeth", "sex", "body", "education", "smoking", "diabetes", "gum_health",
               "gum_treatment", "loose_teeth", "bone_loss", "tooth_doesnt_look_right", "floss_days", "mouthwash_days"];
  var LABELS = { age: "Age", sex: "Sex", bmi: "Height and weight", education: "Education", smoking: "Smoking",
    diabetes: "Diabetes", gum_health: "Self-rated teeth and gums", gum_treatment: "Past treatment for gum disease",
    loose_teeth: "A tooth became loose on its own", bone_loss: "Told of bone loss around teeth",
    tooth_doesnt_look_right: "A tooth that doesn’t look right", floss_days: "Days cleaning between teeth",
    mouthwash_days: "Days using mouthwash for dental problems" };
  var SHORT = { never: "Never smoked", former: "Used to smoke", current: "Smokes now",
    less_than_high_school: "Less than high school", high_school: "High school", more_than_high_school: "More than high school",
    not_sure: "Not sure", very_good: "Very good" };
  var BAND = { low: { name: "Lower risk", group: "lower-risk band (below 30%)", color: "var(--low)" },
               moderate: { name: "Moderate risk", group: "moderate-risk band (30–60%)", color: "var(--mod)" },
               high: { name: "Higher risk", group: "higher-risk band (60% or more)", color: "var(--high)" } };

  var state = { units: "metric", revealed: false };
  var answers = {};
  var $ = function (id) { return document.getElementById(id); };
  var form = $("check");

  function h(tag, attrs, html) {
    var el = document.createElement(tag), k;
    for (k in attrs || {}) el.setAttribute(k, attrs[k]);
    if (html !== undefined) el.innerHTML = html;
    return el;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  /* ---------- build the questions ---------- */
  function fieldset(key, q) {
    var fs = h("fieldset", { "class": "q", id: "q-" + key, "data-key": key });
    var legend = esc(q.text) + (q.tag ? ' <span class="tag">' + q.tag + "</span>" : "");
    fs.appendChild(h("legend", {}, legend));
    if (q.help) fs.appendChild(h("p", { "class": "help", id: "help-" + key }, esc(q.help)));
    return fs;
  }

  function radios(key, choices, cls) {
    var wrap = h("div", { "class": cls });
    choices.forEach(function (c) {
      var lab = h("label");
      lab.appendChild(h("input", { type: "radio", name: key, value: c[0] }));
      lab.appendChild(h("span", {}, esc(c[1])));
      wrap.appendChild(lab);
    });
    return wrap;
  }

  function build() {
    ORDER.forEach(function (key) {
      var q = Q[key], slot = form.querySelector('[data-slot="' + key + '"]'), fs = fieldset(key, q);
      if (q.type === "seg") fs.appendChild(radios(key, q.choices, q.choices.length === 5 ? "seg scale5" : "seg"));
      else if (q.type === "list") fs.appendChild(radios(key, q.choices, "list"));
      else if (q.type === "days") {
        var days = [];
        for (var d = 0; d <= 7; d++) days.push([String(d), String(d)]);
        fs.appendChild(radios(key, days, "seg days"));
        fs.appendChild(h("div", { "class": "scale-ends", "aria-hidden": "true" }, "<span>No days</span><span>Every day</span>"));
      } else if (q.type === "age") {
        var f = h("div", { "class": "fields" });
        f.innerHTML = '<label class="field">Age in years<input id="age" name="age" type="number" inputmode="numeric" min="30" max="120" step="1" aria-describedby="help-age age-msg"></label>';
        fs.appendChild(f);
        fs.appendChild(h("p", { "class": "msg", id: "age-msg", "aria-live": "polite" }));
      } else if (q.type === "body") {
        fs.appendChild(h("div", { "class": "units", role: "group", "aria-label": "Units" },
          '<button type="button" data-units="metric" aria-pressed="true">cm · kg</button><button type="button" data-units="us" aria-pressed="false">ft · in · lb</button>'));
        fs.appendChild(h("div", { "class": "fields", id: "body-metric" },
          '<label class="field">Height (cm)<input id="h-cm" type="number" inputmode="decimal" min="100" max="250" step="any"></label>' +
          '<label class="field">Weight (kg)<input id="w-kg" type="number" inputmode="decimal" min="25" max="350" step="any"></label>'));
        var us = h("div", { "class": "fields", id: "body-us", hidden: "" },
          '<label class="field">Height (ft)<input id="h-ft" class="short" type="number" inputmode="numeric" min="3" max="8" step="1"></label>' +
          '<label class="field">(in)<input id="h-in" class="short" type="number" inputmode="decimal" min="0" max="11.9" step="any"></label>' +
          '<label class="field">Weight (lb)<input id="w-lb" type="number" inputmode="decimal" min="55" max="770" step="any"></label>');
        fs.appendChild(us);
        fs.appendChild(h("label", { "class": "check" }, '<input type="checkbox" id="body-skip"> Prefer not to say'));
        fs.appendChild(h("p", { "class": "msg", id: "body-msg", "aria-live": "polite" }));
      }
      slot.replaceWith(fs);
    });
  }

  /* ---------- read answers ---------- */
  function readRadio(key) {
    var el = form.querySelector('input[name="' + key + '"]:checked');
    return el ? el.value : null;
  }
  // NaN = something was typed that is not a number (the browser reports it as "badInput")
  function val(id) {
    var el = $(id);
    if (el.validity && el.validity.badInput) return NaN;
    var v = el.value.trim();
    return v === "" ? null : Number(v);
  }

  function readBody() {
    if ($("body-skip").checked) return { answered: true, height_cm: null, weight_kg: null };
    var hcm, wkg, msg = "", bad = false;
    if (state.units === "metric") { hcm = val("h-cm"); wkg = val("w-kg"); }
    else {
      var ft = val("h-ft"), inch = val("h-in"), lb = val("w-lb");
      bad = (ft !== null && (isNaN(ft) || ft < 3 || ft > 8 || ft !== Math.floor(ft))) ||
            (inch !== null && (isNaN(inch) || inch < 0 || inch >= 12));
      hcm = ft === null && inch === null ? null : ((ft || 0) * 12 + (inch || 0)) * 2.54;
      wkg = lb === null ? null : lb * 0.45359237;
    }
    var L = PR.limits;
    if (bad || (hcm !== null && (isNaN(hcm) || hcm < L.height_cm[0] || hcm > L.height_cm[1]))) msg = "Please check the height.";
    else if (wkg !== null && (isNaN(wkg) || wkg < L.weight_kg[0] || wkg > L.weight_kg[1])) msg = "Please check the weight.";
    else if (hcm !== null && wkg !== null) {
      var bmi = wkg / Math.pow(hcm / 100, 2);
      if (bmi < L.bmi[0] || bmi > L.bmi[1]) msg = "Please check: these give a body-mass index of " + Math.round(bmi) + ".";
      else return { answered: true, height_cm: hcm, weight_kg: wkg, bmi: bmi };
    }
    return { answered: false, height_cm: null, weight_kg: null, msg: msg, partial: hcm !== null || wkg !== null };
  }

  function readAll() {
    var a = {}, done = {};
    var age = val("age");
    var ageOk = age !== null && !isNaN(age) && age === Math.floor(age) && age >= 1 && age <= 120;
    a.age = ageOk ? age : null; done.age = ageOk;
    ["natural_teeth", "sex", "education", "smoking", "diabetes", "gum_health", "gum_treatment", "loose_teeth",
     "bone_loss", "tooth_doesnt_look_right", "floss_days", "mouthwash_days"].forEach(function (k) {
      var v = readRadio(k);
      a[k] = (k === "floss_days" || k === "mouthwash_days") && v !== null ? Number(v) : v;
      done[k] = v !== null;
    });
    var body = readBody();
    a.height_cm = body.height_cm; a.weight_kg = body.weight_kg; done.body = body.answered;
    return { a: a, done: done, body: body, ageRaw: age };
  }

  /* ---------- update loop ---------- */
  function setDisabled(sectionIds, off) {
    sectionIds.forEach(function (id) {
      var s = $(id);
      if (off) { s.setAttribute("inert", ""); s.setAttribute("aria-disabled", "true"); }
      else { s.removeAttribute("inert"); s.removeAttribute("aria-disabled"); }
    });
  }

  function update(fromUser) {
    var r = readAll(), a = r.a, done = r.done;
    answers = a;

    // age message
    var ageMsg = $("age-msg"), ageIn = $("age");
    if (r.ageRaw !== null && !done.age) {
      ageMsg.textContent = isNaN(r.ageRaw) || r.ageRaw !== Math.floor(r.ageRaw) ? "Please enter your age as a whole number of years." : "Please enter an age between 30 and 120.";
      ageMsg.className = "msg error"; ageIn.setAttribute("aria-invalid", "true");
    }
    else if (done.age && a.age > 80) { ageMsg.textContent = "Ages above 80 are treated as 80, as in the survey."; ageMsg.className = "msg"; ageIn.removeAttribute("aria-invalid"); }
    else { ageMsg.textContent = ""; ageMsg.className = "msg"; ageIn.removeAttribute("aria-invalid"); }

    // body message
    var bm = $("body-msg");
    bm.textContent = r.body.msg || ""; bm.className = r.body.msg ? "msg error" : "msg";
    ["h-cm", "w-kg", "h-ft", "h-in", "w-lb"].forEach(function (id) {
      $(id).disabled = $("body-skip").checked;
      if (r.body.msg) $(id).setAttribute("aria-invalid", "true"); else $(id).removeAttribute("aria-invalid");
    });

    // scope
    var scope = $("scope"), out = null;
    if (done.age && a.age < 30) out = "<b>PerioRisk is for adults aged 30 and over.</b> The survey’s gum examination was only done from age 30, so the model can’t give a reliable estimate for younger people. A regular dental check-up remains the best guide.";
    else if (a.natural_teeth === "no") out = "<b>PerioRisk is for people who still have natural teeth.</b> The model learned only from people with at least two. If you have implants or dentures, ask your dentist to check the gums and tissues around them at your regular visits.";
    if (out) { scope.innerHTML = out; scope.hidden = false; } else { scope.hidden = true; }
    setDisabled(["sec-you", "sec-mouth"], !!out);

    // progress
    var n = ORDER.filter(function (k) { return done[k]; }).length;
    var complete = n === ORDER.length && !out;
    $("progress-label").textContent = out ? "Not applicable — see the note above" : n + " of " + ORDER.length + " answered";
    $("progress-fill").style.width = (out ? 0 : n / ORDER.length * 100) + "%";
    var go = $("go");
    go.setAttribute("aria-disabled", complete ? "false" : "true");
    go.textContent = state.revealed ? "View my result" : "See my result";
    [["sec-start", ["age", "natural_teeth"]], ["sec-you", ["sex", "body", "education", "smoking", "diabetes"]],
     ["sec-mouth", ["gum_health", "gum_treatment", "loose_teeth", "bone_loss", "tooth_doesnt_look_right", "floss_days", "mouthwash_days"]]]
      .forEach(function (s) { $(s[0]).classList.toggle("done", s[1].every(function (k) { return done[k]; })); });

    if (state.revealed) {
      if (out) {                       // no longer applicable: take the old result away
        state.revealed = false; lastP = null;
        $("result").hidden = true; $("result").classList.remove("show");
        go.textContent = "See my result";
      } else if (complete) { render(fromUser); $("result").classList.remove("stale"); $("stale").hidden = true; }
      else { $("result").classList.add("stale"); $("stale").hidden = false; }
    }
    return complete;
  }

  /* ---------- result ---------- */
  function display(key, a) {
    if (key === "bmi") return "BMI " + a.bmi.toFixed(1);
    if (key === "age") return String(a.age);
    if (key === "floss_days" || key === "mouthwash_days") return a[key] + " of 7 days";
    var v = a[key];
    if (SHORT[v]) return SHORT[v];
    return v.charAt(0).toUpperCase() + v.slice(1);
  }

  function tips(a) {
    var t = [];
    if (a.smoking === "current") t.push("Smoking is one of the strongest risk factors for gum disease that you can change. Stopping improves gum health and how well treatment works.");
    if (a.diabetes === "yes" || a.diabetes === "borderline") t.push("Diabetes and gum disease affect each other. Good blood-sugar control helps your gums, and treating gum disease can help blood-sugar control.");
    if (a.floss_days !== null && a.floss_days < 3) t.push("Cleaning between your teeth once a day, with floss or small interdental brushes, removes plaque where a toothbrush can’t reach.");
    if (a.loose_teeth === "yes" || a.bone_loss === "yes") t.push("Loose teeth or bone loss around teeth are signs worth discussing with a dentist or periodontist soon.");
    if (a.tooth_doesnt_look_right === "yes") t.push("A tooth that doesn’t look right is worth showing to a dentist, whatever your estimate.");
    return t;
  }

  var lastP = null;
  function render(fromUser) {
    var res;
    try { res = PR.assess(MODEL, answers); } catch (e) { return; }
    var p = res.probability, band = BAND[res.band], el = $("result");
    el.style.setProperty("--band", band.color);
    var pct = Math.round(p * 100);
    $("pct").textContent = pct < 1 ? "<1" : String(pct);
    $("band-name").textContent = band.name;
    $("pin").style.left = (Math.min(Math.max(p, 0.01), 0.99) * 100).toFixed(1) + "%";
    var per100 = Math.round(res.bandObserved * 100);
    $("meaning").textContent = "Among people in the 2013–2014 survey whose answers put them in the " + band.group +
      ", about " + per100 + " in 100 had moderate or severe gum disease when a dentist examined them.";
    $("next").textContent = {
      low: "Keep up your regular dental check-ups. Gum disease often has no symptoms early on, so a dentist’s check is still the only way to be sure.",
      moderate: "At your next dental visit, ask for a gum (periodontal) check — a quick measurement of the small spaces between your gums and teeth.",
      high: "Book a dental visit and ask specifically for a full periodontal examination. Gum disease is treatable, and treating it early protects your teeth."
    }[res.band];

    // contributions
    var list = $("contrib"), shown = res.contributions.filter(function (c) { return Math.abs(c.change) >= 0.005; }).slice(0, 6);
    var maxAbs = Math.max(0.1, shown.reduce(function (m, c) { return Math.max(m, Math.abs(c.change)); }, 0));
    list.innerHTML = "";
    if (!shown.length) list.appendChild(h("li", {}, '<span class="fine">No single answer moved your estimate by more than half a point.</span>'));
    shown.forEach(function (c) {
      var up = c.change > 0, w = Math.abs(c.change) / maxAbs * 50;
      var li = h("li", { "class": up ? "up" : "down" });
      li.innerHTML = '<span class="what"><b>' + esc(LABELS[c.key]) + "</b><small>" + esc(display(c.key, res.answers)) + "</small></span>" +
        '<span class="bar2" aria-hidden="true"><i style="left:' + (up ? 50 : 50 - w) + "%;width:" + w + '%"></i></span>' +
        '<span class="val">' + (up ? "+" : "−") + (Math.abs(c.change) * 100).toFixed(1) +
        '<span aria-hidden="true"> pts</span><span class="sr-only"> percentage points ' + (up ? "higher" : "lower") + "</span></span>";
      list.appendChild(li);
    });

    var t = tips(res.answers), tl = $("tips");
    tl.innerHTML = "";
    t.forEach(function (s) { tl.appendChild(h("li", {}, esc(s))); });
    $("tips-wrap").hidden = !t.length;

    // printable summary
    var rows = ORDER.filter(function (k) { return k !== "natural_teeth"; }).map(function (k) {
      var ans;
      if (k === "body") ans = res.answers.bmi === null ? "Not given" : display("bmi", res.answers);
      else ans = display(k, res.answers);
      return "<tr><td>" + esc(Q[k].text) + "</td><td><b>" + esc(ans) + "</b></td></tr>";
    });
    $("print-answers").innerHTML = rows.join("");
    $("print-date").textContent = new Date().toLocaleDateString();
    $("print-meta").textContent = "PerioRisk model " + MODEL.model_version + " · " + new Date().toLocaleDateString() +
      " · hezarani.github.io/periorisk · An educational screening estimate, not a diagnosis.";

    if (fromUser && lastP !== null && Math.abs(lastP - p) > 1e-12) {
      var u = $("updated"); u.classList.add("on");
      clearTimeout(render.t); render.t = setTimeout(function () { u.classList.remove("on"); }, 1400);
      $("announce").textContent = "Estimate updated: " + $("pct").textContent + " percent, " + band.name.toLowerCase() + ".";
    }
    lastP = p;
  }

  function reveal() {
    var complete = update(false);
    if (!complete) {
      var first = ORDER.filter(function (k) { return !readAll().done[k]; })[0];
      var target = first === "age" ? $("age") : $("q-" + first);
      if (target) {
        target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
        var focusable = first === "age" ? $("age") : Array.prototype.filter.call(target.querySelectorAll("input"), function (i) {
          return !i.disabled && i.offsetParent !== null;
        })[0];
        if (focusable) setTimeout(function () { focusable.focus({ preventScroll: true }); }, reduceMotion ? 0 : 350);
      }
      return;
    }
    var el = $("result");
    if (!state.revealed) {
      state.revealed = true;
      el.hidden = false;
      $("pin").style.transition = "none"; $("pin").style.left = "0%";
      void el.offsetWidth;
      $("pin").style.transition = "";
      el.classList.add("show");
    }
    render(false);
    $("go").textContent = "View my result";
    el.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    el.focus({ preventScroll: true });
  }

  /* ---------- metrics from the model file ---------- */
  function fillMetrics() {
    var m = MODEL.performance, map = {
      auc: m.test_auc.toFixed(2),
      auc_ci: "(95% CI " + m.test_auc_ci[0].toFixed(2) + "–" + m.test_auc_ci[1].toFixed(2) + ")",
      sens30: Math.round(m.sensitivity_at_30 * 100) + "%", spec60: Math.round(m.specificity_at_60 * 100) + "%",
      ntest: m.n_test.toLocaleString("en-US"), oe: String(Math.round(m.o_e_ratio * 100)),
      oe_pop: String(Math.round(m.survey_weighted_o_e_ratio * 100)), version: MODEL.model_version
    };
    Array.prototype.forEach.call(document.querySelectorAll("[data-metric]"), function (el) {
      var k = el.getAttribute("data-metric");
      if (map[k] !== undefined) el.textContent = map[k];
    });
  }

  /* ---------- wire up ---------- */
  function init() {
    if (!MODEL || !PR) return;
    build();
    fillMetrics();
    form.addEventListener("input", function () { update(true); });
    form.addEventListener("change", function () { update(true); });
    form.addEventListener("submit", function (e) { e.preventDefault(); });
    form.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && e.target.tagName === "INPUT" && e.target.type !== "checkbox") e.preventDefault();
    });
    Array.prototype.forEach.call(form.querySelectorAll("[data-units]"), function (b) {
      b.addEventListener("click", function () {
        state.units = b.getAttribute("data-units");
        Array.prototype.forEach.call(form.querySelectorAll("[data-units]"), function (x) {
          x.setAttribute("aria-pressed", String(x === b));
        });
        $("body-metric").hidden = state.units !== "metric";
        $("body-us").hidden = state.units !== "us";
        update(true);
      });
    });
    $("go").addEventListener("click", reveal);
    $("print").addEventListener("click", function () { window.print(); });
    $("edit").addEventListener("click", function () {
      $("check").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
      setTimeout(function () { $("age").focus({ preventScroll: true }); }, reduceMotion ? 0 : 400);
    });
    $("reset").addEventListener("click", function () {
      form.reset();
      state.revealed = false; lastP = null;
      var el = $("result"); el.hidden = true; el.classList.remove("show", "stale"); $("stale").hidden = true;
      $("body-skip").checked = false;
      update(false);
      window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
      setTimeout(function () { $("age").focus({ preventScroll: true }); }, reduceMotion ? 0 : 400);
    });
    // hide the dock while the result is on screen, so it never covers it
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (en) { $("dock").classList.toggle("away", en.isIntersecting && state.revealed); });
      }, { threshold: 0 }).observe($("result"));
    }
    update(false);
  }
  init();
})();
