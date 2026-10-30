// Helper for tests/test_web_engine.py: runs docs/periorisk.js under Node.
//   node run_engine.js features <model.json> <rows.json>   -> probabilities for model-input rows
//   node run_engine.js answers  <model.json> <rows.json>   -> assess() results for answer sets
//   node run_engine.js modeljs  <docs/model.js>             -> the model object embedded in model.js
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const PerioRisk = require(path.join(__dirname, "..", "..", "docs", "periorisk.js"));

const [mode, a, b] = process.argv.slice(2);
if (mode === "modeljs") {
  const sandbox = { window: {} };
  vm.runInNewContext(fs.readFileSync(a, "utf8"), sandbox);
  process.stdout.write(JSON.stringify(sandbox.window.PERIORISK_MODEL));
} else {
  const model = JSON.parse(fs.readFileSync(a, "utf8"));
  const rows = JSON.parse(fs.readFileSync(b, "utf8"));
  const out = rows.map((r) => {
    if (mode === "features") return PerioRisk.probability(model, r);
    try {
      const res = PerioRisk.assess(model, r);
      return { probability: res.probability, band: res.band, screen_positive: res.screenPositive,
               contributions: res.contributions, unanswered: res.unanswered };
    } catch (e) {
      return { error: e.code || "error", message: e.message };
    }
  });
  // full precision: JSON.stringify writes the shortest round-trip representation
  process.stdout.write(JSON.stringify(out));
}
