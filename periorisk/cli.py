"""Command line interface.

    periorisk predict --age 58 --sex male --smoking current --gum-health fair --floss-days 0
    periorisk batch answers.csv results.csv
    periorisk questions
    periorisk info
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from . import (OutOfScopeError, PerioRiskError, __version__, model_info, predict, questionnaire,
               recalibrate)
from .questions import QUESTIONS

DISCLAIMER = ("PerioRisk gives an educational screening estimate, not a diagnosis. "
              "Only a periodontal examination can confirm gum disease.")


def _add_answer_options(p: argparse.ArgumentParser) -> None:
    for q in QUESTIONS:
        flag = "--" + q.key.replace("_", "-")
        kw = {"dest": q.key, "default": None}
        if q.choices:   # validated (with friendly spelling, e.g. "not-sure") by periorisk.predict
            kw["metavar"] = "{" + ",".join(q.choices) + "}"
            kw["help"] = q.text
        elif q.key == "age":
            kw["type"] = float
            kw["required"] = True
            kw["help"] = "age in years (30 or older)"
        else:
            kw["type"] = float
            kw["help"] = q.text
        p.add_argument(flag, **kw)
    p.add_argument("--bmi", type=float, default=None, help="body-mass index (instead of height and weight)")


def _prevalence(text: str) -> float:
    try:
        x = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0.0 < x < 1.0:
        raise argparse.ArgumentTypeError("the prevalence must be between 0 and 1, e.g. 0.3 for 30%")
    return x


def _key(header: str) -> str:
    return header.strip().lower().replace("-", "_").replace(" ", "_")


def _cmd_predict(args) -> int:
    answers = {q.key: getattr(args, q.key) for q in QUESTIONS} | {"bmi": args.bmi}
    answers = {k: v for k, v in answers.items() if v is not None}
    r = predict(answers)
    adjusted = None if args.recalibrate_to is None else recalibrate(r.probability, args.recalibrate_to)
    if args.json:
        d = r.to_dict()
        if adjusted is not None:
            d |= {"adjusted_probability": adjusted, "adjusted_to_prevalence": args.recalibrate_to}
        print(json.dumps(d, indent=2, default=str))
    else:
        print(r.summary())
        if adjusted is not None:
            print(f"Adjusted for a population prevalence of {args.recalibrate_to:.0%} (research use): "
                  f"{adjusted * 100:.0f}%. The band, explanation and survey figures above refer to the "
                  "unadjusted estimate.")
    return 0


def _cmd_batch(args) -> int:
    keys = {q.key for q in QUESTIONS} | {"bmi"}
    with open(args.input, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = [f for f in (reader.fieldnames or []) if f is not None]
    mapping = {f: _key(f) for f in fields}                       # "Floss days" -> "floss_days"
    used = {f: k for f, k in mapping.items() if k in keys}
    if "age" not in used.values():
        raise PerioRiskError(f"{args.input} has no 'age' column; columns must be named like the "
                             "questions (see `periorisk questions`)")
    ignored = [f for f in fields if f not in used]
    if ignored:
        print("Note: these columns are not PerioRisk questions and were ignored: " + ", ".join(ignored),
              file=sys.stderr)
    out_fields = fields + ["probability", "percent", "band", "screen_positive", "error"]
    if args.recalibrate_to is not None:
        out_fields.insert(len(fields) + 1, "adjusted_probability")
    n_ok = 0
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            row.pop(None, None)                                  # values beyond the header row
            answers = {k: row.get(f) for f, k in used.items() if row.get(f) not in (None, "")}
            out = {f: row.get(f, "") for f in fields}
            try:
                r = predict(answers, explain=False)
                out |= {"probability": f"{r.probability:.6f}", "percent": round(r.probability * 100),
                        "band": r.band, "screen_positive": int(r.screen_positive), "error": ""}
                if args.recalibrate_to is not None:
                    out["adjusted_probability"] = f"{recalibrate(r.probability, args.recalibrate_to):.6f}"
                n_ok += 1
            except (PerioRiskError, ValueError, OverflowError) as err:
                out |= {"probability": "", "percent": "", "band": "", "screen_positive": "", "error": str(err)}
            w.writerow(out)
    print(f"{n_ok} of {len(rows)} rows scored -> {args.output}", file=sys.stderr)
    return 0


def _cmd_questions(args) -> int:
    for q in questionnaire():
        print(f"{q.key}: {q.text}")
        if q.choices:
            print("    answers: " + " | ".join(f"{k} ({v})" for k, v in q.choices.items()))
        if q.note:
            print(f"    note: {q.note}")
    return 0


def _cmd_info(args) -> int:
    print(json.dumps(model_info(), indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="periorisk", description=__doc__.split("\n")[0] + " " + DISCLAIMER,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"periorisk {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="estimate risk for one person")
    _add_answer_options(p)
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--recalibrate-to", type=_prevalence, default=None, metavar="PREVALENCE",
                   help="also report the estimate adjusted to a population prevalence between 0 and 1 (research use)")
    p.set_defaults(func=_cmd_predict)

    b = sub.add_parser("batch", help="score every row of a CSV file (columns = question keys)")
    b.add_argument("input")
    b.add_argument("output")
    b.add_argument("--recalibrate-to", type=_prevalence, default=None, metavar="PREVALENCE",
                   help="add an adjusted_probability column for a population prevalence between 0 and 1")
    b.set_defaults(func=_cmd_batch)

    sub.add_parser("questions", help="list the questions and allowed answers").set_defaults(func=_cmd_questions)
    sub.add_parser("info", help="model version, population and performance").set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except OutOfScopeError as err:
        print(f"Not applicable: {err}", file=sys.stderr)
        return 2
    except PerioRiskError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
