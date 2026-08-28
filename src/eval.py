"""Rough evaluation of the two-step article-level pipeline against the real
labeled spans in oversimplification_data.csv (SemEval-derived propaganda
annotations). This is a proof-of-concept sanity check, not a rigorous
metric: each row's `span_text` is run through the full two-step pipeline as
if it were a standalone "article", and we check whether any extracted
instance names the gold technique. The dataset has no explicit
non-propaganda ("none") examples, so this only measures recall/technique-
confusion, not false-positive rate. Deprioritized relative to running the
pipeline over real articles (see the notebook, section 8).
"""

import csv
import random
from pathlib import Path

from .codebook import CSV_TECHNIQUE_TO_KEY
from .prompting import ParseError, analyze_article

DATA_PATH = Path(__file__).resolve().parent.parent / "oversimplification_data.csv"


def load_labeled_examples(path: Path = DATA_PATH, sample_size: int | None = None, seed: int = 0) -> list[dict]:
    """Load rows from oversimplification_data.csv, mapping each row's CSV
    technique label to the internal codebook key. Rows with an unrecognized
    technique label are skipped. Pass `sample_size` to randomly subsample
    (useful since a local 8B model is slow, and each example now costs 2
    LLM calls)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    examples = []
    for row in rows:
        key = CSV_TECHNIQUE_TO_KEY.get(row["technique"])
        if key is None:
            continue
        examples.append({"id": row["article_id"], "text": row["span_text"], "technique": key})

    if sample_size is not None and sample_size < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, sample_size)

    return examples


def evaluate(generate_fn, examples: list[dict] | None = None, max_retries: int = 2) -> dict:
    """`generate_fn(messages: list[dict]) -> str` should call the LLM and
    return its raw text response for the given chat messages."""
    examples = examples if examples is not None else load_labeled_examples()

    correct = 0
    wrong_technique = 0
    missed = 0
    parse_failures = 0
    details = []

    for ex in examples:
        text, gold = ex["text"], ex["technique"]

        try:
            result = analyze_article(generate_fn, text, max_retries=max_retries)
        except ParseError as e:
            parse_failures += 1
            details.append({"id": ex["id"], "text": text, "gold": gold, "result": f"PARSE_FAIL: {e}"})
            continue

        found_techniques = {inst["technique"] for inst in result["instances"]}

        if gold in found_techniques:
            correct += 1
            details.append({"id": ex["id"], "text": text, "gold": gold, "result": "OK"})
        elif found_techniques:
            wrong_technique += 1
            details.append({"id": ex["id"], "text": text, "gold": gold, "result": f"WRONG_TECHNIQUE: {found_techniques}"})
        else:
            missed += 1
            details.append({"id": ex["id"], "text": text, "gold": gold, "result": "MISSED (no instances)"})

    n = len(examples)
    summary = {
        "n_examples": n,
        "correct": correct,
        "wrong_technique": wrong_technique,
        "missed": missed,
        "parse_failures": parse_failures,
        "accuracy": correct / n if n else 0.0,
    }
    return {"summary": summary, "details": details}


def print_report(result: dict) -> None:
    s = result["summary"]
    print("=== Evaluation summary ===")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print("\n=== Per-example detail ===")
    for d in result["details"]:
        print(f"[{d['id']}] gold={d['gold']:<28} -> {d['result']}   \"{d['text'][:70]}\"")
