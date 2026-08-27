"""Rough evaluation of the two-step paragraph pipeline against the real
labeled spans in oversimplification_data.csv (SemEval-derived propaganda
annotations). This is a proof-of-concept sanity check, not a rigorous
metric: each row's `span_text` is used directly as the "paragraph" input,
and we check whether the model's Step 2 YES/NO for the gold technique came
back YES. The dataset has no explicit non-propaganda ("none") examples, so
this only measures recall/technique-confusion, not false-positive rate.
"""

import csv
import random
from pathlib import Path

from .codebook import CSV_TECHNIQUE_TO_KEY
from .prompting import ParseError, build_chat_messages, parse_paragraph_result

DATA_PATH = Path(__file__).resolve().parent.parent / "oversimplification_data.csv"


def load_labeled_examples(path: Path = DATA_PATH, sample_size: int | None = None, seed: int = 0) -> list[dict]:
    """Load rows from oversimplification_data.csv, mapping each row's CSV
    technique label to the internal codebook key. Rows with an unrecognized
    technique label are skipped. Pass `sample_size` to randomly subsample
    (useful since a local 8B model is slow, and the full file has ~390 rows)."""
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
        messages = build_chat_messages(text)

        result = None
        last_err = None
        for _ in range(max_retries):
            raw = generate_fn(messages)
            try:
                result = parse_paragraph_result(raw)
                break
            except ParseError as e:
                last_err = e
                continue

        if result is None:
            parse_failures += 1
            details.append({"id": ex["id"], "text": text, "gold": gold, "result": f"PARSE_FAIL: {last_err}"})
            continue

        present_techniques = {k for k, v in result["techniques"].items() if v["present"]}

        if gold in present_techniques:
            correct += 1
            details.append({"id": ex["id"], "text": text, "gold": gold, "result": "OK"})
        elif present_techniques:
            wrong_technique += 1
            details.append({"id": ex["id"], "text": text, "gold": gold, "result": f"WRONG_TECHNIQUE: {present_techniques}"})
        else:
            missed += 1
            details.append({"id": ex["id"], "text": text, "gold": gold, "result": "MISSED (all NO)"})

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
