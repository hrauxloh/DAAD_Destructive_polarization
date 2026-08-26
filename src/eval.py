"""Rough evaluation of the extraction pipeline against the hand-labeled
sample set. This is a proof-of-concept sanity check, not a rigorous metric:
it checks whether the model flagged *any* span in a sentence with the
correct technique (sentence-level accuracy), and separately whether it
stayed quiet on sentences with no propaganda technique (specificity).
"""

import csv
from pathlib import Path

from .prompting import ParseError, build_chat_messages, parse_extractions

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_labeled_examples.csv"


def load_labeled_examples(path: Path = DATA_PATH) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate(generate_fn, examples: list[dict] | None = None, max_retries: int = 2) -> dict:
    """`generate_fn(messages: list[dict]) -> str` should call the LLM and
    return its raw text response for the given chat messages."""
    examples = examples or load_labeled_examples()

    correct = 0
    wrong_technique = 0
    missed = 0
    false_positive_on_none = 0
    parse_failures = 0
    details = []

    for ex in examples:
        text, gold = ex["text"], ex["technique"]
        messages = build_chat_messages(text)

        extractions = None
        last_err = None
        for _ in range(max_retries):
            raw = generate_fn(messages)
            try:
                extractions = parse_extractions(raw)
                break
            except ParseError as e:
                last_err = e
                continue

        if extractions is None:
            parse_failures += 1
            details.append({"id": ex["id"], "text": text, "gold": gold, "result": f"PARSE_FAIL: {last_err}"})
            continue

        predicted_techniques = {e["technique"] for e in extractions}

        if gold == "none":
            if predicted_techniques:
                false_positive_on_none += 1
                details.append({"id": ex["id"], "text": text, "gold": gold, "result": f"FALSE_POSITIVE: {predicted_techniques}"})
            else:
                correct += 1
                details.append({"id": ex["id"], "text": text, "gold": gold, "result": "OK (correctly quiet)"})
        else:
            if gold in predicted_techniques:
                correct += 1
                details.append({"id": ex["id"], "text": text, "gold": gold, "result": "OK"})
            elif predicted_techniques:
                wrong_technique += 1
                details.append({"id": ex["id"], "text": text, "gold": gold, "result": f"WRONG_TECHNIQUE: {predicted_techniques}"})
            else:
                missed += 1
                details.append({"id": ex["id"], "text": text, "gold": gold, "result": "MISSED (no extraction)"})

    n = len(examples)
    summary = {
        "n_examples": n,
        "correct": correct,
        "wrong_technique": wrong_technique,
        "missed": missed,
        "false_positive_on_none": false_positive_on_none,
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
