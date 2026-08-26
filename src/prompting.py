"""Prompt construction and output parsing for propaganda-technique
extraction with an instruction-tuned LLM (designed for Llama 3.1 8B Instruct
run locally via llama-cpp-python, but model-agnostic).
"""

import json
import re

from .codebook import CODEBOOK, build_codebook_text

VALID_KEYS = set(CODEBOOK.keys())

SYSTEM_PROMPT = """You are an assistant helping researchers annotate news text \
for specific propaganda techniques, for academic research on political \
polarization. You are precise, conservative, and only flag text that clearly \
matches one of the technique definitions below. If nothing in the text \
matches, return an empty list.

TECHNIQUES:
{codebook}

OUTPUT FORMAT:
Return ONLY a JSON array (no prose, no markdown fences) where each element is:
{{"quote": "<exact substring from the input text>", \
"technique": "<one of: {keys}>", \
"confidence": <float between 0 and 1>, \
"rationale": "<one short sentence>"}}

If no techniques are present, return an empty array: []
"""

USER_TEMPLATE = """Analyze the following news text and extract every span that \
matches one of the propaganda techniques defined above.

TEXT:
\"\"\"
{text}
\"\"\"

JSON array:"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        codebook=build_codebook_text(),
        keys=", ".join(sorted(VALID_KEYS)),
    )


def build_user_prompt(text: str) -> str:
    return USER_TEMPLATE.format(text=text.strip())


def build_chat_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(text)},
    ]


class ParseError(ValueError):
    pass


def _extract_json_array(raw: str) -> str:
    """Pull the first top-level JSON array out of a model response that may
    contain surrounding prose or markdown code fences."""
    raw = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ParseError(f"No JSON array found in model output: {raw[:200]!r}")
    return raw[start : end + 1]


def parse_extractions(raw: str) -> list[dict]:
    """Parse and validate the model's JSON output. Raises ParseError on
    malformed structure so the caller can retry."""
    candidate = _extract_json_array(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    if not isinstance(data, list):
        raise ParseError("Top-level JSON is not a list")

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            raise ParseError(f"Array element is not an object: {item!r}")
        quote = item.get("quote")
        technique = item.get("technique")
        if not isinstance(quote, str) or not quote.strip():
            raise ParseError(f"Missing/empty 'quote' field: {item!r}")
        if technique not in VALID_KEYS:
            raise ParseError(f"Unknown technique key {technique!r}: {item!r}")
        confidence = item.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        cleaned.append(
            {
                "quote": quote.strip(),
                "technique": technique,
                "confidence": confidence,
                "rationale": item.get("rationale", ""),
            }
        )
    return cleaned
