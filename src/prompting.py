"""Two-step prompt construction and output parsing for detecting
complexity-erasing language features, per paragraph, with an
instruction-tuned LLM (designed for Llama 3.1 8B Instruct run locally via
llama-cpp-python, but model-agnostic).

Mirrors the project's stereotyping methodology:
  Step 1 - extract the discrete claims/assertions made in the paragraph.
  Step 2 - for each technique, decide YES/NO whether the paragraph contains
           a clear instance, applying the codebook's "distinction from X"
           guardrails to avoid flagging surface patterns that don't actually
           fit the definition.
"""

import json
import re

from .codebook import CODEBOOK, build_codebook_text

VALID_KEYS = list(CODEBOOK.keys())

SYSTEM_PROMPT = """You are an assistant helping researchers analyze news text \
for language features that erase complexity in public debate, for academic \
research on political polarization. You are precise and conservative: only \
answer YES when the paragraph clearly matches a technique's definition AND \
none of its "distinction from X" guardrails apply. When in doubt, answer NO.

TECHNIQUES:
{codebook}

You will be given ONE paragraph. Perform two steps:

STEP 1 - Claim extraction: identify the discrete claims, assertions, or \
arguments made in the paragraph (short phrases, one per distinct claim).

STEP 2 - Technique identification: for each of the {n_techniques} technique(s) \
above (keys: {keys}), decide YES or NO whether the paragraph contains a clear \
instance of it. If YES, quote the exact substring from the paragraph that \
triggered it and give a one-sentence rationale that references which claim \
from Step 1 it operates on and why the guardrails don't rule it out.

OUTPUT FORMAT:
Return ONLY a single JSON object (no prose, no markdown fences) shaped \
exactly like this:
{{
  "claims": ["<claim 1>", "<claim 2>", ...],
  "techniques": {{
{techniques_schema}
  }}
}}
"""

USER_TEMPLATE = """PARAGRAPH:
\"\"\"
{text}
\"\"\"

JSON object:"""


def _build_techniques_schema() -> str:
    lines = [
        f'    "{key}": {{"present": true|false, "quote": "<exact substring or null>", "rationale": "<one sentence or null>"}}'
        for key in VALID_KEYS
    ]
    return ",\n".join(lines)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        codebook=build_codebook_text(),
        keys=", ".join(VALID_KEYS),
        n_techniques=len(VALID_KEYS),
        techniques_schema=_build_techniques_schema(),
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


def _extract_json_object(raw: str) -> str:
    """Pull the first top-level JSON object out of a model response that may
    contain surrounding prose or markdown code fences."""
    raw = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ParseError(f"No JSON object found in model output: {raw[:200]!r}")
    return raw[start : end + 1]


def parse_paragraph_result(raw: str) -> dict:
    """Parse and validate the model's two-step JSON output. Raises
    ParseError on malformed structure so the caller can retry."""
    candidate = _extract_json_object(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ParseError("Top-level JSON is not an object")

    claims = data.get("claims")
    if not isinstance(claims, list):
        raise ParseError(f"Missing/invalid 'claims' list: {data!r}")
    claims = [str(c) for c in claims]

    techniques_raw = data.get("techniques")
    if not isinstance(techniques_raw, dict):
        raise ParseError(f"Missing/invalid 'techniques' object: {data!r}")

    techniques = {}
    for key in VALID_KEYS:
        entry = techniques_raw.get(key)
        if not isinstance(entry, dict):
            raise ParseError(f"Missing/invalid entry for technique {key!r}: {data!r}")
        present = entry.get("present")
        if not isinstance(present, bool):
            raise ParseError(f"'present' for {key!r} is not a bool: {entry!r}")
        techniques[key] = {
            "present": present,
            "quote": entry.get("quote") if present else None,
            "rationale": entry.get("rationale") if present else None,
        }

    return {"claims": claims, "techniques": techniques}
