"""Two-step, article-level prompt construction and output parsing for
detecting complexity-erasing language features, with an instruction-tuned
LLM (designed for Llama 3.1 8B Instruct run locally via llama-cpp-python,
but model-agnostic).

Mirrors the project's stereotyping methodology, at article granularity:
  Step 1 - extract the discrete claims/assertions made across the WHOLE
           article (one LLM call, technique-agnostic).
  Step 2 - given the article and the Step 1 claims, pull out every concrete
           INSTANCE where one of the techniques appears: an exact quote, the
           claim it operates on, and a rationale grounded in the codebook's
           "distinction from X" guardrails (a second LLM call).
"""

import json
import re

from .codebook import CODEBOOK, build_codebook_text

VALID_KEYS = list(CODEBOOK.keys())

# ---------------------------------------------------------------------------
# Step 1: article-level claim extraction (technique-agnostic)
# ---------------------------------------------------------------------------

STEP1_SYSTEM_PROMPT = """You are an assistant helping researchers analyze \
news articles for academic research on political polarization.

You will be given a full news article. Identify the discrete claims, \
assertions, or arguments it makes — one short phrase per distinct claim, \
covering the whole article, not just the opening paragraph.

Return ONLY a single JSON object (no prose, no markdown fences) shaped \
exactly like this:
{
  "claims": ["<claim 1>", "<claim 2>", ...]
}
"""

STEP1_USER_TEMPLATE = """ARTICLE:
\"\"\"
{text}
\"\"\"

JSON object:"""


def build_step1_messages(article_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STEP1_SYSTEM_PROMPT},
        {"role": "user", "content": STEP1_USER_TEMPLATE.format(text=article_text.strip())},
    ]


def parse_step1_result(raw: str) -> list[str]:
    """Raises ParseError on malformed structure so the caller can retry."""
    data = _load_json_object(raw)
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise ParseError(f"Missing/invalid 'claims' list: {data!r}")
    return [str(c) for c in claims]


# ---------------------------------------------------------------------------
# Step 2: article-level technique instance extraction
# ---------------------------------------------------------------------------

STEP2_SYSTEM_PROMPT = """You are an assistant helping researchers analyze \
news articles for language features that erase complexity in public debate, \
for academic research on political polarization. You are precise and \
conservative: only report an instance when the text clearly matches a \
technique's definition AND none of its "distinction from X" guardrails \
apply. When in doubt, don't report it.

TECHNIQUES:
{codebook}

You will be given a full news article and the list of claims it makes \
(from a prior analysis step). Find every instance anywhere in the article \
where one of the {n_techniques} technique(s) above (keys: {keys}) appears. \
For each instance: quote the exact substring from the article that \
triggered it, name which claim (from the list below) it operates on, name \
the technique key, and give a one-sentence rationale explaining why the \
guardrails don't rule it out. The same technique may appear more than once \
in the article; report every distinct instance.

CLAIMS FROM STEP 1:
{claims}

OUTPUT FORMAT:
Return ONLY a single JSON object (no prose, no markdown fences) shaped \
exactly like this:
{{
  "instances": [
    {{"technique": "<one of: {keys}>", "quote": "<exact substring from the article>", "claim": "<the claim from the list above it operates on>", "rationale": "<one sentence>"}}
  ]
}}
If no instances are found, return {{"instances": []}}.
"""

STEP2_USER_TEMPLATE = """ARTICLE:
\"\"\"
{text}
\"\"\"

JSON object:"""


def build_step2_messages(article_text: str, claims: list[str]) -> list[dict[str, str]]:
    system = STEP2_SYSTEM_PROMPT.format(
        codebook=build_codebook_text(),
        keys=", ".join(VALID_KEYS),
        n_techniques=len(VALID_KEYS),
        claims="\n".join(f"- {c}" for c in claims) if claims else "(none extracted)",
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": STEP2_USER_TEMPLATE.format(text=article_text.strip())},
    ]


def parse_step2_result(raw: str) -> list[dict]:
    """Raises ParseError on malformed structure so the caller can retry."""
    data = _load_json_object(raw)
    instances_raw = data.get("instances")
    if not isinstance(instances_raw, list):
        raise ParseError(f"Missing/invalid 'instances' list: {data!r}")

    instances = []
    for entry in instances_raw:
        if not isinstance(entry, dict):
            raise ParseError(f"Instance is not an object: {entry!r}")
        technique = entry.get("technique")
        quote = entry.get("quote")
        if technique not in VALID_KEYS:
            raise ParseError(f"Unknown/missing technique key {technique!r}: {entry!r}")
        if not isinstance(quote, str) or not quote.strip():
            raise ParseError(f"Missing/empty 'quote': {entry!r}")
        instances.append(
            {
                "technique": technique,
                "quote": quote.strip(),
                "claim": entry.get("claim"),
                "rationale": entry.get("rationale"),
            }
        )
    return instances


# ---------------------------------------------------------------------------
# Combined driver
# ---------------------------------------------------------------------------


def analyze_article(generate_fn, article_text: str, max_retries: int = 2) -> dict:
    """Runs both steps against `generate_fn(messages) -> raw_text` and
    returns {"claims": [...], "instances": [...]}."""
    claims = None
    last_err = None
    for _ in range(max_retries):
        raw = generate_fn(build_step1_messages(article_text))
        try:
            claims = parse_step1_result(raw)
            break
        except ParseError as e:
            last_err = e
    if claims is None:
        raise last_err

    instances = None
    last_err = None
    for _ in range(max_retries):
        raw = generate_fn(build_step2_messages(article_text, claims))
        try:
            instances = parse_step2_result(raw)
            break
        except ParseError as e:
            last_err = e
    if instances is None:
        raise last_err

    return {"claims": claims, "instances": instances}


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


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


def _load_json_object(raw: str) -> dict:
    candidate = _extract_json_object(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ParseError("Top-level JSON is not an object")
    return data
