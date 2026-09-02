"""LLM-based detection of "erasure of complexities" -- the collapse of
plural, multidimensional political/social identity into a single,
seemingly natural and inescapable two-camp antagonism -- in a full
article. A single-call, article-level companion to the three non-LLM
proxies in src/bipolarity.py, so all four approaches can be compared on
the same articles.

This is a DIFFERENT, more theoretically specific construct than the
`black_and_white` SemEval-style logical fallacy in src/codebook.py (which
is about presenting only two OPTIONS on a specific issue). The definition
here is deliberately kept separate rather than reusing/editing
src/codebook.py, and is based on the following theoretical account of
destructive polarization (paraphrased in the prompt below):

  In communicating with others, individuals normally navigate a complex
  social landscape with multiple, cross-cutting identities and ideologies.
  "Erasure of complexities" is the process by which this plurality gets
  amalgamated into a SINGLE dimension: a group's own identity comes to be
  understood as opposed by everyone else, or political life comes to be
  seen as naturally and inescapably defined by exactly two overarching,
  opposing partisan camps. This can happen because one distinction becomes
  so prominent it subsumes all other identity/ideological characteristics
  -- groups start identifying each other exclusively by the one trait that
  makes them different, absorbed into all-encompassing "us" and "them"
  identities. A specific, concrete manifestation: citing a stark,
  attention-grabbing statistic or claim about partisan division as if it
  proves the two sides are simply, naturally, totally opposed, WITHOUT the
  surrounding nuance/context that would show the real picture is more
  mixed. This is distinct from ordinary reporting of an actual two-choice
  situation, from balanced reporting that quotes two opposing views on one
  issue, and from describing a real two-party political system -- none of
  those are erasure unless the text also treats that binary as exhaustive
  of who the people/groups involved ARE, not just what they think about
  one issue.

Unlike src/bipolarity.py, this is NOT CPU-only: it requires an LLM (this
project uses Llama 3.1 8B Instruct via llama-cpp-python -- see
notebooks/colab_propaganda_poc.ipynb for the model-loading setup).
"""

import json
import re

# DEFINITION = (
#     "The collapse of a person's or group's plural, multidimensional identity "
#     "and ideology into a single dimension -- such that a group's own "
#     "identity is understood as opposed by everyone else, or political life "
#     "is depicted as naturally and inescapably defined by exactly two "
#     "overarching, opposing partisan camps. This often happens because one "
#     "distinction (e.g. a single hot-button issue, or party label) becomes "
#     "so prominent that it subsumes all other identity and ideological "
#     "characteristics: people or groups are described/treated as if that one "
#     "distinction is the only thing that defines them, absorbed into "
#     "all-encompassing 'us' vs. 'them' identities rather than as people with "
#     "many overlapping, sometimes cross-cutting views and affiliations."
# )

DEFINITION = (
  "The description of topics, groups, parties or issues as unidimensional"
  "and/or being to the exclusion of all others, resulting in us vs. them dynamics"
  "and identities rather than as people with many overlapping,"
  "sometimes cross-cutting views and affiliations."
  "This often happens because one distinction"
  "(e.g. a single hot-button issue, or party label) becomes so prominent"
  "that it subsumes all other identity and ideological characteristics."
  "This includes the implicit exclusion of other positions or identities
  "through statements of being the sole holder or legitimacy, moral authority or feasibility."
  
)

DISTINCTIONS = [
    "Distinction from reporting a real two-choice situation or an actual "
    "two-party/two-option institutional fact (e.g. 'the vote passed or "
    "failed', 'the two major parties are X and Y'): that is NOT this "
    "pattern by itself. It becomes this pattern only if the text ALSO "
    "treats that binary as exhaustive of who the people/groups involved."
    "Distinction from ordinary balanced reporting that quotes two opposing "
    "viewpoints on ONE issue: presenting two sides of a specific debate is "
    "NOT erasure of complexity unless the article implies those two "
    "camps also define the people/groups more broadly (their whole "
    "identity, character, or worth), not just their opinion on that issue.",
    "The mirror image is a guardrail, not an instance: if the text DOES "
    "provide the nuancing context around such a statistic or claim (e.g. "
    "'but a closer look shows...', noting exceptions, overlap, or "
    "cross-cutting cases), that is the OPPOSITE of erasure."
]

SYSTEM_PROMPT = """You are an assistant helping researchers analyze news \
articles for academic research on destructive political polarization. You \
are precise and conservative: only report an instance when the text \
clearly matches the definition below AND none of the guardrails apply. \
When in doubt, don't report it.

CONCEPT:{definition}

GUARDRAILS:
{distinctions}

You will be given a full news article. Find every instance anywhere in \
the article that matches the definiton. For each instance, quote the exact \
substring from the article that triggered it and give a one-sentence \
rationale explaining why the guardrails don't rule it out. The same \
pattern may appear more than once; report every distinct instance.

OUTPUT FORMAT:
Return ONLY a single JSON object (no prose, no markdown fences) shaped \
exactly like this:
{{
  "instances": [
    {{"quote": "<exact substring from the article>", "rationale": "<one sentence>"}}
  ]
}}
If no instances are found, return {{"instances": []}}.
"""

USER_TEMPLATE = """ARTICLE:
\"\"\"
{text}
\"\"\"

JSON object:"""


def _build_system_prompt() -> str:
    distinctions = "\n".join(f"- {d}" for d in DISTINCTIONS)
    return SYSTEM_PROMPT.format(definition=DEFINITION, distinctions=distinctions)


def build_dichotomy_messages(article_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": USER_TEMPLATE.format(text=article_text.strip())},
    ]


class ParseError(ValueError):
    pass


def _extract_json_object(raw: str) -> str:
    raw = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ParseError(f"No JSON object found in model output: {raw[:200]!r}")
    return raw[start : end + 1]


def parse_dichotomy_result(raw: str) -> list[dict]:
    candidate = _extract_json_object(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ParseError("Top-level JSON is not an object")
    instances_raw = data.get("instances")
    if not isinstance(instances_raw, list):
        raise ParseError(f"Missing/invalid 'instances' list: {data!r}")

    instances = []
    for entry in instances_raw:
        if not isinstance(entry, dict):
            raise ParseError(f"Instance is not an object: {entry!r}")
        quote = entry.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            raise ParseError(f"Missing/empty 'quote': {entry!r}")
        instances.append({"quote": quote.strip(), "rationale": entry.get("rationale")})
    return instances


def analyze_dichotomy(generate_fn, article_text: str, max_retries: int = 2) -> list[dict]:
    last_err = None
    for _ in range(max_retries):
        raw = generate_fn(build_dichotomy_messages(article_text))
        try:
            return parse_dichotomy_result(raw)
        except ParseError as e:
            last_err = e
    raise last_err


def compute_llm_dichotomy_row(
    document_id: str, title: str, text: str, generate_fn, max_retries: int = 2
) -> dict:
    word_count = len(text.split())
    try:
        instances = analyze_dichotomy(generate_fn, text, max_retries=max_retries)
        parse_error = None
    except ParseError as e:
        instances = []
        parse_error = str(e)

    density = 1000 * len(instances) / word_count if word_count else None
    return {
        "document_id": document_id,
        "title": title,
        "llm_dichotomy_instance_count": len(instances),
        "llm_dichotomy_density": density,
        "llm_dichotomy_instances": instances,
        "llm_dichotomy_parse_error": parse_error,
    }


def compute_llm_dichotomy_table(articles: list[dict], generate_fn, max_retries: int = 2) -> list[dict]:
    return [
        compute_llm_dichotomy_row(
            a["document_id"], a["title"], a["full_text"], generate_fn, max_retries=max_retries
        )
        for a in articles
    ]
