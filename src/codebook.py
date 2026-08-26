"""Codebook for a 3-technique subset of the SemEval / Da San Martino et al.
propaganda taxonomy, matching the techniques present in
`oversimplification_data.csv`. Each entry gives the definition used for
annotation and the few-shot examples fed to the LLM prompt.

Definitions are paraphrased from:
  Da San Martino et al., "Fine-Grained Analysis of Propaganda in News
  Articles" (EMNLP 2019) / SemEval-2020 Task 11.
"""

from dataclasses import dataclass, field


@dataclass
class Technique:
    key: str
    name: str
    definition: str
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)


# Maps the technique labels used in oversimplification_data.csv to the
# internal codebook keys.
CSV_TECHNIQUE_TO_KEY: dict[str, str] = {
    "Causal_Oversimplification": "causal_oversimplification",
    "Black-and-White_Fallacy": "black_and_white",
    "Thought-terminating_Cliches": "thought_terminating_cliche",
}


CODEBOOK: dict[str, Technique] = {
    "causal_oversimplification": Technique(
        key="causal_oversimplification",
        name="Causal oversimplification",
        definition=(
            "Assuming a single cause or reason for a complex issue that "
            "actually has multiple causes, including scapegoating a person "
            "or group without examining the full picture."
        ),
        positive_examples=[
            "The economy collapsed because of one man's greed.",
            "Crime is rising simply because judges are too soft.",
            "Immigrants are the reason wages have stagnated.",
        ],
        negative_examples=[
            "Economists point to several factors behind the slowdown, "
            "including trade policy, energy prices, and consumer demand.",
        ],
    ),
    "black_and_white": Technique(
        key="black_and_white",
        name="Black-and-white fallacy / dictatorship",
        definition=(
            "Presenting only two alternatives when more options exist "
            "('you're either with us or against us'), or, in the extreme "
            "case, telling the audience exactly what action to take with no "
            "room for alternatives."
        ),
        positive_examples=[
            "Either you support this bill or you don't care about children.",
            "There is no other choice: we must act now, exactly as proposed.",
            "You're either a patriot or a traitor.",
        ],
        negative_examples=[
            "Lawmakers debated several amendments before reaching a compromise.",
        ],
    ),
    "thought_terminating_cliche": Technique(
        key="thought_terminating_cliche",
        name="Thought-terminating cliché",
        definition=(
            "A short, generic phrase that discourages further critical "
            "thought or debate by offering a seemingly simple, final answer "
            "to a complex question."
        ),
        positive_examples=[
            "It is what it is.",
            "Boys will be boys.",
            "That's just the way things are, end of story.",
        ],
        negative_examples=[
            "The report outlines three possible explanations for the delay.",
        ],
    ),
}


def build_codebook_text() -> str:
    """Render the codebook as prompt-ready text."""
    lines = []
    for i, tech in enumerate(CODEBOOK.values(), start=1):
        lines.append(f"{i}. {tech.name} [key: {tech.key}]")
        lines.append(f"   Definition: {tech.definition}")
        for ex in tech.positive_examples:
            lines.append(f"   Example (positive): \"{ex}\"")
        for ex in tech.negative_examples:
            lines.append(f"   Example (NOT this technique): \"{ex}\"")
        lines.append("")
    return "\n".join(lines)
