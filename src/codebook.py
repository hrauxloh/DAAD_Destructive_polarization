"""Codebook for 2 "complexity-erasing" language features (a subset of the
SemEval / Da San Martino et al. propaganda taxonomy), currently narrowed to
black-and-white / dichotomous reasoning and thought-terminating clichés.
Each entry follows the same style as the project's stereotyping codebook: a
simplified definition plus explicit "distinction from X" guardrails against
adjacent concepts, to keep the model from flagging surface patterns (a short
sentence, a rejection of blame) that don't actually fit the definition.

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
    distinctions: list[str] = field(default_factory=list)
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)


# Maps the technique labels used in oversimplification_data.csv to the
# internal codebook keys. Only techniques currently in CODEBOOK are mapped;
# rows with any other label are skipped by src/eval.py.
CSV_TECHNIQUE_TO_KEY: dict[str, str] = {
    "Black-and-White_Fallacy": "black_and_white",
    "Thought-terminating_Cliches": "thought_terminating_cliche",
}


CODEBOOK: dict[str, Technique] = {
    "black_and_white": Technique(
        key="black_and_white",
        name="Black-and-white fallacy / dichotomous reasoning",
        definition=(
            "Presenting only two alternatives or outcomes on a complex issue "
            "as if they were the only possibilities, when in fact a range of "
            "other options, nuances, or outcomes exists. In its extreme form "
            "('dictatorship'), it also prescribes one specific action as the "
            "only acceptable course, eliminating any other choice."
        ),
        distinctions=[
            "Distinction from a genuinely binary situation: reporting an "
            "actual binary state of affairs (a bill passed or it didn't) is "
            "NOT this fallacy. The fallacy requires that real alternatives "
            "are being erased or ignored, not that the situation described "
            "is truly binary.",
            "A statement that explicitly REJECTS blame or a binary framing "
            "(e.g. 'there is no villain here, just several factors') is the "
            "opposite of this technique, not an instance of it.",
        ],
        positive_examples=[
            "Either you support this bill or you don't care about children.",
            "There is no other choice: we must act now, exactly as proposed.",
            "You're either a patriot or a traitor.",
        ],
        negative_examples=[
            "Lawmakers debated several amendments before reaching a compromise.",
            "There is no 'baddie' here, just outdated governance arrangements "
            "that need to be reviewed.",
        ],
    ),
    "thought_terminating_cliche": Technique(
        key="thought_terminating_cliche",
        name="Thought-terminating cliché",
        definition=(
            "A short, generic, ready-made phrase that shuts down further "
            "inquiry or debate on a genuinely complex question by offering "
            "an easy, final-sounding answer, rather than engaging with the "
            "substance."
        ),
        distinctions=[
            "Distinction from ordinary concise language: brevity alone is "
            "not the marker. The phrase must function to END discussion or "
            "critical thought on something complex (e.g. 'it is what it "
            "is', 'boys will be boys'), not simply be a short sentence that "
            "conveys real information (e.g. 'the vote passed 54-46' is short "
            "but is not a cliché-shutdown).",
            "A routine announcement of a plan or policy focus area is not "
            "this technique merely because it is brief or uses confident "
            "phrasing — it must be a generic phrase that forecloses "
            "further thought, not a substantive (if short) claim.",
        ],
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
        for d in tech.distinctions:
            lines.append(f"   {d}")
        for ex in tech.positive_examples:
            lines.append(f"   Example (positive): \"{ex}\"")
        for ex in tech.negative_examples:
            lines.append(f"   Example (NOT this technique): \"{ex}\"")
        lines.append("")
    return "\n".join(lines)
