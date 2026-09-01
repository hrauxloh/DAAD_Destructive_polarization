"""Codebook for 2 "complexity-erasing" language features (a subset of the
SemEval / Da San Martino et al. propaganda taxonomy), currently narrowed to
black-and-white / dichotomous reasoning and thought-terminating clichés.
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
            "There’s only two options he has to achieve this, and one is to massively cut the services that South Australians rely on from tje state government or to increase taxes.",
            "We either stand squarely alongside the people of our democratic sister country of Israel, or we support those who want to wipe Israel from the map of the Middle East.",
            "either replace non-dairy cattle or limit emissions",
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
            "Distinction from ordinary concise language."
            "The phrase must function to END discussion or critical thought."
        ],
        positive_examples=[
            "It is what it is.",
            "Let's not rest until we get this done.",
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
