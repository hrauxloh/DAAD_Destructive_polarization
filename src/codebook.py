"""Codebook for a 7-technique subset of the SemEval / Da San Martino et al.
propaganda taxonomy. Each entry gives the definition used for annotation and
the few-shot examples fed to the LLM prompt.

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


CODEBOOK: dict[str, Technique] = {
    "name_calling": Technique(
        key="name_calling",
        name="Name calling or labeling",
        definition=(
            "Labeling the object of the propaganda campaign as something the "
            "target audience fears, hates, finds undesirable, or loves and "
            "praises, without argument."
        ),
        positive_examples=[
            "The so-called 'reformer' is nothing but a corrupt puppet.",
            "These radical extremists want to destroy our way of life.",
            "Our brave patriots stood firm against the tyrants.",
        ],
        negative_examples=[
            "The senator voted against the bill on Tuesday.",
            "Critics argue the policy will raise costs by 4%.",
        ],
    ),
    "exaggeration_minimization": Technique(
        key="exaggeration_minimization",
        name="Exaggeration or minimization",
        definition=(
            "Representing something in an excessive manner (making it larger, "
            "better, or worse than it is), or minimizing it (making it seem "
            "less important or smaller than it actually is)."
        ),
        positive_examples=[
            "This is the single greatest catastrophe in human history.",
            "It was just a minor scuffle, nothing to worry about at all.",
            "Millions upon millions will suffer if this law passes.",
        ],
        negative_examples=[
            "The storm caused $2 million in damage across three counties.",
            "Turnout was slightly lower than in the previous election.",
        ],
    ),
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
    "slogans": Technique(
        key="slogans",
        name="Slogans",
        definition=(
            "A brief, striking, memorable phrase, often used as an emotional "
            "appeal, that may include labeling or stereotyping."
        ),
        positive_examples=[
            "Build the wall!",
            "Take back our country!",
            "Justice for all, corruption for none.",
        ],
        negative_examples=[
            "The committee will reconvene next Thursday to review the draft.",
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
    "reductio_ad_hitlerum": Technique(
        key="reductio_ad_hitlerum",
        name="Reductio ad Hitlerum",
        definition=(
            "Discrediting an idea or action by associating it with a group or "
            "figure the audience already despises, rather than addressing "
            "the idea on its merits (not limited to actual references to "
            "Hitler or Nazis)."
        ),
        positive_examples=[
            "This policy is exactly what dictators have always used to "
            "control the population.",
            "Only fascists have ever supported this kind of censorship.",
            "That argument sounds like something straight out of a "
            "totalitarian playbook.",
        ],
        negative_examples=[
            "Historians compared the policy to similar measures adopted in "
            "the 1930s, citing three peer-reviewed studies.",
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
