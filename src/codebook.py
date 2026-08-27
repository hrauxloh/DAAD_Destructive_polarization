"""Codebook for 4 "complexity-erasing" language features (a subset of the
SemEval / Da San Martino et al. propaganda taxonomy). Each entry follows the
same style as the project's stereotyping codebook: a simplified definition
plus explicit "distinction from X" guardrails against adjacent concepts, to
keep the model from flagging surface patterns (a single-cause sentence, a
short sentence, a rejection of blame) that don't actually fit the definition.

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
# internal codebook keys. reductio_ad_hitlerum has no labeled rows in that
# file (yet), but is still part of the codebook/prompt.
CSV_TECHNIQUE_TO_KEY: dict[str, str] = {
    "Causal_Oversimplification": "causal_oversimplification",
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
            "Distinction from causal oversimplification: this technique is "
            "about the number of OPTIONS or OUTCOMES presented (either/or), "
            "not the number of CAUSES attributed to an event. A sentence can "
            "name a single cause without presenting a binary choice, and "
            "vice versa.",
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
    "causal_oversimplification": Technique(
        key="causal_oversimplification",
        name="Causal oversimplification",
        definition=(
            "Attributing a complex outcome to a single cause or a single "
            "actor in a way that shuts down consideration of other genuinely "
            "contested or contributing factors, including scapegoating "
            "(assigning blame to one person or group without examining the "
            "fuller picture)."
        ),
        distinctions=[
            "Distinction from citing an established/consensus cause: naming "
            "a well-evidenced, scientifically or historically established "
            "primary driver of a phenomenon (e.g. 'greenhouse gas emissions "
            "are the main driver of global warming') is NOT oversimplification "
            "— that is accurate causal attribution backed by evidence, not "
            "the erasure of real alternative causes. The fallacy requires "
            "that the claim glosses over causes that are genuinely contested "
            "or multiple among experts.",
            "Distinction from black-and-white fallacy: see that entry — "
            "this technique is about the number of CAUSES, not the number of "
            "options/outcomes.",
        ],
        positive_examples=[
            "The economy collapsed because of one man's greed.",
            "Crime is rising simply because judges are too soft.",
            "Immigrants are the reason wages have stagnated.",
        ],
        negative_examples=[
            "Economists point to several factors behind the slowdown, "
            "including trade policy, energy prices, and consumer demand.",
            "These impacts will keep growing as long as we burn coal, oil, "
            "and natural gas, scientists say, citing decades of climate data.",
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
    "reductio_ad_hitlerum": Technique(
        key="reductio_ad_hitlerum",
        name="Reductio ad Hitlerum",
        definition=(
            "Discrediting an idea, policy, or person by associating it with "
            "a group or figure the target audience already despises or "
            "holds in contempt (not limited to actual references to Hitler "
            "or Nazis), instead of engaging with the argument on its merits."
        ),
        distinctions=[
            "Distinction from a substantive historical comparison: a "
            "specific, evidence-based comparison drawn to illuminate an "
            "actual mechanism or precedent (e.g. citing a documented "
            "historical parallel with sourced detail) is NOT this fallacy. "
            "The fallacy is the rhetorical move of guilt-by-association used "
            "to end debate, not analysis that engages with specifics.",
        ],
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
        for d in tech.distinctions:
            lines.append(f"   {d}")
        for ex in tech.positive_examples:
            lines.append(f"   Example (positive): \"{ex}\"")
        for ex in tech.negative_examples:
            lines.append(f"   Example (NOT this technique): \"{ex}\"")
        lines.append("")
    return "\n".join(lines)
