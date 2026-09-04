"""Plots the complexity indices (from aus_complexity_scores.csv, see
scripts/run_complexity.py) for every article, grouped and colored by
publisher, as a grid of small-multiple strip plots -- one panel per index.

Requires aus_complexity_scores.csv to already exist (run
scripts/run_complexity.py first) and australia_498sample_climatechange.csv
(for the `publisher` column) in the repo root.

Usage:
    python scripts/plot_complexity.py

Produces complexity_by_publisher.png in the repo root.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORES_CSV = REPO_ROOT / "aus_complexity_scores.csv"
ARTICLES_CSV = REPO_ROOT / "australia_498sample_climatechange.csv"
OUTPUT_PNG = REPO_ROOT / "complexity_by_publisher.png"

# Validated categorical palette (8 slots, fixed order -- see the dataviz
# skill's references/palette.md). Slot 8 (red) is reserved for the "Other"
# bucket so the top-7 named publishers get slots 1-7.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
OTHER_COLOR = "#898781"  # muted gray, not the reserved red -- "Other" isn't a peer category

# Secondary encoding (marker shape) so identity never rests on hue alone.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

METRICS = [
    ("flesch_reading_ease", "Flesch Reading Ease"),
    ("flesch_kincaid_grade", "Flesch-Kincaid Grade"),
    ("gunning_fog", "Gunning Fog"),
    ("coleman_liau_index", "Coleman-Liau Index"),
    ("automated_readability_index", "Automated Readability Index"),
    ("referential_cohesion", "Referential Cohesion"),
    ("causal_connective_density", "Causal Connective Density"),
    ("logical_connective_density", "Logical Connective Density"),
    ("temporal_connective_density", "Temporal Connective Density"),
    ("additive_connective_density", "Additive Connective Density"),
    ("mean_dependency_depth", "Mean Dependency Depth"),
]

TOP_N_PUBLISHERS = 7


def clean_publisher(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raw = ""
    raw = str(raw).strip()
    if not raw or raw.lower() == "none":
        return "Unknown"
    # Known data-entry typo in the source corpus.
    if raw == "Australian Broadca sting Corporation":
        return "Australian Broadcasting Corporation"
    return raw


def load_data() -> pd.DataFrame:
    scores = pd.read_csv(SCORES_CSV)
    with open(ARTICLES_CSV, newline="", encoding="utf-8") as f:
        articles = {row["document_id"]: row["publisher"] for row in csv.DictReader(f)}

    scores["publisher"] = scores["document_id"].map(articles).map(clean_publisher)
    return scores


def bucket_publishers(df: pd.DataFrame, top_n: int = TOP_N_PUBLISHERS) -> pd.DataFrame:
    counts = df["publisher"].value_counts()
    top = list(counts.head(top_n).index)
    df = df.copy()
    df["publisher_group"] = df["publisher"].where(df["publisher"].isin(top), "Other")
    return df, top


def plot(df: pd.DataFrame, top_publishers: list[str]) -> None:
    categories = top_publishers + (["Other"] if "Other" in df["publisher_group"].values else [])
    color_map = {pub: CATEGORICAL[i] for i, pub in enumerate(top_publishers)}
    color_map["Other"] = OTHER_COLOR
    marker_map = {pub: MARKERS[i % len(MARKERS)] for i, pub in enumerate(top_publishers)}
    marker_map["Other"] = MARKERS[-1]

    n_cols = 3
    n_rows = -(-len(METRICS) // n_cols)  # ceil division

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows), facecolor=SURFACE)
    axes = np.array(axes).reshape(-1)

    x_positions = {cat: i for i, cat in enumerate(categories)}
    rng = np.random.default_rng(0)

    n_clipped_total = 0

    for ax, (col, label) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        sub = df[["publisher_group", col]].dropna()

        # Robust y-limits: a single degenerate article (e.g. sentence-splitting
        # failure producing one 188-word "sentence") can blow out the raw
        # min/max and compress every other point into an unreadable band.
        # Scale to the 1st-99th percentile instead; outlier points still plot,
        # just clipped at the axis edge rather than distorting the whole view.
        lo, hi = np.nanpercentile(sub[col], [1, 99])
        pad = (hi - lo) * 0.15 or 1.0
        ax.set_ylim(lo - pad, hi + pad)
        n_clipped_total += int(((sub[col] < lo - pad) | (sub[col] > hi + pad)).sum())

        for cat in categories:
            vals = sub.loc[sub["publisher_group"] == cat, col].to_numpy()
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.28, 0.28, size=len(vals))
            ax.scatter(
                x_positions[cat] + jitter,
                vals,
                s=22,
                marker=marker_map[cat],
                facecolor=color_map[cat],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.75,
                zorder=3,
            )
            # Median marker: a short bold tick, not another dot competing for attention.
            ax.hlines(
                np.median(vals), x_positions[cat] - 0.32, x_positions[cat] + 0.32,
                color=INK_PRIMARY, linewidth=1.6, zorder=4,
            )

        ax.set_title(label, fontsize=11, color=INK_PRIMARY, loc="left", pad=6)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=40, ha="right", fontsize=7.5, color=INK_SECONDARY)
        ax.tick_params(axis="y", labelsize=8, colors=INK_SECONDARY)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRIDLINE)
        ax.set_xlim(-0.6, len(categories) - 0.4)

    for ax in axes[len(METRICS):]:
        ax.axis("off")

    fig.suptitle(
        "Linguistic complexity indices by publisher",
        fontsize=16, color=INK_PRIMARY, x=0.02, ha="left", y=0.998,
    )
    outlier_note = (
        f" Axes clipped to the 1st-99th percentile per index ({n_clipped_total} "
        f"outlier point(s) fall outside and are clipped at the edge)."
        if n_clipped_total else ""
    )
    subtitle = (
        f"Each point is one article (n={len(df)}); black tick = per-publisher median. "
        f"Top {TOP_N_PUBLISHERS} publishers by count shown, rest grouped as \"Other\"."
        f"\n{outlier_note.strip()}"
    )
    fig.text(0.02, 0.975, subtitle, fontsize=9.5, color=INK_SECONDARY, ha="left", va="top")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTPUT_PNG, dpi=200, facecolor=SURFACE)
    print(f"saved {OUTPUT_PNG}")


def main():
    if not SCORES_CSV.exists():
        raise SystemExit(
            f"{SCORES_CSV} not found -- run `python scripts/run_complexity.py` first."
        )
    df = load_data()
    df, top_publishers = bucket_publishers(df)
    plot(df, top_publishers)


if __name__ == "__main__":
    main()
