#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pilot_3_rater_analysis.py

Inter-rater reliability (ICC) analysis for the 3-rater / 36-dialog English
training pilot (survey_responses_en_train_pilot.csv).

For each of the 7 rating dimensions, computes the full pingouin ICC table
(ICC(1,1), ICC(A,1), ICC(C,1), ICC(1,k), ICC(A,k), ICC(C,k)) across the 3
raters and 36 dialogs:
- Single-rater reliability ("icc per rater"): ICC(1,1) / ICC(A,1) / ICC(C,1)
- Reliability of the mean of all 3 raters ("icc of average of 3 raters"):
  ICC(1,k) / ICC(A,k) / ICC(C,k)

ICC(A,*) (two-way random, absolute agreement) and ICC(C,*) (two-way fixed,
consistency) are the two most commonly reported variants for this kind of
fixed-rater-panel design; ICC(1,*) (one-way random) is included for
completeness. See Koo & Li (2016) for guidance on choosing among them.

Also renders a grouped bar chart (single rater vs. 3-rater average, per
dimension) with Cicchetti (1994) interpretation thresholds overlaid:
< 0.40 poor, 0.40-0.59 fair, 0.60-0.74 good, >= 0.75 excellent.

Requires: pandas, pingouin, matplotlib, numpy (pip install pingouin matplotlib)

Input:
- survey_responses_en_train_pilot.csv (in this folder)

Output (written to this folder):
- icc_results_en_train_pilot.csv: full ICC table (all types) per dimension
- icc_chart_en_train_pilot.png: bar chart of ICC(C,1) vs ICC(C,k) per dimension
- Console summary highlighting ICC(C,1) (single rater) and ICC(C,k) (average
  of 3 raters) per dimension
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg

# =============================================================================
# EDITABLE CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "survey_responses_en_train_pilot.csv"
OUTPUT_PATH = SCRIPT_DIR / "icc_results_en_train_pilot.csv"
CHART_OUTPUT_PATH = SCRIPT_DIR / "icc_chart_en_train_pilot.png"

# Expected number of dialogs rated by each complete rater in this pilot.
EXPECTED_N_DIALOGS = 36

DIMENSIONS = [
    "user_engagement_enjoyment",
    "user_self_disclosure",
    "user_topical_alignment",
    "user_elaboration_informativeness",
    "user_initiative_active_contribution",
    "user_politeness",
    "user_frustration_dissatisfaction",
]

# Short x-axis labels for the chart.
DIMENSION_LABELS = {
    "user_engagement_enjoyment": "Engagement\n/ enjoyment",
    "user_self_disclosure": "Self-\ndisclosure",
    "user_topical_alignment": "Topical\nalignment",
    "user_elaboration_informativeness": "Elaboration /\ninformativeness",
    "user_initiative_active_contribution": "Initiative /\ncontribution",
    "user_politeness": "Politeness",
    "user_frustration_dissatisfaction": "Frustration /\ndissatisfaction",
}

# Cicchetti (1994) ICC interpretation zones: (lower bound, upper bound, label, fill color).
ICC_ZONES = [
    (0.00, 0.40, "Poor", "#f4b6b6"),
    (0.40, 0.60, "Fair", "#f7dfa5"),
    (0.60, 0.75, "Good", "#c9e6b0"),
    (0.75, 1.05, "Excellent", "#9fd6a3"),
]

# =============================================================================


def load_responses(input_path: Path) -> pd.DataFrame:
    """Load the raw survey responses CSV."""
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df["participant_id"] = df["participant_id"].astype(str)
    return df


def select_complete_raters(df: pd.DataFrame, expected_n_dialogs: int) -> pd.DataFrame:
    """
    Keep only raters (participant_id) with exactly one rating for each of the
    expected dialogs, dropping incomplete/stray participants (e.g. test
    submissions). Prints a warning for anything excluded.
    """
    # If a rater rated the same dialog more than once, keep their latest
    # submission.
    df = df.sort_values("submitted_at_utc")
    df = df.drop_duplicates(subset=["participant_id", "dialog_id"], keep="last")

    counts = df.groupby("participant_id")["dialog_id"].nunique()
    complete_raters = counts[counts == expected_n_dialogs].index.tolist()
    incomplete_raters = counts[counts != expected_n_dialogs]

    if not incomplete_raters.empty:
        print("Excluding incomplete/stray participant(s):")
        for participant_id, n in incomplete_raters.items():
            print(f"  {participant_id}: {n} dialog(s) rated (expected {expected_n_dialogs})")

    if len(complete_raters) != 3:
        raise ValueError(
            f"Expected exactly 3 complete raters, found {len(complete_raters)}: "
            f"{complete_raters}"
        )

    print(f"Using raters: {complete_raters}\n")
    return df[df["participant_id"].isin(complete_raters)].copy()


def compute_icc_for_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Run pingouin's intraclass_corr for one rating dimension."""
    long_df = df[["dialog_id", "participant_id", dimension]].dropna()

    icc = pg.intraclass_corr(
        data=long_df,
        targets="dialog_id",
        raters="participant_id",
        ratings=dimension,
    )
    icc.insert(0, "dimension", dimension)
    return icc


def print_summary(all_results: pd.DataFrame) -> None:
    """Print a compact per-dimension summary: single-rater ICC(C,1) vs average-of-3 ICC(C,k)."""
    ci_col = "CI95%" if "CI95%" in all_results.columns else "CI95"

    print("--- Summary: ICC(C,1) single rater vs ICC(C,k) average of 3 raters (two-way fixed, consistency) ---")
    header = f"{'dimension':38s} {'ICC(C,1)':>10s} {'95% CI':>18s} {'ICC(C,k)':>10s} {'95% CI':>18s}"
    print(header)
    print("-" * len(header))

    for dimension in DIMENSIONS:
        subset = all_results[all_results["dimension"] == dimension].set_index("Type")
        icc_single = subset.loc["ICC(C,1)"]
        icc_avg = subset.loc["ICC(C,k)"]
        print(
            f"{dimension:38s} {icc_single['ICC']:10.3f} {str(icc_single[ci_col]):>18s} "
            f"{icc_avg['ICC']:10.3f} {str(icc_avg[ci_col]):>18s}"
        )


def plot_icc_summary(all_results: pd.DataFrame, output_path: Path) -> None:
    """
    Grouped bar chart: single-rater ICC(3,1) vs 3-rater-average ICC(3,k) per
    dimension (Shrout & Fleiss two-way fixed, consistency model; pingouin
    labels these ICC(C,1) / ICC(C,k)), with Cicchetti (1994) interpretation
    thresholds overlaid.
    """
    single = [
        all_results.loc[
            (all_results["dimension"] == dim) & (all_results["Type"] == "ICC(C,1)"), "ICC"
        ].iloc[0]
        for dim in DIMENSIONS
    ]
    average = [
        all_results.loc[
            (all_results["dimension"] == dim) & (all_results["Type"] == "ICC(C,k)"), "ICC"
        ].iloc[0]
        for dim in DIMENSIONS
    ]
    labels = [DIMENSION_LABELS[dim] for dim in DIMENSIONS]

    x = np.arange(len(DIMENSIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 7))

    # Shaded Cicchetti (1994) zones, separated by dashed lines at the zone
    # boundaries, with the zone name labeled inside each band on the right.
    for lower, upper, zone_label, color in ICC_ZONES:
        ax.axhspan(lower, upper, color=color, alpha=0.35, zorder=0)
        ax.text(
            0.995, (lower + upper) / 2, zone_label,
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=10, fontweight="bold", color="dimgray", zorder=1,
        )
    for boundary in (0.40, 0.60, 0.75):
        ax.axhline(boundary, color="dimgray", linestyle="--", linewidth=1, zorder=1)

    bars_single = ax.bar(
        x - width / 2, single, width, label="Single rater — ICC(3,1)", color="tab:blue", zorder=2
    )
    bars_average = ax.bar(
        x + width / 2, average, width, label="3-rater average — ICC(3,k)", color="tab:orange", zorder=2
    )
    ax.bar_label(bars_single, fmt="%.2f", padding=3)
    ax.bar_label(bars_average, fmt="%.2f", padding=3)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ICC score")
    ax.set_title("Interrater Reliability by Dimension (ICC(3,1) vs ICC(3,k))")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left")

    fig.text(
        0.02, 0.01,
        "ICC(3,1) = single-rater reliability, ICC(3,k) = reliability of the 3-rater average "
        "(two-way fixed, consistency model). Interpretation scale (Cicchetti, 1994): "
        "ICC < 0.40 = poor; 0.40-0.59 = fair; 0.60-0.74 = good; >= 0.75 = excellent.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Chart written to: {output_path}\n")


def main() -> None:
    df = load_responses(INPUT_PATH)
    df = select_complete_raters(df, EXPECTED_N_DIALOGS)

    all_results = pd.concat(
        [compute_icc_for_dimension(df, dimension) for dimension in DIMENSIONS],
        ignore_index=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(OUTPUT_PATH, index=False)
    print(f"Full ICC table (all types) written to: {OUTPUT_PATH}\n")

    print_summary(all_results)
    plot_icc_summary(all_results, CHART_OUTPUT_PATH)


if __name__ == "__main__":
    main()
