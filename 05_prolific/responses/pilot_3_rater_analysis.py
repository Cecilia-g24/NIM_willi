#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pilot_3_rater_analysis.py

Inter-rater reliability (ICC) analysis for 3-rater / 36-dialog English
training pilot runs. Each subfolder of data/results/pilot_runs/ (e.g.
pilot_1st_0709_6+30/, pilot_2nd_0716_6+30/) holds one pilot run's CSV and
is analyzed independently.

For each rating dimension with data in a given pilot run (dimensions added
in a later pilot round, e.g. overall_conversational_interaction_quality,
are skipped automatically for earlier runs that don't have them), computes
the full pingouin ICC table
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
- Every *.csv found one level under data/results/pilot_runs/<pilot_name>/
  (relative to the repo root)

Output (written into each pilot run's own folder, alongside its CSV):
- icc_results.csv: full ICC table (all types) per dimension
- icc_chart.png: bar chart of ICC(C,1) vs ICC(C,k) per dimension
- Console summary highlighting ICC(C,1) (single rater) and ICC(C,k) (average
  of 3 raters) per dimension

Additionally, runs two before/after comparisons of the earliest pilot run
("before") against the latest one ("after") — e.g. to check whether
agreement improved after the rating manual was introduced. Both are written
to data/results/pilot_runs/before_after/ with the same panel layout and
legend wording (blue = single rater, orange = 3-rater average) as above:
- icc_results_6_training.csv / icc_chart_6_training.png: restricted to the
  6 dialogs common to every pilot run's "training" phase
- icc_results_36.csv / icc_chart_36.png: all 36 dialogs (the full pilot)
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
REPO_ROOT = SCRIPT_DIR.parent.parent
PILOT_RUNS_DIR = REPO_ROOT / "data" / "results" / "pilot_runs"

# Expected number of dialogs rated by each complete rater in this pilot.
EXPECTED_N_DIALOGS = 36

# The 6 "training" dialogs are the same across every pilot run and are used
# for the before/after comparison (e.g. before vs. after the rating manual).
TRAINING_PHASE = "training"
EXPECTED_N_TRAINING_DIALOGS = 6

DIMENSIONS = [
    "user_engagement_enjoyment",
    "user_self_disclosure",
    "user_topical_alignment",
    "user_elaboration_informativeness",
    "user_initiative_active_contribution",
    "user_politeness",
    "user_frustration_dissatisfaction",
    "overall_conversational_interaction_quality",
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
    "overall_conversational_interaction_quality": "Overall\nquality",
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


def compute_all_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run compute_icc_for_dimension for every configured dimension that has
    data in df, skipping (with a console note) any dimension that's entirely
    missing — e.g. a question added in a later pilot round won't exist yet
    in earlier ones.
    """
    available = [dimension for dimension in DIMENSIONS if df[dimension].notna().any()]
    skipped = [dimension for dimension in DIMENSIONS if dimension not in available]
    if skipped:
        print(f"Skipping dimension(s) with no data in this file: {', '.join(skipped)}")

    return pd.concat(
        [compute_icc_for_dimension(df, dimension) for dimension in available],
        ignore_index=True,
    )


def dims_present(all_results: pd.DataFrame) -> list[str]:
    """Return DIMENSIONS filtered to those actually present in all_results, in canonical order."""
    present = set(all_results["dimension"].unique())
    return [dimension for dimension in DIMENSIONS if dimension in present]


def print_summary(all_results: pd.DataFrame, pilot_name: str) -> None:
    """Print a compact per-dimension summary: single-rater ICC(C,1) vs average-of-3 ICC(C,k)."""
    ci_col = "CI95%" if "CI95%" in all_results.columns else "CI95"

    print(f"--- Summary for {pilot_name}: ICC(C,1) single rater vs ICC(C,k) average of 3 raters (two-way fixed, consistency) ---")
    header = f"{'dimension':38s} {'ICC(C,1)':>10s} {'95% CI':>18s} {'ICC(C,k)':>10s} {'95% CI':>18s}"
    print(header)
    print("-" * len(header))

    for dimension in dims_present(all_results):
        subset = all_results[all_results["dimension"] == dimension].set_index("Type")
        icc_single = subset.loc["ICC(C,1)"]
        icc_avg = subset.loc["ICC(C,k)"]
        print(
            f"{dimension:38s} {icc_single['ICC']:10.3f} {str(icc_single[ci_col]):>18s} "
            f"{icc_avg['ICC']:10.3f} {str(icc_avg[ci_col]):>18s}"
        )


def plot_icc_bars(ax: plt.Axes, all_results: pd.DataFrame, title: str) -> None:
    """
    Draw the grouped bar chart onto an existing axes: single-rater ICC(3,1)
    (blue) vs 3-rater-average ICC(3,k) (orange) per dimension (Shrout &
    Fleiss two-way fixed, consistency model; pingouin labels these
    ICC(C,1) / ICC(C,k)), with Cicchetti (1994) interpretation thresholds
    overlaid. Shared by plot_icc_summary and plot_before_after_summary so
    every ICC chart in this script uses the same layout.
    """
    dims = dims_present(all_results)
    single = [
        all_results.loc[
            (all_results["dimension"] == dim) & (all_results["Type"] == "ICC(C,1)"), "ICC"
        ].iloc[0]
        for dim in dims
    ]
    average = [
        all_results.loc[
            (all_results["dimension"] == dim) & (all_results["Type"] == "ICC(C,k)"), "ICC"
        ].iloc[0]
        for dim in dims
    ]

    x = np.arange(len(dims))
    width = 0.35

    # ICC can go negative (worse than "poor"). Extend the lower limit below 0
    # when that happens so those bars stay visible instead of being clipped.
    y_lower = min(0.0, min(single + average) - 0.05)

    # Shaded Cicchetti (1994) zones, separated by dashed lines at the zone
    # boundaries, with the zone name labeled inside each band on the right.
    # The bottom ("Poor") zone is stretched down to y_lower so shading covers
    # the full visible range even when bars dip below 0.
    for lower, upper, zone_label, color in ICC_ZONES:
        zone_lower = y_lower if lower <= 0 else lower
        ax.axhspan(zone_lower, upper, color=color, alpha=0.35, zorder=0)
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

    ax.set_ylim(y_lower, 1.05)
    ax.set_ylabel("ICC score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(loc="upper left")


def plot_icc_summary(all_results: pd.DataFrame, output_path: Path) -> None:
    """Single-panel ICC(3,1) vs ICC(3,k) chart for one pilot run."""
    labels = [DIMENSION_LABELS[dim] for dim in dims_present(all_results)]

    fig, ax = plt.subplots(figsize=(13, 7))
    plot_icc_bars(ax, all_results, "Interrater Reliability by Dimension (ICC(3,1) vs ICC(3,k))")
    ax.set_xticklabels(labels)

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


def analyze_pilot_run(input_path: Path) -> None:
    """Run the full ICC analysis for one pilot-run CSV and write its outputs
    alongside it, inside its pilot-run folder.
    """
    pilot_name = input_path.parent.name
    output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Analyzing {input_path.name} ===")

    df = load_responses(input_path)
    df = select_complete_raters(df, EXPECTED_N_DIALOGS)

    all_results = compute_all_dimensions(df)

    output_path = output_dir / "icc_results.csv"
    all_results.to_csv(output_path, index=False)
    print(f"Full ICC table (all types) written to: {output_path}\n")

    print_summary(all_results, pilot_name)
    plot_icc_summary(all_results, output_dir / "icc_chart.png")


def print_before_after_summary(
    all_results: pd.DataFrame, before_name: str, after_name: str, dialog_desc: str
) -> None:
    """Print a compact per-dimension Before vs After comparison for ICC(C,1) and ICC(C,k)."""
    print(
        f"--- Before/after summary ({dialog_desc}): {before_name} (before) vs {after_name} (after) ---"
    )
    header = (
        f"{'dimension':38s} {'ICC(C,1) before':>16s} {'ICC(C,1) after':>16s} "
        f"{'ICC(C,k) before':>16s} {'ICC(C,k) after':>16s}"
    )
    print(header)
    print("-" * len(header))

    def fmt(period_df: pd.DataFrame, icc_type: str) -> str:
        # A dimension added in a later pilot round (e.g.
        # overall_conversational_interaction_quality) has no "before" data.
        if icc_type not in period_df.index:
            return "n/a"
        return f"{period_df.loc[icc_type, 'ICC']:.3f}"

    for dimension in dims_present(all_results):
        dim_results = all_results[all_results["dimension"] == dimension]
        before = dim_results[dim_results["period"] == "Before"].set_index("Type")
        after = dim_results[dim_results["period"] == "After"].set_index("Type")
        print(
            f"{dimension:38s} "
            f"{fmt(before, 'ICC(C,1)'):>16s} "
            f"{fmt(after, 'ICC(C,1)'):>16s} "
            f"{fmt(before, 'ICC(C,k)'):>16s} "
            f"{fmt(after, 'ICC(C,k)'):>16s}"
        )
    print()


def plot_before_after_summary(
    all_results: pd.DataFrame,
    before_name: str,
    after_name: str,
    dialog_desc: str,
    output_path: Path,
) -> None:
    """
    Two-panel chart comparing Before vs After. Each panel uses the same
    layout and legend wording as the per-pilot-run chart (plot_icc_bars:
    blue = "Single rater — ICC(3,1)", orange = "3-rater average — ICC(3,k)")
    — top panel is the "before" pilot run, bottom panel is "after".
    """
    labels = [DIMENSION_LABELS[dim] for dim in dims_present(all_results)]

    fig, axes = plt.subplots(2, 1, figsize=(13, 13), sharex=True)

    before_results = all_results[all_results["period"] == "Before"]
    after_results = all_results[all_results["period"] == "After"]

    plot_icc_bars(axes[0], before_results, f"Before — {before_name} ({dialog_desc})")
    plot_icc_bars(axes[1], after_results, f"After — {after_name} ({dialog_desc})")
    axes[-1].set_xticklabels(labels)

    fig.suptitle(f"Inter-rater Reliability on {dialog_desc}: Before vs After", fontsize=13)
    fig.text(
        0.02, 0.005,
        f"Based on {dialog_desc}. ICC(3,1) = single-rater reliability, ICC(3,k) = reliability of "
        "the 3-rater average (two-way fixed, consistency model). Interpretation scale (Cicchetti, "
        "1994): ICC < 0.40 = poor; 0.40-0.59 = fair; 0.60-0.74 = good; >= 0.75 = excellent.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Before/after chart written to: {output_path}\n")


def analyze_before_after(
    pilot_csvs: list[Path],
    phase_filter: str | None,
    expected_n_dialogs: int,
    dialog_desc: str,
    results_filename: str,
    chart_filename: str,
) -> None:
    """
    Compare inter-rater reliability between the earliest ("before") and
    latest ("after") pilot run — e.g. to check whether introducing the
    rating manual between pilot_1st and pilot_2nd improved agreement.

    phase_filter restricts to one "phase" value (e.g. the 6 shared training
    dialogs); pass None to use every dialog in the pilot run (e.g. all 36).
    """
    if len(pilot_csvs) < 2:
        print(f"Skipping before/after analysis ({dialog_desc}): need at least 2 pilot runs.\n")
        return

    before_path, after_path = pilot_csvs[0], pilot_csvs[-1]

    period_results = []
    for period_label, path in (("Before", before_path), ("After", after_path)):
        pilot_name = path.parent.name
        print(f"\n=== {dialog_desc} ({period_label}: {pilot_name}) ===")

        df = load_responses(path)
        if phase_filter is not None:
            df = df[df["phase"] == phase_filter].copy()
        df = select_complete_raters(df, expected_n_dialogs)

        results = compute_all_dimensions(df)
        results.insert(0, "period", period_label)
        results.insert(1, "pilot_name", pilot_name)
        period_results.append(results)

    all_results = pd.concat(period_results, ignore_index=True)

    # A dimension added in a later pilot round (e.g.
    # overall_conversational_interaction_quality) has no "before" data — it's
    # still included below (marked "n/a" for the missing period) rather than
    # dropped, so later pilot runs are shown in full.
    before_dims = set(period_results[0]["dimension"])
    after_dims = set(period_results[1]["dimension"])
    only_after = after_dims - before_dims
    if only_after:
        print(f"Dimension(s) only present in the 'after' pilot run (no before data): {', '.join(sorted(only_after))}")

    output_dir = PILOT_RUNS_DIR / "before_after"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / results_filename
    all_results.to_csv(output_path, index=False)
    print(f"\nBefore/after ICC table written to: {output_path}\n")

    before_name, after_name = before_path.parent.name, after_path.parent.name
    print_before_after_summary(all_results, before_name, after_name, dialog_desc)
    plot_before_after_summary(
        all_results, before_name, after_name, dialog_desc, output_dir / chart_filename
    )


def main() -> None:
    # Each pilot-run folder holds one source CSV named after the folder
    # itself (plus icc_results.csv/icc_chart.png once analyzed), so match on
    # that instead of "*/*.csv" to avoid re-ingesting our own output.
    pilot_csvs = sorted(
        pilot_dir / f"{pilot_dir.name}.csv"
        for pilot_dir in PILOT_RUNS_DIR.iterdir()
        if pilot_dir.is_dir() and (pilot_dir / f"{pilot_dir.name}.csv").exists()
    )
    if not pilot_csvs:
        raise FileNotFoundError(f"No CSV files found in {PILOT_RUNS_DIR}/*/")

    for input_path in pilot_csvs:
        analyze_pilot_run(input_path)

    analyze_before_after(
        pilot_csvs,
        phase_filter=TRAINING_PHASE,
        expected_n_dialogs=EXPECTED_N_TRAINING_DIALOGS,
        dialog_desc=f"{EXPECTED_N_TRAINING_DIALOGS} shared training dialogs",
        results_filename="icc_results_6_training.csv",
        chart_filename="icc_chart_6_training.png",
    )
    analyze_before_after(
        pilot_csvs,
        phase_filter=None,
        expected_n_dialogs=EXPECTED_N_DIALOGS,
        dialog_desc=f"all {EXPECTED_N_DIALOGS} dialogs",
        results_filename="icc_results_36.csv",
        chart_filename="icc_chart_36.png",
    )


if __name__ == "__main__":
    main()
