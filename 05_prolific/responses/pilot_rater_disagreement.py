#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pilot_rater_disagreement.py

Finds the dialogs and dimensions where the 3 raters disagree the most, for
one pilot run (data/results/pilot_runs/<pilot_name>/<pilot_name>.csv).

For each (dialog_id, dimension) pair, disagreement is measured as:
- range = max(rating) - min(rating) across the 3 raters
- std   = sample standard deviation across the 3 raters

Output (written into the pilot run's own folder, alongside its CSV):
- disagreement_by_dialog_dimension.csv: every (dialog_id, dimension) pair,
  with each rater's rating, range, and std, sorted by range descending.
- disagreement_by_dimension.csv: per-dimension mean/max range and std
  across all dialogs, sorted by mean range descending.
- most_disagreed_dialogs.csv: the top 10 dialogs by average disagreement
  across all dimensions, each with its full transcript (joined from
  data/data_clean/dialogs_full.json by dialog_id) and every rater's rating
  on every dimension, for close reading.

Also prints to console:
- Top N single (dialog_id, dimension) pairs with the largest range.
- Top N dialogs by average range across all dimensions (i.e. dialogs where
  raters disagree the most overall, not just on one dimension).
- Per-dimension summary ranked by how much raters disagree on it.

Requires: pandas (same "willi" conda env as pilot_3_rater_analysis.py)

Usage:
    python pilot_rater_disagreement.py [pilot_name]

    pilot_name defaults to the most recently modified pilot run folder under
    data/results/pilot_runs/ (excluding before_after/).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_3_rater_analysis import (  # noqa: E402
    DIMENSIONS,
    EXPECTED_N_DIALOGS,
    PILOT_RUNS_DIR,
    REPO_ROOT,
    load_responses,
    select_complete_raters,
)

# Number of top rows to print to console for each ranking, and number of
# dialogs to export full transcripts for in most_disagreed_dialogs.csv.
TOP_N = 10

DIALOGS_JSON_PATH = REPO_ROOT / "data" / "data_clean" / "dialogs_full.json"


def latest_pilot_dir() -> Path:
    candidates = [
        pilot_dir
        for pilot_dir in PILOT_RUNS_DIR.iterdir()
        if pilot_dir.is_dir()
        and pilot_dir.name != "before_after"
        and (pilot_dir / f"{pilot_dir.name}.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No pilot run CSVs found in {PILOT_RUNS_DIR}/*/")
    return max(candidates, key=lambda p: (p / f"{p.name}.csv").stat().st_mtime)


def compute_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (dialog_id, dimension): each rater's rating plus range and
    std across the 3 raters.
    """
    available_dims = [dim for dim in DIMENSIONS if df[dim].notna().any()]
    rows = []

    for dim in available_dims:
        pivot = df.pivot(index="dialog_id", columns="participant_id", values=dim)
        pivot = pivot.dropna()  # keep only dialogs all 3 raters rated on this dimension
        for dialog_id, ratings in pivot.iterrows():
            rows.append(
                {
                    "dialog_id": dialog_id,
                    "dimension": dim,
                    **{f"rater_{rid}": val for rid, val in ratings.items()},
                    "range": ratings.max() - ratings.min(),
                    "std": ratings.std(ddof=1),
                }
            )

    return pd.DataFrame(rows).sort_values("range", ascending=False, ignore_index=True)

def summarize_by_dimension(detail: pd.DataFrame) -> pd.DataFrame:
    """Per-dimension mean/max range and mean std, ranked by mean range descending."""
    summary = (
        detail.groupby("dimension")
        .agg(
            mean_range=("range", "mean"),
            max_range=("range", "max"),
            mean_std=("std", "mean"),
            n_dialogs=("dialog_id", "nunique"),
        )
        .sort_values("mean_range", ascending=False)
    )
    return summary


def summarize_by_dialog(detail: pd.DataFrame) -> pd.DataFrame:
    """Per-dialog mean/max range across all dimensions, ranked by mean range descending."""
    summary = (
        detail.groupby("dialog_id")
        .agg(
            mean_range=("range", "mean"),
            max_range=("range", "max"),
            worst_dimension=("range", lambda s: detail.loc[s.idxmax(), "dimension"]),
            n_dimensions=("dimension", "nunique"),
        )
        .sort_values("mean_range", ascending=False)
    )
    return summary


def load_dialog_texts(dialog_ids: set[int]) -> dict[int, dict]:
    """Load full dialog records (messages, topic, condition, ...) for the
    given dialog_ids from dialogs_full.json, keyed by dialog id."""
    with open(DIALOGS_JSON_PATH, encoding="utf-8") as f:
        all_dialogs = json.load(f)
    return {d["id"]: d for d in all_dialogs if d["id"] in dialog_ids}


def format_transcript(messages: list[dict]) -> str:
    """Render a dialog's visitor/robot turns as plain text, one blank line
    apart. The system prompt (persona/instructions) is excluded."""
    lines = []
    for msg in messages:
        if msg["role"] not in ("user", "assistant"):
            continue
        speaker = "Visitor" if msg["role"] == "user" else "Robot"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n\n".join(lines)


def export_most_disagreed_dialogs(
    detail: pd.DataFrame,
    by_dialog: pd.DataFrame,
    df: pd.DataFrame,
    output_path: Path,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """
    Write the top_n most-disagreed-upon dialogs to output_path, each with its
    full transcript and every rater's rating on every dimension, so they can
    be read closely to understand *why* raters disagreed.
    """
    top_dialog_ids = by_dialog.head(top_n).index.tolist()
    dialogs = load_dialog_texts(set(top_dialog_ids))
    # subject/condition/language are repeated across the 3 raters' rows for
    # a given dialog_id, so any one row's values will do.
    meta = df.drop_duplicates("dialog_id").set_index("dialog_id")
    rater_cols = [c for c in detail.columns if c.startswith("rater_")]

    rows = []
    for rank, dialog_id in enumerate(top_dialog_ids, start=1):
        summary = by_dialog.loc[dialog_id]
        dialog_detail = detail[detail["dialog_id"] == dialog_id].set_index("dimension")

        per_dimension_ratings = {}
        for dim in dialog_detail.index:
            for rater_col in rater_cols:
                per_dimension_ratings[f"{dim}__{rater_col}"] = dialog_detail.loc[dim, rater_col]
            per_dimension_ratings[f"{dim}__range"] = dialog_detail.loc[dim, "range"]

        dialog = dialogs.get(dialog_id)
        rows.append(
            {
                "rank": rank,
                "dialog_id": dialog_id,
                "mean_range_all_dims": summary["mean_range"],
                "max_range": summary["max_range"],
                "worst_dimension": summary["worst_dimension"],
                "subject": meta.loc[dialog_id, "subject"] if dialog_id in meta.index else None,
                "condition": meta.loc[dialog_id, "condition"] if dialog_id in meta.index else None,
                "language": meta.loc[dialog_id, "language"] if dialog_id in meta.index else None,
                **per_dimension_ratings,
                "transcript": format_transcript(dialog["messages"])
                if dialog is not None
                else "MISSING FROM dialogs_full.json",
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    pilot_name = sys.argv[1] if len(sys.argv) > 1 else None
    pilot_dir = PILOT_RUNS_DIR / pilot_name if pilot_name else latest_pilot_dir()
    input_path = pilot_dir / f"{pilot_dir.name}.csv"

    print(f"=== Rater disagreement analysis: {input_path.name} ===\n")

    df = load_responses(input_path)
    df = select_complete_raters(df, EXPECTED_N_DIALOGS)

    detail = compute_disagreement(df)
    detail.to_csv(pilot_dir / "disagreement_by_dialog_dimension.csv", index=False)

    by_dimension = summarize_by_dimension(detail)
    by_dimension.to_csv(pilot_dir / "disagreement_by_dimension.csv")

    by_dialog = summarize_by_dialog(detail)

    most_disagreed_path = pilot_dir / "most_disagreed_dialogs.csv"
    export_most_disagreed_dialogs(detail, by_dialog, df, most_disagreed_path)

    pd.set_option("display.width", 120)

    print(f"--- Top {TOP_N} single (dialog_id, dimension) disagreements (largest range) ---")
    print(detail.head(TOP_N).to_string(index=False))

    print(f"\n--- Top {TOP_N} dialogs by average disagreement across all dimensions ---")
    print(by_dialog.head(TOP_N).to_string(float_format="%.2f"))

    print("\n--- Dimensions ranked by how much the 3 raters disagree on them ---")
    print(by_dimension.to_string(float_format="%.2f"))

    print(
        "\nFull detail written to: "
        f"{pilot_dir / 'disagreement_by_dialog_dimension.csv'}"
    )
    print(f"Per-dimension summary written to: {pilot_dir / 'disagreement_by_dimension.csv'}")
    print(f"Top {TOP_N} disagreed dialogs (with full transcripts) written to: {most_disagreed_path}")


if __name__ == "__main__":
    main()
