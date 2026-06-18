#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_test_samples.py

Create small Streamlit test-sample CSVs from the annotation-ready dialog
exports produced by 01_preprocess/preprocess.py:
- data/data_clean/dialogs_for_annotation_en.csv
- data/data_clean/dialogs_for_annotation_de.csv

Design:
- Run interactively: the script first prints how many eligible dialogs are
  available per subject/language/condition. German is then handled fully
  (prompt, sample, write) before English is handled fully, as two separate
  steps. The two totals do not have to match.
- Within each language, dialogs are split evenly across the 3 subjects
  (Breakfast, Watches, Vacation). Dialogs with an unrecognized/"Unknown" topic
  are excluded.
- Within each subject-language cell, Condition A (Willi) and Condition B (WV-34)
  are always sampled in equal numbers.

Output (always written to this folder, with the chosen total in the name):
- 05_prolific/streamlit_test_sample_de_<total>.csv
- 05_prolific/streamlit_test_sample_en_<total>.csv

Visible participant-facing columns:
- language
- subject
- dialog_text

Hidden metadata columns:
- All columns beginning with META_
"""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

# =============================================================================
# EDITABLE CONFIG
# =============================================================================

# Random seed for reproducible sampling.
RANDOM_SEED = 42

# Input dialog files (already split by language and annotation-ready).
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CLEAN_DIR = SCRIPT_DIR.parent / "data" / "data_clean"
INPUT_EN_PATH = DATA_CLEAN_DIR / "dialogs_for_annotation_en.csv"
INPUT_DE_PATH = DATA_CLEAN_DIR / "dialogs_for_annotation_de.csv"

# =============================================================================

SUBJECTS = ["Breakfast", "Watches", "Vacation"]
LANGUAGES = ["German", "English"]
CONDITIONS = ["Condition A (Willi)", "Condition B (WV-34)"]

INPUT_PATHS_BY_LANGUAGE = {
    "German": INPUT_DE_PATH,
    "English": INPUT_EN_PATH,
}

LANGUAGE_CODES = {
    "German": "de",
    "English": "en",
}


def output_path_for(language: str, total: int) -> Path:
    """Output CSV path, always written into this folder, including the total."""
    code = LANGUAGE_CODES[language]
    return SCRIPT_DIR / f"streamlit_test_sample_{code}_{total}.csv"


def load_annotation_csv(input_path: Path, language: str) -> List[Dict[str, Any]]:
    """Load one annotation-ready CSV and attach its (fixed) language."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["_language"] = language

    return rows


def prepare_eligible_dialogs(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Counter]:
    """Keep only dialogs with a recognized subject, recognized condition, and text."""
    eligible = []
    exclusion_counts: Counter = Counter()

    for row in rows:
        subject = (row.get("topic_main") or "").strip()
        if subject not in SUBJECTS:
            exclusion_counts["excluded_unrecognized_or_unknown_subject"] += 1
            continue

        condition = (row.get("condition_hidden") or "").strip()
        if condition not in CONDITIONS:
            exclusion_counts["excluded_unrecognized_condition"] += 1
            continue

        dialog_text = (row.get("dialogue_for_annotation") or "").strip()
        if not dialog_text:
            exclusion_counts["excluded_empty_dialog_text"] += 1
            continue

        enriched = dict(row)
        enriched["_subject"] = subject
        enriched["_condition"] = condition
        eligible.append(enriched)

    return eligible, exclusion_counts


def compute_pool_counts(eligible: List[Dict[str, Any]]) -> Counter:
    """Count eligible dialogs per (subject, language, condition)."""
    return Counter(
        (row["_subject"], row["_language"], row["_condition"]) for row in eligible
    )


def print_pool_report(language: str, pool_counts: Counter) -> None:
    """Print available dialog counts per subject/condition for one language."""
    print(f"\nAvailable {language} dialog pool (eligible):")

    language_total = sum(
        pool_counts[(subject, language, condition)]
        for subject in SUBJECTS
        for condition in CONDITIONS
    )
    print(f"{language} total: {language_total}")

    for subject in SUBJECTS:
        stratum_total = sum(
            pool_counts[(subject, language, condition)] for condition in CONDITIONS
        )
        print(f"  {subject:9s}: {stratum_total} available")
        for condition in CONDITIONS:
            available = pool_counts[(subject, language, condition)]
            print(f"    {condition:22s}: {available} available")


def prompt_total_for_language(language: str, pool_counts: Counter) -> int:
    """
    Interactively ask for the number of conversations to generate for one
    language, re-prompting until the value is valid and available.
    """
    while True:
        raw = input(
            f"\nHow many {language} conversations do you want to generate? "
        ).strip()

        try:
            total = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if total <= 0:
            print("Please enter a positive number.")
            continue

        if total % len(SUBJECTS) != 0:
            print(
                f"Total must be evenly divisible by {len(SUBJECTS)} "
                f"(one share per subject: {', '.join(SUBJECTS)})."
            )
            continue

        per_subject = total // len(SUBJECTS)
        if per_subject % 2 != 0:
            print(
                f"Dialogs per subject ({per_subject}) must be even so "
                "Condition A and Condition B can be split equally."
            )
            continue

        shortfalls = []
        for subject in SUBJECTS:
            available = sum(
                pool_counts[(subject, language, condition)] for condition in CONDITIONS
            )
            if available < per_subject:
                shortfalls.append(
                    f"  {subject}: requested {per_subject}, only {available} available"
                )

        if shortfalls:
            print(f"Not enough {language} dialogs available for that total:")
            for line in shortfalls:
                print(line)
            continue

        return total


def sample_one_stratum(
    eligible: List[Dict[str, Any]],
    subject: str,
    language: str,
    per_stratum: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Sample one subject-language stratum.

    Always tries to select an equal number of Condition A and Condition B
    dialogs. If exact condition balance is not possible, it fills from the
    same subject-language stratum and prints a warning.
    """
    pool = [
        d for d in eligible
        if d.get("_subject") == subject and d.get("_language") == language
    ]

    target_per_condition = per_stratum // 2
    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for dialog in pool:
        by_condition[dialog["_condition"]].append(dialog)

    selected: List[Dict[str, Any]] = []

    for condition in CONDITIONS:
        candidates = by_condition.get(condition, [])
        k = min(target_per_condition, len(candidates))
        selected.extend(rng.sample(candidates, k))

        if k < target_per_condition:
            print(
                f"Warning: only found {k}/{target_per_condition} dialogs for "
                f"{subject} / {language} / {condition}. Filling from same stratum.",
                file=sys.stderr,
            )

    selected_ids = {str(d.get("dialog_id")) for d in selected}
    remaining = [d for d in pool if str(d.get("dialog_id")) not in selected_ids]

    if len(selected) < per_stratum:
        need = per_stratum - len(selected)
        selected.extend(rng.sample(remaining, need))

    if len(selected) > per_stratum:
        selected = rng.sample(selected, per_stratum)

    rng.shuffle(selected)
    return selected


def sample_language_batch(
    eligible: List[Dict[str, Any]],
    language: str,
    n_per_subject_language: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Sample the full batch for one language."""
    selected: List[Dict[str, Any]] = []

    for subject in SUBJECTS:
        selected.extend(
            sample_one_stratum(
                eligible=eligible,
                subject=subject,
                language=language,
                per_stratum=n_per_subject_language,
                rng=rng,
            )
        )

    rng.shuffle(selected)
    return selected


def make_row(row: Dict[str, Any], sample_stratum: str) -> Dict[str, Any]:
    """Convert one eligible annotation row into one output CSV row."""
    return {
        # Hidden metadata columns for Prolific / Streamlit.
        "META_dialog_id": row.get("dialog_id", ""),
        "META_condition": row.get("condition_hidden", ""),
        "META_feedback": row.get("feedback_existing", ""),
        "META_original_topics": row.get("topics_json", ""),
        "META_sample_stratum": sample_stratum,
        "META_n_user_turns": row.get("n_visitor_turns", ""),
        "META_n_robot_turns": row.get("n_robot_turns", ""),
        "META_n_visible_turns": row.get("n_turns_visible", ""),

        # Visible annotation columns.
        "language": row["_language"],
        "subject": row["_subject"],
        "dialog_text": row.get("dialogue_for_annotation", ""),
    }


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "META_dialog_id",
        "META_condition",
        "META_feedback",
        "META_original_topics",
        "META_sample_stratum",
        "META_n_user_turns",
        "META_n_robot_turns",
        "META_n_visible_turns",
        "language",
        "subject",
        "dialog_text",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)


def print_language_summary(
    language: str,
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Print a compact summary for the test sample just generated for one language."""
    print(f"\n--- {language} test sample written ---")
    print(f"Output rows: {len(rows)}")
    print(f"Output file: {output_path}")

    print("Counts by subject:")
    by_subject = Counter(row["subject"] for row in rows)
    for subject in SUBJECTS:
        print(f"  {subject:9s}: {by_subject[subject]}")

    print("Counts by subject and condition:")
    by_subject_condition = Counter((row["subject"], row["META_condition"]) for row in rows)
    for subject in SUBJECTS:
        for condition in CONDITIONS:
            print(f"  {subject:9s} / {condition:22s}: {by_subject_condition[(subject, condition)]}")


def process_language(
    language: str,
    eligible: List[Dict[str, Any]],
    pool_counts: Counter,
    rng: random.Random,
) -> None:
    """Prompt, sample, and write the test sample CSV for one language, start to finish."""
    print(f"\n==============================")
    print(f"{language}")
    print(f"==============================")

    print_pool_report(language, pool_counts)
    total = prompt_total_for_language(language, pool_counts)
    n_per_subject_language = total // len(SUBJECTS)

    sampled_dialogs = sample_language_batch(
        eligible=eligible,
        language=language,
        n_per_subject_language=n_per_subject_language,
        rng=rng,
    )

    rows = [
        make_row(dialog, f"{dialog['_subject']}_{dialog['_language']}")
        for dialog in sampled_dialogs
    ]

    output_path = output_path_for(language, total)
    write_csv(rows, output_path)
    print_language_summary(language, rows, output_path)


def main() -> None:
    for language, path in INPUT_PATHS_BY_LANGUAGE.items():
        if not path.exists():
            raise FileNotFoundError(f"Input file not found for {language}: {path}")

    all_rows: List[Dict[str, Any]] = []
    for language, path in INPUT_PATHS_BY_LANGUAGE.items():
        all_rows.extend(load_annotation_csv(path, language))

    eligible, exclusion_counts = prepare_eligible_dialogs(all_rows)
    pool_counts = compute_pool_counts(eligible)

    if exclusion_counts:
        print("\nExcluded dialogs:")
        for key, value in exclusion_counts.items():
            print(f"  {key}: {value}")

    rng = random.Random(RANDOM_SEED)

    # German is fully handled (prompt, sample, write) before English starts.
    for language in LANGUAGES:
        process_language(language, eligible, pool_counts, rng)

    print("\nAll done.")


if __name__ == "__main__":
    main()
