#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_train_set.py

Create the Streamlit "training" sample CSVs (6 fixed + 30 random dialogs per
language) from the annotation-ready dialog exports produced by
01_preprocess/preprocess.py:
- data/data_clean/dialogs_for_annotation_en.csv
- data/data_clean/dialogs_for_annotation_de.csv

Design:
- Per language, the output is always 36 dialogs total:
    - 6 fixed "training representative" dialogs, hard-coded below
      (see TRAIN_REPRESENTATIVES_EN / TRAIN_REPRESENTATIVES_DE). These were
      manually picked, 2 per engagement level (High/Moderate/Low), spread
      across the 3 subjects, each with a short "why selected" rationale.
    - 30 randomly sampled dialogs, chosen the same way as
      create_test_samples.py: split evenly across the 3 subjects
      (Breakfast, Watches, Vacation; 10 each), and within each subject split
      evenly across Condition A (Willi) and Condition B (WV-34) (5 each).
      The 6 fixed dialogs are excluded from this random pool so nothing is
      picked twice.
- Runs non-interactively (no terminal prompts): both languages are always
  generated in one run.
- After writing both CSVs, prints stats per language: total rows, counts by
  fixed vs. random, counts by condition, and counts by subject + condition.

Output (always written to this folder):
- 05_prolific/streamlit_train_sample_de_36.csv
- 05_prolific/streamlit_train_sample_en_36.csv

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

# Number of randomly-sampled dialogs per language (on top of the 6 fixed ones).
N_RANDOM_PER_LANGUAGE = 30

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

# The robot's opening turn always introduces itself by name ("I'm Willi" /
# "I'm WV-34"), which leaks the hidden condition to whoever reads dialog_text.
# Replace that first turn with one neutral, condition-blind greeting so both
# conditions look identical at the start.
NEUTRAL_GREETINGS = {
    "English": (
        "Robot: Welcome to JOSEPHS! What would you like to talk about? "
        "Choose a topic: Watches, Breakfast, or Vacation. By the way, "
        "say stop at any time to end this conversation."
    ),
    "German": (
        "Roboter: Willkommen im JOSEPHS! Worüber möchten Sie sprechen? "
        "Wählen Sie ein Thema: Uhren, Frühstück oder Urlaub. Sie können "
        "jederzeit \"stop\" sagen, um das Gespräch zu beenden."
    ),
}

# -----------------------------------------------------------------------------
# Fixed "training representative" dialogs (6 per language).
#
# Manually picked per language: 2 dialogs per engagement level (High,
# Moderate, Low), spread across different subjects rather than repeating the
# same one, so the 6 examples together illustrate the range of engagement
# levels and the range of subjects. Levels were judged from dialog length,
# depth, and how reactive/sustained the visitor's participation was:
#   - High:     long, multi-turn, detailed answers and follow-up questions.
#   - Moderate: several turns with genuine back-and-forth, but shorter/less
#               sustained than High.
#   - Low:      minimal engagement - a topic pick and at most one short
#               question before disengaging.
# -----------------------------------------------------------------------------

TRAIN_REPRESENTATIVES_EN: List[Dict[str, Any]] = [
    {
        "level": "High",
        "dialog_id": 1349,
        "turns": 17,
        "subject": "Watches",
        "why_selected": "Long, rich, multi-topic exchange with detailed answers and follow-ups",
    },
    {
        "level": "Moderate",
        "dialog_id": 614,
        "turns": 7,
        "subject": "Vacation",
        "why_selected": "Gives trip context and asks for recommendations",
    },
    {
        "level": "Low",
        "dialog_id": 507,
        "turns": 1,
        "subject": "Breakfast",
        "why_selected": "Only selects the topic",
    },
    {
        "level": "High",
        "dialog_id": 732,
        "turns": 11,
        "subject": "Breakfast",
        "why_selected": "Personal, reactive, and sustained interaction",
    },
    {
        "level": "Moderate",
        "dialog_id": 926,
        "turns": 6,
        "subject": "Watches",
        "why_selected": "Gives feedback on voice/experience, then exits",
    },
    {
        "level": "Low",
        "dialog_id": 888,
        "turns": 2,
        "subject": "Vacation",
        "why_selected": "Brief topic request plus one short question",
    },
]

TRAIN_REPRESENTATIVES_DE: List[Dict[str, Any]] = [
    {
        "level": "High",
        "dialog_id": 1347,
        "turns": 23,
        "subject": "Watches",
        "why_selected": "Very detailed watch-focused interaction with strong interest",
    },
    {
        "level": "Moderate",
        "dialog_id": 712,
        "turns": 6,
        "subject": "Breakfast",
        "why_selected": "Coherent breakfast routine with limited back-and-forth",
    },
    {
        "level": "Low",
        "dialog_id": 1157,
        "turns": 1,
        "subject": "Vacation",
        "why_selected": "Single recommendation question, no continuation",
    },
    {
        "level": "High",
        "dialog_id": 144,
        "turns": 26,
        "subject": "Breakfast",
        "why_selected": "Rich personal details plus scientific follow-up questions",
    },
    {
        "level": "Moderate",
        "dialog_id": 1378,
        "turns": 7,
        "subject": "Vacation",
        "why_selected": "Vacation preferences plus a skiing-related follow-up",
    },
    {
        "level": "Low",
        "dialog_id": 244,
        "turns": 2,
        "subject": "Watches",
        "why_selected": "Topic chosen, then conversation quickly ended",
    },
]

TRAIN_REPRESENTATIVES_BY_LANGUAGE = {
    "English": TRAIN_REPRESENTATIVES_EN,
    "German": TRAIN_REPRESENTATIVES_DE,
}


def anonymize_intro(dialog_text: str, language: str) -> str:
    """Replace the robot's opening turn with a neutral, condition-blind greeting."""
    turns = dialog_text.split("\n\n")
    if not turns:
        return dialog_text

    greeting = NEUTRAL_GREETINGS.get(language)
    if greeting is None:
        return dialog_text

    turns[0] = greeting
    return "\n\n".join(turns)


def output_path_for(language: str, total: int) -> Path:
    """Output CSV path, always written into this folder, including the total."""
    code = LANGUAGE_CODES[language]
    return SCRIPT_DIR / f"streamlit_train_sample_{code}_{total}.csv"


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


def pick_fixed_rows(
    eligible: List[Dict[str, Any]], language: str
) -> List[Dict[str, Any]]:
    """Look up the 6 hard-coded training representative dialogs for one language."""
    by_id = {
        str(row.get("dialog_id")): row
        for row in eligible
        if row["_language"] == language
    }

    fixed_rows = []
    for entry in TRAIN_REPRESENTATIVES_BY_LANGUAGE[language]:
        dialog_id = str(entry["dialog_id"])
        row = by_id.get(dialog_id)
        if row is None:
            raise ValueError(
                f"Fixed training dialog_id {dialog_id} not found in eligible "
                f"{language} pool (check {INPUT_PATHS_BY_LANGUAGE[language]})."
            )

        enriched = dict(row)
        enriched["_selection"] = "fixed"
        enriched["_level"] = entry["level"]
        enriched["_why_selected"] = entry["why_selected"]
        fixed_rows.append(enriched)

    return fixed_rows


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
                f"{subject} / {language} / {condition}. Filling from same stratum."
            )

    selected_ids = {str(d.get("dialog_id")) for d in selected}
    remaining = [d for d in pool if str(d.get("dialog_id")) not in selected_ids]

    if len(selected) < per_stratum:
        need = per_stratum - len(selected)
        selected.extend(rng.sample(remaining, min(need, len(remaining))))

    if len(selected) > per_stratum:
        selected = rng.sample(selected, per_stratum)

    rng.shuffle(selected)
    return selected


def sample_random_rows(
    eligible: List[Dict[str, Any]],
    language: str,
    n_random: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Randomly sample n_random dialogs for one language, split evenly across subjects."""
    per_subject = n_random // len(SUBJECTS)
    selected: List[Dict[str, Any]] = []

    for subject in SUBJECTS:
        rows = sample_one_stratum(
            eligible=eligible,
            subject=subject,
            language=language,
            per_stratum=per_subject,
            rng=rng,
        )
        for row in rows:
            row["_selection"] = "random"
            row["_level"] = ""
            row["_why_selected"] = ""
        selected.extend(rows)

    rng.shuffle(selected)
    return selected


def make_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one eligible annotation row into one output CSV row."""
    return {
        # Hidden metadata columns for Prolific / Streamlit.
        "META_dialog_id": row.get("dialog_id", ""),
        "META_condition": row.get("condition_hidden", ""),
        "META_feedback": row.get("feedback_existing", ""),
        "META_original_topics": row.get("topics_json", ""),
        "META_selection": row["_selection"],
        "META_level": row["_level"],
        "META_why_selected": row["_why_selected"],
        "META_n_user_turns": row.get("n_visitor_turns", ""),
        "META_n_robot_turns": row.get("n_robot_turns", ""),
        "META_n_visible_turns": row.get("n_turns_visible", ""),

        # Visible annotation columns.
        "language": row["_language"],
        "subject": row["_subject"],
        "dialog_text": anonymize_intro(
            row.get("dialogue_for_annotation", ""), row["_language"]
        ),
    }


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "META_dialog_id",
        "META_condition",
        "META_feedback",
        "META_original_topics",
        "META_selection",
        "META_level",
        "META_why_selected",
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
    """Print a compact summary for the train sample just generated for one language."""
    print(f"\n--- {language} train sample written ---")
    print(f"Output rows: {len(rows)}")
    print(f"Output file: {output_path}")

    by_selection = Counter(row["META_selection"] for row in rows)
    print(f"Counts by selection: fixed={by_selection['fixed']}, random={by_selection['random']}")

    print("Fixed dialogs (level, dialog_id, subject, condition):")
    for row in rows:
        if row["META_selection"] != "fixed":
            continue
        print(
            f"  {row['META_level']:8s} {row['META_dialog_id']:>6} "
            f"{row['subject']:9s} {row['META_condition']}"
        )

    print("Counts by condition:")
    by_condition = Counter(row["META_condition"] for row in rows)
    for condition in CONDITIONS:
        print(f"  {condition:22s}: {by_condition[condition]}")

    print("Counts by subject and condition:")
    by_subject_condition = Counter((row["subject"], row["META_condition"]) for row in rows)
    for subject in SUBJECTS:
        for condition in CONDITIONS:
            print(f"  {subject:9s} / {condition:22s}: {by_subject_condition[(subject, condition)]}")


def process_language(
    language: str,
    eligible: List[Dict[str, Any]],
    rng: random.Random,
) -> None:
    """Pick the 6 fixed + N random dialogs and write the train sample CSV for one language."""
    print(f"\n==============================")
    print(f"{language}")
    print(f"==============================")

    fixed_rows = pick_fixed_rows(eligible, language)
    fixed_ids = {str(row.get("dialog_id")) for row in fixed_rows}

    remaining_pool = [
        row for row in eligible
        if row["_language"] == language and str(row.get("dialog_id")) not in fixed_ids
    ]

    random_rows = sample_random_rows(
        eligible=remaining_pool,
        language=language,
        n_random=N_RANDOM_PER_LANGUAGE,
        rng=rng,
    )

    all_rows = [make_row(row) for row in fixed_rows + random_rows]

    total = len(all_rows)
    output_path = output_path_for(language, total)
    write_csv(all_rows, output_path)
    print_language_summary(language, all_rows, output_path)


def main() -> None:
    for language, path in INPUT_PATHS_BY_LANGUAGE.items():
        if not path.exists():
            raise FileNotFoundError(f"Input file not found for {language}: {path}")

    all_rows: List[Dict[str, Any]] = []
    for language, path in INPUT_PATHS_BY_LANGUAGE.items():
        all_rows.extend(load_annotation_csv(path, language))

    eligible, exclusion_counts = prepare_eligible_dialogs(all_rows)

    if exclusion_counts:
        print("\nExcluded dialogs:")
        for key, value in exclusion_counts.items():
            print(f"  {key}: {value}")

    rng = random.Random(RANDOM_SEED)

    for language in LANGUAGES:
        process_language(language, eligible, rng)

    print("\nAll done.")


if __name__ == "__main__":
    main()
