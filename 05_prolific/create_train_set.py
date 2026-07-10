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
    - 30 randomly sampled dialogs, drawn uniformly from all eligible dialogs
      of that language. There is no stratification by subject or condition:
      any dialog that has not yet been rated can be chosen. The 6 fixed
      dialogs are excluded from this random pool so nothing is picked twice.
      Per language, further ids can be excluded from the random pool via
      EXCLUDED_RANDOM_IDS_BY_LANGUAGE (used to keep the second pilot's
      English sample disjoint from the first pilot's 36 dialogs).
- Runs non-interactively (no terminal prompts). The languages generated in one
  run are set via GENERATE_LANGUAGES (currently English only: the German fixed
  dialog 1347 is missing from the anonymized DE export).
- The condition is no longer present in the anonymized annotation CSVs; it is
  joined back in from data/data_clean/metadata_only_for_later_analysis.csv.
- After writing both CSVs, prints stats per language: total rows, counts by
  fixed vs. random, counts by condition, and counts by subject + condition.

Output (always written to this folder, with the optional OUTPUT_TAG suffix):
- 05_prolific/streamlit_train_sample_en_36_pilot2.csv

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
import warnings
from collections import Counter
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

# The public annotation CSVs are fully anonymized and no longer carry the
# condition. It is joined back in from this private metadata file (needed for
# condition-balanced sampling and the hidden META_condition output column).
METADATA_PATH = DATA_CLEAN_DIR / "metadata_only_for_later_analysis.csv"

# Languages to generate in this run. The second pilot is English-only; the
# German fixed dialog 1347 is currently missing from the anonymized DE export,
# so German cannot be generated until its fixed representatives are re-checked.
GENERATE_LANGUAGES = ["English"]

# Number of randomly-sampled dialogs per language (on top of the 6 fixed ones).
N_RANDOM_PER_LANGUAGE = 30

# Optional tag appended to the output filename, e.g. "pilot2" gives
# streamlit_train_sample_en_36_pilot2.csv. Set to "" for no tag.
OUTPUT_TAG = "pilot2"

# =============================================================================

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

# Robot identity anonymization (neutral greeting + scrubbing of "Willi" /
# "WV-34" mentions in all turns) happens upstream in
# 01_preprocess/preprocess.py, so dialogue_for_annotation is already
# condition-blind when it arrives here.

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

# -----------------------------------------------------------------------------
# Record of the 36 English dialogs used in the first pilot test (2026-07-09,
# 3 raters). Extracted from responses/survey_responses_en_train_pilot.csv.
# The first 6 are the fixed training dialogs (in presentation order); the
# remaining 30 are the randomly sampled "main" dialogs (sorted by id).
# Note: the requested name "0709_1st_pilot_dialogs" is not a valid Python
# identifier (cannot start with a digit), hence this spelling.
# -----------------------------------------------------------------------------

PILOT_0709_1ST_DIALOGS: List[int] = [
    # Fixed training dialogs (presentation order):
    1349, 614, 507, 732, 926, 888,
    # Random "main" dialogs:
    392, 434, 460, 498, 518, 519, 532, 537, 605, 666,
    731, 751, 819, 833, 864, 913, 934, 935, 999, 1000,
    1030, 1102, 1107, 1124, 1197, 1255, 1267, 1289, 1325, 1382,
]

# Dialogs to keep out of the random pool per language. For the second pilot,
# the English random sample must avoid everything already used in the first
# pilot (the 6 fixed dialogs are excluded from the random pool anyway, but the
# 30 first-pilot random dialogs must not be drawn again). The 6 fixed training
# dialogs themselves stay the same in every pilot.
EXCLUDED_RANDOM_IDS_BY_LANGUAGE: Dict[str, set] = {
    "English": {str(dialog_id) for dialog_id in PILOT_0709_1ST_DIALOGS},
    "German": set(),
}

# -----------------------------------------------------------------------------
# Record of the 30 randomly-sampled dialogs chosen for the second pilot.
# Populated automatically by process_language() when this script runs (keyed
# by language), so it always reflects the dialogs actually written to the
# most recent output CSV. main() prints a copy-pasteable block for this at
# the end of the run; once the pilot is finalized, copy that block here (or
# into EXCLUDED_RANDOM_IDS_BY_LANGUAGE for the next pilot), mirroring how
# PILOT_0709_1ST_DIALOGS was recorded for the first pilot.
# -----------------------------------------------------------------------------

pilot_2nd_dialogs: Dict[str, List[int]] = {}


def output_path_for(language: str, total: int) -> Path:
    """Output CSV path, always written into this folder, including the total."""
    code = LANGUAGE_CODES[language]
    tag = f"_{OUTPUT_TAG}" if OUTPUT_TAG else ""
    return SCRIPT_DIR / f"streamlit_train_sample_{code}_{total}{tag}.csv"


def load_annotation_csv(input_path: Path, language: str) -> List[Dict[str, Any]]:
    """Load one annotation-ready CSV and attach its (fixed) language."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["_language"] = language

    return rows


def load_condition_map(metadata_path: Path) -> Dict[tuple[str, str], str]:
    """
    Load the private metadata CSV and return a map
    (language_code, dialog_id) -> condition_hidden.

    The metadata file uses language codes ("en"/"de"), matching LANGUAGE_CODES.
    """
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    return {
        (
            (row.get("language") or "").strip(),
            str(row.get("dialog_id") or "").strip(),
        ): (row.get("condition_hidden") or "").strip()
        for row in rows
    }


def prepare_eligible_dialogs(
    rows: List[Dict[str, Any]],
    condition_map: Dict[tuple[str, str], str],
) -> tuple[List[Dict[str, Any]], Counter]:
    """Keep only dialogs with a recognized condition and non-empty text.

    There is no restriction on the subject/topic: the dialog's topic_main is
    passed through as-is into the visible "subject" column.
    """
    eligible = []
    exclusion_counts: Counter = Counter()

    for row in rows:
        subject = (row.get("topic_main") or "").strip()

        # The anonymized annotation CSVs carry no condition column; look the
        # condition up in the private metadata file instead.
        language_code = LANGUAGE_CODES[row["_language"]]
        dialog_id = str(row.get("dialog_id") or "").strip()
        condition = condition_map.get((language_code, dialog_id), "")
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


def sample_random_rows(
    eligible: List[Dict[str, Any]],
    language: str,
    n_random: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Randomly sample n_random dialogs for one language.

    No stratification: any eligible dialog in the pool (i.e. not fixed and not
    already rated in an earlier pilot, see process_language) can be chosen,
    regardless of subject or condition.
    """
    pool = [d for d in eligible if d.get("_language") == language]

    if len(pool) < n_random:
        raise ValueError(
            f"Only {len(pool)} eligible {language} dialogs available, "
            f"but {n_random} random dialogs were requested."
        )

    selected = rng.sample(pool, n_random)
    for row in selected:
        row["_selection"] = "random"
        row["_level"] = ""
        row["_why_selected"] = ""

    rng.shuffle(selected)
    return selected


def make_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one eligible annotation row into one output CSV row."""
    return {
        # Hidden metadata columns for Prolific / Streamlit.
        "META_dialog_id": row.get("dialog_id", ""),
        # Condition joined from the private metadata file (see
        # prepare_eligible_dialogs); the annotation CSV itself is condition-blind.
        "META_condition": row["_condition"],
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
        # Already anonymized (condition-blind) by preprocess.py.
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
    for (subject, condition), count in sorted(by_subject_condition.items()):
        print(f"  {subject:9s} / {condition:22s}: {count}")


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
    excluded_ids = fixed_ids | EXCLUDED_RANDOM_IDS_BY_LANGUAGE.get(language, set())

    remaining_pool = [
        row for row in eligible
        if row["_language"] == language and str(row.get("dialog_id")) not in excluded_ids
    ]
    pool_size_before_sampling = len(remaining_pool)

    random_rows = sample_random_rows(
        eligible=remaining_pool,
        language=language,
        n_random=N_RANDOM_PER_LANGUAGE,
        rng=rng,
    )

    # Record the dialogs actually chosen this run, so they can be copied into
    # a permanent record (see PILOT_0709_1ST_DIALOGS) and excluded from future
    # pilots' random pools.
    pilot_2nd_dialogs[language] = sorted(int(row["dialog_id"]) for row in random_rows)

    dialogs_left = pool_size_before_sampling - len(random_rows)
    print(
        f"\n{language}: {dialogs_left} eligible dialogs left and available "
        f"for future pilots (after this pilot's {len(random_rows)} picks)."
    )
    if dialogs_left < N_RANDOM_PER_LANGUAGE:
        warnings.warn(
            f"Only {dialogs_left} eligible {language} dialogs remain after this "
            f"pilot's picks - fewer than the {N_RANDOM_PER_LANGUAGE} needed for "
            "another pilot of the same size.",
            stacklevel=2,
        )

    all_rows = [make_row(row) for row in fixed_rows + random_rows]

    total = len(all_rows)
    output_path = output_path_for(language, total)
    write_csv(all_rows, output_path)
    print_language_summary(language, all_rows, output_path)


def main() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    for language in GENERATE_LANGUAGES:
        path = INPUT_PATHS_BY_LANGUAGE[language]
        if not path.exists():
            raise FileNotFoundError(f"Input file not found for {language}: {path}")

    condition_map = load_condition_map(METADATA_PATH)

    all_rows: List[Dict[str, Any]] = []
    for language in GENERATE_LANGUAGES:
        all_rows.extend(load_annotation_csv(INPUT_PATHS_BY_LANGUAGE[language], language))

    eligible, exclusion_counts = prepare_eligible_dialogs(all_rows, condition_map)

    if exclusion_counts:
        print("\nExcluded dialogs:")
        for key, value in exclusion_counts.items():
            print(f"  {key}: {value}")

    rng = random.Random(RANDOM_SEED)

    for language in GENERATE_LANGUAGES:
        process_language(language, eligible, rng)

    print("\npilot_2nd_dialogs (copy into a permanent record, e.g. next to "
          "PILOT_0709_1ST_DIALOGS):")
    for language, dialog_ids in pilot_2nd_dialogs.items():
        print(f"  {language}: {dialog_ids}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
