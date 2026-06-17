#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_batch.py

Create Prolific AI Task Builder CSV batches from dialogs.json.

Pilot design:
- 60 dialogs total
- 30 German dialogs + 30 English dialogs
- 3 subjects: Breakfast, Watches, Vacation
- For each language: 10 dialogs per subject
- Within each subject-language cell, the script tries to balance conditions:
  5 x Condition A (Willi) and 5 x Condition B (WV-34), if available.

Output:
- 05_prolific/pilot_batch_de.csv by default
- 05_prolific/pilot_batch_en.csv by default

Visible participant-facing columns:
- language
- subject
- dialog_text

Hidden metadata columns:
- All columns beginning with META_
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SUBJECTS = ["Breakfast", "Watches", "Vacation"]
LANGUAGES = ["German", "English"]

CONDITIONS = ["Condition A (Willi)", "Condition B (WV-34)"]

# Canonical mapping from raw topic labels to the three study subjects.
SUBJECT_MAP = {
    "breakfast": "Breakfast",
    "frühstück": "Breakfast",
    "fruehstueck": "Breakfast",
    "watches": "Watches",
    "watch": "Watches",
    "uhren": "Watches",
    "uhr": "Watches",
    "vacation": "Vacation",
    "urlaub": "Vacation",
}

ROLE_LABELS = {
    "assistant": "Robot",
    "user": "User",
}


def normalize_text(value: Any) -> str:
    """Normalize text for matching."""
    return str(value or "").strip().lower()


def clean_content(value: Any) -> str:
    """
    Clean message content for display.

    Important:
    - This does NOT correct speech-recognition errors.
    - This does NOT remove robot persona or robot names.
    - It only normalizes line breaks and excessive empty lines.
    """
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_dialogs(input_path: Path) -> List[Dict[str, Any]]:
    """Load dialogs from a JSON file."""
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either a list of dialogs or a dict wrapper such as {"dialogs": [...]}.
    if isinstance(data, list):
        dialogs = data
    elif isinstance(data, dict) and isinstance(data.get("dialogs"), list):
        dialogs = data["dialogs"]
    else:
        raise ValueError(
            "Input JSON must be a list of dialogs or a dictionary with a 'dialogs' list."
        )

    if not all(isinstance(d, dict) for d in dialogs):
        raise ValueError("Every dialog entry must be a JSON object.")

    return dialogs


def get_recognized_subjects(dialog: Dict[str, Any]) -> List[str]:
    """
    Return recognized canonical subjects from the dialog-level topics
    and message-level topic fields.
    """
    raw_topics: List[Any] = []

    raw_topics.extend(dialog.get("topics") or [])

    for msg in dialog.get("messages") or []:
        topic = msg.get("topic")
        if topic:
            raw_topics.append(topic)

    subjects = []
    for raw_topic in raw_topics:
        mapped = SUBJECT_MAP.get(normalize_text(raw_topic))
        if mapped:
            subjects.append(mapped)

    return subjects


def infer_subject(dialog: Dict[str, Any]) -> Optional[str]:
    """
    Infer the dialog subject.

    For the pilot batch, we only keep dialogs that map clearly to one
    of the three study subjects. If a dialog contains multiple subjects,
    it is excluded because it is less clean for subject-balanced annotation.
    """
    subjects = set(get_recognized_subjects(dialog))

    if len(subjects) == 1:
        return next(iter(subjects))

    return None


def detect_language(dialog: Dict[str, Any]) -> str:
    """
    Detect German vs. English using system hints first, then simple text markers.

    The data often includes system messages such as:
    - "remember to ONLY reply in ENGLISH"
    - "Antworte IMMER NUR auf DEUTSCH"

    These messages are used only for language detection and are removed
    from the displayed transcript.
    """
    all_text = "\n".join(
        str(msg.get("content", "")) for msg in (dialog.get("messages") or [])
    )
    lower = f" {all_text.lower()} "

    if "remember to only reply in english" in lower:
        return "English"
    if "antworte immer nur auf deutsch" in lower:
        return "German"

    # Fallback marker-based scoring. This avoids external dependencies.
    german_markers = [
        "ä", "ö", "ü", "ß",
        " der ", " die ", " das ", " und ", " ich ", " du ", " nicht ",
        " mit ", " über ", " uhren", " frühstück", " urlaub",
        " willkommen ", " kannst ", " möchte ",
    ]
    english_markers = [
        " the ", " and ", " i ", " you ", " is ", " are ", " do ",
        " breakfast", " watches", " vacation", " welcome ", " please ",
        " would ", " could ",
    ]

    german_score = sum(lower.count(marker) for marker in german_markers)
    english_score = sum(lower.count(marker) for marker in english_markers)

    return "German" if german_score > english_score else "English"


def build_dialog_text(dialog: Dict[str, Any]) -> str:
    """
    Build participant-facing dialog text.

    Removes system messages, because they are internal instructions.
    Keeps:
    - robot persona
    - robot names such as Willi and WV-34
    - speech-recognition errors
    - repetitions
    - awkward or incomplete wording
    """
    visible_turns: List[str] = []

    for msg in dialog.get("messages") or []:
        role = normalize_text(msg.get("role"))
        if role == "system":
            continue

        if role not in ROLE_LABELS:
            continue

        content = clean_content(msg.get("content", ""))
        if not content:
            continue

        label = ROLE_LABELS[role]
        visible_turns.append(f"{label}: {content}")

    return "\n\n".join(visible_turns)


def count_visible_roles(dialog: Dict[str, Any]) -> Tuple[int, int, int]:
    """Return number of visible user, robot, and total turns."""
    n_user = 0
    n_robot = 0

    for msg in dialog.get("messages") or []:
        role = normalize_text(msg.get("role"))
        if role == "user":
            n_user += 1
        elif role == "assistant":
            n_robot += 1

    return n_user, n_robot, n_user + n_robot


def make_row(dialog: Dict[str, Any], subject: str, language: str, sample_stratum: str) -> Dict[str, Any]:
    """Convert one dialog into one CSV row."""
    n_user, n_robot, n_visible_turns = count_visible_roles(dialog)

    return {
        # Hidden metadata columns for Prolific.
        "META_dialog_id": dialog.get("id", ""),
        "META_condition": dialog.get("condition", ""),
        "META_feedback": "" if dialog.get("feedback") is None else dialog.get("feedback"),
        "META_timestamp": dialog.get("timestamp", ""),
        "META_original_topics": json.dumps(dialog.get("topics") or [], ensure_ascii=False),
        "META_sample_stratum": sample_stratum,
        "META_n_user_turns": n_user,
        "META_n_robot_turns": n_robot,
        "META_n_visible_turns": n_visible_turns,

        # Visible annotation columns.
        "language": language,
        "subject": subject,
        "dialog_text": build_dialog_text(dialog),
    }


def prepare_eligible_dialogs(dialogs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter]:
    """
    Add inferred fields and keep only dialogs that are suitable for the pilot batch.
    """
    eligible = []
    exclusion_counts = Counter()

    for dialog in dialogs:
        subject = infer_subject(dialog)
        if subject is None:
            exclusion_counts["excluded_no_or_multiple_subjects"] += 1
            continue

        language = detect_language(dialog)
        if language not in LANGUAGES:
            exclusion_counts["excluded_unknown_language"] += 1
            continue

        dialog_text = build_dialog_text(dialog)
        if not dialog_text.strip():
            exclusion_counts["excluded_empty_dialog_text"] += 1
            continue

        # Make a shallow copy so we can attach inferred fields safely.
        enriched = dict(dialog)
        enriched["_subject"] = subject
        enriched["_language"] = language
        eligible.append(enriched)

    return eligible, exclusion_counts


def sample_one_stratum(
    eligible: List[Dict[str, Any]],
    subject: str,
    language: str,
    per_stratum: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Sample one subject-language stratum.

    Tries to select 5 Willi and 5 WV-34 when per_stratum=10.
    If exact condition balance is not possible, it fills from the same
    subject-language stratum and prints a warning.
    """
    pool = [
        d for d in eligible
        if d.get("_subject") == subject and d.get("_language") == language
    ]

    if len(pool) < per_stratum:
        raise ValueError(
            f"Not enough dialogs for subject={subject}, language={language}. "
            f"Need {per_stratum}, found {len(pool)}."
        )

    target_per_condition = per_stratum // 2
    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for dialog in pool:
        by_condition[str(dialog.get("condition", ""))].append(dialog)

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

    selected_ids = {str(d.get("id")) for d in selected}
    remaining = [d for d in pool if str(d.get("id")) not in selected_ids]

    if len(selected) < per_stratum:
        need = per_stratum - len(selected)
        selected.extend(rng.sample(remaining, need))

    # If per_stratum is odd in future use, top up from remaining.
    if len(selected) > per_stratum:
        selected = rng.sample(selected, per_stratum)

    rng.shuffle(selected)
    return selected


def sample_pilot_batch(
    eligible: List[Dict[str, Any]],
    n_per_subject_language: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Sample the full pilot batch."""
    selected: List[Dict[str, Any]] = []

    for subject in SUBJECTS:
        for language in LANGUAGES:
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


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "META_dialog_id",
        "META_condition",
        "META_feedback",
        "META_timestamp",
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


def split_rows_by_language(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split already sampled rows into separate language-specific batches."""
    rows_by_language: Dict[str, List[Dict[str, Any]]] = {language: [] for language in LANGUAGES}

    for row in rows:
        language = row.get("language")
        if language in rows_by_language:
            rows_by_language[language].append(row)

    return rows_by_language


def print_summary(
    dialogs: List[Dict[str, Any]],
    eligible: List[Dict[str, Any]],
    sampled_rows: List[Dict[str, Any]],
    exclusion_counts: Counter,
    output_de_path: Path,
    output_en_path: Path,
) -> None:
    """Print a compact summary for checking the split batches."""
    print("\n==============================")
    print("Prolific Pilot Batch Summary")
    print("==============================")
    print(f"Loaded dialogs: {len(dialogs)}")
    print(f"Eligible clean single-subject dialogs: {len(eligible)}")

    if exclusion_counts:
        print("\nExcluded dialogs:")
        for key, value in exclusion_counts.items():
            print(f"  {key}: {value}")

    rows_by_language = split_rows_by_language(sampled_rows)

    print(f"\nOutput rows total: {len(sampled_rows)}")
    print(f"German output rows: {len(rows_by_language['German'])}")
    print(f"English output rows: {len(rows_by_language['English'])}")
    print(f"German output file: {output_de_path}")
    print(f"English output file: {output_en_path}")

    print("\nCounts by subject and language:")
    by_subject_language = Counter(
        (row["subject"], row["language"]) for row in sampled_rows
    )
    for subject in SUBJECTS:
        for language in LANGUAGES:
            print(f"  {subject:9s} / {language:7s}: {by_subject_language[(subject, language)]}")

    print("\nCounts by subject, language, and condition:")
    by_subject_language_condition = Counter(
        (row["subject"], row["language"], row["META_condition"])
        for row in sampled_rows
    )
    for subject in SUBJECTS:
        for language in LANGUAGES:
            for condition in CONDITIONS:
                print(
                    f"  {subject:9s} / {language:7s} / {condition:22s}: "
                    f"{by_subject_language_condition[(subject, language, condition)]}"
                )

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Create split German and English 30-dialog pilot CSV batches "
            "for Prolific annotation."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "dialogs.json",
        help="Path to input dialogs.json. Default: ../dialogs.json relative to 05_prolific.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory for output CSVs. Default: 05_prolific folder.",
    )
    parser.add_argument(
        "--output-de",
        type=Path,
        default=None,
        help=(
            "Path to German output CSV. "
            "Default: <output-dir>/pilot_batch_de.csv."
        ),
    )
    parser.add_argument(
        "--output-en",
        type=Path,
        default=None,
        help=(
            "Path to English output CSV. "
            "Default: <output-dir>/pilot_batch_en.csv."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Default: 42.",
    )
    parser.add_argument(
        "--n-per-subject-language",
        type=int,
        default=10,
        help=(
            "Number of dialogs per subject-language cell. "
            "Default 10 gives 3 subjects x 10 = 30 dialogs per language, "
            "and 60 dialogs total."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_de_path = (
        args.output_de.resolve()
        if args.output_de is not None
        else output_dir / "pilot_batch_de.csv"
    )
    output_en_path = (
        args.output_en.resolve()
        if args.output_en is not None
        else output_dir / "pilot_batch_en.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rng = random.Random(args.seed)

    dialogs = load_dialogs(input_path)
    eligible, exclusion_counts = prepare_eligible_dialogs(dialogs)

    sampled_dialogs = sample_pilot_batch(
        eligible=eligible,
        n_per_subject_language=args.n_per_subject_language,
        rng=rng,
    )

    rows = []
    for dialog in sampled_dialogs:
        subject = dialog["_subject"]
        language = dialog["_language"]
        sample_stratum = f"{subject}_{language}"
        rows.append(make_row(dialog, subject, language, sample_stratum))

    rows_by_language = split_rows_by_language(rows)
    german_rows = rows_by_language["German"]
    english_rows = rows_by_language["English"]

    expected_per_language = len(SUBJECTS) * args.n_per_subject_language
    if len(german_rows) != expected_per_language:
        raise ValueError(
            f"German batch should contain {expected_per_language} rows, "
            f"but contains {len(german_rows)}."
        )
    if len(english_rows) != expected_per_language:
        raise ValueError(
            f"English batch should contain {expected_per_language} rows, "
            f"but contains {len(english_rows)}."
        )

    write_csv(german_rows, output_de_path)
    write_csv(english_rows, output_en_path)
    print_summary(
        dialogs,
        eligible,
        rows,
        exclusion_counts,
        output_de_path,
        output_en_path,
    )


if __name__ == "__main__":
    main()
