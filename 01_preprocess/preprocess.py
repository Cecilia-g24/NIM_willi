"""
Refactored preprocessing script for the human-robot interaction dialog data.

What this script does:
1. Loads the same raw JSON input path as the original preprocess.py.
2. Removes empty / malformed interactions.
3. Classifies each dialog into Condition A (Willi), Condition B (WV-34), or Unknown.
4. Detects dialog language as English or German.
5. Saves original dialogs, split by language, as CSV files.
6. Creates annotation-ready dialog text:
   - removes all system messages
   - keeps the conversation as turn-by-turn text
   - renames roles to Robot / Visitor or Roboter / Besucher
   - converts robot gesture markers such as <<smile>> into readable text
   - keeps user disfluencies and ASR artifacts unchanged
7. Saves annotation-ready dialogs, split by language, as CSV files.
8. Saves audit files for removed and unknown-condition dialogs.

Expected input location:
    data/data_raw/dialogs-1771498506071_raw.json

CSV output folder:
    data/data_clean/

JSON report folder:
    01_preprocess/preprocess_report/

Main output files:
    data/data_clean/dialogs_en.csv
    data/data_clean/dialogs_de.csv
    data/data_clean/dialogs_for_annotation_en.csv
    data/data_clean/dialogs_for_annotation_de.csv
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

INPUT_FILE = REPO_ROOT / "data" / "data_raw" / "dialogs-1771498506071_raw.json"
CSV_OUTPUT_DIR = REPO_ROOT / "data" / "data_clean"
REPORT_DIR = SCRIPT_DIR / "preprocess_report"

ORIGINAL_EN_CSV = CSV_OUTPUT_DIR / "dialogs_en.csv"
ORIGINAL_DE_CSV = CSV_OUTPUT_DIR / "dialogs_de.csv"
ANNOTATION_EN_CSV = CSV_OUTPUT_DIR / "dialogs_for_annotation_en.csv"
ANNOTATION_DE_CSV = CSV_OUTPUT_DIR / "dialogs_for_annotation_de.csv"

# Optional audit JSON outputs and reports.
CLEANED_JSON_FILE = REPORT_DIR / "dialogs_cleaned_with_metadata.json"
REMOVED_IDS_FILE = REPORT_DIR / "removed_ids.json"
UNKNOWN_ENTRIES_FILE = REPORT_DIR / "unknown_entries.json"
REPORT_FILE = REPORT_DIR / "preprocessing_report.json"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CONDITION_A = "Condition A (Willi)"
CONDITION_B = "Condition B (WV-34)"
CONDITION_UNKNOWN = "Condition Unknown"

LANG_EN = "en"
LANG_DE = "de"

GESTURE_REPLACEMENTS = {
    "smile": "robot smiles",
    "bigsmile": "robot smiles broadly",
    "browraise": "robot raises eyebrows",
    "thoughtful": "robot looks thoughtful",
    "wink": "robot winks",
    "gazeaway": "robot looks away",
    "neutral": "robot has a neutral expression",
    "surprised": "robot looks surprised",
    "sad": "robot looks sad",
    "happy": "robot looks happy",
}

TOPIC_MAP = {
    "breakfast": "Breakfast",
    "frühstück": "Breakfast",
    "fruehstueck": "Breakfast",
    "watches": "Watches",
    "uhren": "Watches",
    "vacation": "Vacation",
    "urlaub": "Vacation",
}

IGNORED_TOPIC_VALUES = {
    "",
    "none",
    "null",
    "general",
    "waiting_for_choice",
    "wartet_auf_auswahl",
}

# -----------------------------------------------------------------------------
# Basic file helpers
# -----------------------------------------------------------------------------


def load_json(filepath: Path) -> Any:
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filepath: Path, data: Any) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(filepath: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Validation and metadata helpers
# -----------------------------------------------------------------------------


def is_empty_interaction(item: Any) -> bool:
    """Return True for malformed or empty dialog records."""
    if not isinstance(item, dict):
        return True

    messages = item.get("messages")
    if not isinstance(messages, list):
        return True

    return len(messages) == 0


def iter_messages(dialog: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield only dictionary messages from a dialog."""
    for message in dialog.get("messages") or []:
        if isinstance(message, dict):
            yield message


def get_first_assistant_message(messages: List[Dict[str, Any]]) -> str:
    """Return the first assistant message content, if available."""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content", "") or "")
    return ""


def get_leading_system_message(messages: List[Dict[str, Any]]) -> str:
    """Return the first message if it is a system prompt; otherwise empty string."""
    if (
        isinstance(messages, list)
        and messages
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "system"
    ):
        return str(messages[0].get("content", "") or "")
    return ""


def build_message_snippet(message: str, max_chars: int = 180) -> str:
    if not message:
        return "[No message found]"
    return normalize_whitespace(message)[:max_chars]


# -----------------------------------------------------------------------------
# Condition classification
# -----------------------------------------------------------------------------


def classify_from_assistant_message(message: str) -> Optional[str]:
    """Classify condition from the first assistant greeting."""
    if "WV-34" in message:
        return CONDITION_B
    if "Willi" in message:
        return CONDITION_A
    return None


def classify_from_system_prompt(prompt: str) -> Optional[str]:
    """Classify condition from a leading system prompt, used only as fallback."""
    prompt_lower = prompt.lower()

    # Strong WV-34 cues.
    if "wv-34" in prompt_lower:
        return CONDITION_B
    if "ohne mimik" in prompt_lower and "deine stimme klingt blechern" in prompt_lower:
        return CONDITION_B
    if "keine mimik oder augenbewegungen" in prompt_lower:
        return CONDITION_B

    # Strong Willi cues.
    if "du bist willi" in prompt_lower:
        return CONDITION_A
    if "mit mimik" in prompt_lower:
        return CONDITION_A
    if "augenbewegung und stimme stimmung vermitteln" in prompt_lower:
        return CONDITION_A

    return None


def classify_condition(dialog: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return the condition and the classification source.

    The condition is assigned before removing system messages.
    """
    messages = dialog.get("messages") or []
    first_assistant = get_first_assistant_message(messages)

    condition = classify_from_assistant_message(first_assistant)
    if condition is not None:
        return condition, "assistant"

    leading_system = get_leading_system_message(messages)
    condition = classify_from_system_prompt(leading_system)
    if condition is not None:
        return condition, "system_fallback"

    return CONDITION_UNKNOWN, "unknown"


# -----------------------------------------------------------------------------
# Language detection
# -----------------------------------------------------------------------------


def detect_language(dialog: Dict[str, Any]) -> str:
    """
    Detect German vs. English.

    This mirrors your earlier logic:
    1. Use explicit system-message hints first.
    2. Fall back to German/English marker scoring.
    """
    all_text = "\n".join(str(msg.get("content", "")) for msg in iter_messages(dialog))
    lower = f" {all_text.lower()} "

    if "remember to only reply in english" in lower:
        return LANG_EN
    if "antworte immer nur auf deutsch" in lower:
        return LANG_DE

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

    return LANG_DE if german_score > english_score else LANG_EN


# -----------------------------------------------------------------------------
# Text cleaning and formatting
# -----------------------------------------------------------------------------


def normalize_whitespace(text: Any) -> str:
    """Normalize whitespace inside one message while preserving message content."""
    text = str(text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def convert_gesture_markers(text: str, keep_gestures: bool = True) -> str:
    """
    Convert markers like <<smile>> into readable annotation text.

    If keep_gestures=False, all <<...>> markers are removed.
    """
    if not keep_gestures:
        return re.sub(r"<<[^>]+>>", "", text).strip()

    def replace_marker(match: re.Match[str]) -> str:
        raw_marker = match.group(1).strip().lower()
        marker_text = GESTURE_REPLACEMENTS.get(raw_marker)
        if marker_text is None:
            marker_text = f"robot {raw_marker.replace('_', ' ')}"
        return f"[{marker_text}]"

    return re.sub(r"<<([^>]+)>>", replace_marker, text).strip()


def clean_message_content(content: Any, keep_gestures: bool = True) -> str:
    """Clean one message content for annotation display."""
    text = normalize_whitespace(content)
    text = convert_gesture_markers(text, keep_gestures=keep_gestures)
    text = normalize_whitespace(text)
    return text


def role_label(role: str, language: str) -> Optional[str]:
    """Map JSON roles to human-readable speaker labels."""
    if language == LANG_DE:
        labels = {
            "assistant": "Roboter",
            "user": "Besucher",
        }
    else:
        labels = {
            "assistant": "Robot",
            "user": "Visitor",
        }
    return labels.get(role)


def role_label_original(role: str) -> str:
    """Labels for audit/original dialog text."""
    return {
        "assistant": "assistant",
        "user": "user",
        "system": "system",
    }.get(role, role or "unknown")


def format_original_dialogue(dialog: Dict[str, Any]) -> str:
    """
    Format the original dialog as text for audit.

    This keeps all roles, including system messages, because this file is meant
    to preserve the original conversation structure.
    """
    turns: List[str] = []
    for message in iter_messages(dialog):
        role = role_label_original(str(message.get("role", "")))
        content = normalize_whitespace(message.get("content", ""))
        if content:
            turns.append(f"{role}: {content}")
    return "\n\n".join(turns)


def format_annotation_dialogue(
    dialog: Dict[str, Any],
    language: str,
    keep_gestures: bool = True,
) -> str:
    """
    Format one dialog for human annotation.

    This removes all system messages and keeps only the visible human-robot
    interaction as chronological turn-by-turn text.
    """
    turns: List[str] = []

    for message in iter_messages(dialog):
        role = str(message.get("role", ""))

        # Important: remove all system prompts, not just a leading one.
        if role == "system":
            continue

        speaker = role_label(role, language)
        if speaker is None:
            continue

        content = clean_message_content(message.get("content", ""), keep_gestures=keep_gestures)
        if content:
            turns.append(f"{speaker}: {content}")

    return "\n\n".join(turns)


def collect_role_text(dialog: Dict[str, Any], role: str, keep_gestures: bool = True) -> str:
    """Collect all visible text from one role. Useful for later feature extraction."""
    texts: List[str] = []
    for message in iter_messages(dialog):
        if message.get("role") != role:
            continue
        text = clean_message_content(message.get("content", ""), keep_gestures=keep_gestures)
        if text:
            texts.append(text)
    return "\n".join(texts)


# -----------------------------------------------------------------------------
# Topic and count helpers
# -----------------------------------------------------------------------------


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def get_topic_main(dialog: Dict[str, Any]) -> str:
    """Return a canonical topic label where possible."""
    raw_topics = dialog.get("topics") or []

    # First use the dialog-level topics.
    if isinstance(raw_topics, list):
        for topic in raw_topics:
            topic_text = normalize_whitespace(topic).lower()
            if topic_text in IGNORED_TOPIC_VALUES:
                continue
            if topic_text in TOPIC_MAP:
                return TOPIC_MAP[topic_text]

    # Fallback: use per-message topic values.
    for message in iter_messages(dialog):
        topic_text = normalize_whitespace(message.get("topic", "")).lower()
        if topic_text in IGNORED_TOPIC_VALUES:
            continue
        if topic_text in TOPIC_MAP:
            return TOPIC_MAP[topic_text]

    return "Unknown"


def count_role(dialog: Dict[str, Any], role: str) -> int:
    return sum(1 for msg in iter_messages(dialog) if msg.get("role") == role)


def count_visible_turns(dialog: Dict[str, Any]) -> int:
    return sum(1 for msg in iter_messages(dialog) if msg.get("role") in {"assistant", "user"})


# -----------------------------------------------------------------------------
# Row builders
# -----------------------------------------------------------------------------


def build_original_row(dialog: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Build one CSV row preserving the original dialog content."""
    return {
        "dialog_id": dialog.get("id"),
        "timestamp": dialog.get("timestamp"),
        "language": language,
        "condition": dialog.get("condition"),
        "condition_source": dialog.get("condition_source"),
        "topic_main": get_topic_main(dialog),
        "topics_json": safe_json_dumps(dialog.get("topics")),
        "feedback": dialog.get("feedback"),
        "n_messages_total": len(list(iter_messages(dialog))),
        "n_system_turns": count_role(dialog, "system"),
        "n_robot_turns": count_role(dialog, "assistant"),
        "n_visitor_turns": count_role(dialog, "user"),
        "messages_json": safe_json_dumps(dialog.get("messages")),
        "dialogue_original_text": format_original_dialogue(dialog),
    }


def build_annotation_row(dialog: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Build one annotation-ready CSV row."""
    return {
        "dialog_id": dialog.get("id"),
        "language": language,
        # Keep condition as hidden metadata for analysis. Do not display this column to annotators.
        "condition_hidden": dialog.get("condition"),
        "topic_main": get_topic_main(dialog),
        "topics_json": safe_json_dumps(dialog.get("topics")),
        "feedback_existing": dialog.get("feedback"),
        "n_turns_visible": count_visible_turns(dialog),
        "n_robot_turns": count_role(dialog, "assistant"),
        "n_visitor_turns": count_role(dialog, "user"),
        "n_system_turns_removed": count_role(dialog, "system"),
        "dialogue_for_annotation": format_annotation_dialogue(dialog, language=language, keep_gestures=True),
        "robot_only_text": collect_role_text(dialog, "assistant", keep_gestures=True),
        "visitor_only_text": collect_role_text(dialog, "user", keep_gestures=False),
    }


# -----------------------------------------------------------------------------
# Main preprocessing pipeline
# -----------------------------------------------------------------------------


def preprocess_dialogs(input_file: Path = INPUT_FILE, csv_output_dir: Path = CSV_OUTPUT_DIR) -> None:
    data = load_json(input_file)

    if not isinstance(data, list):
        raise ValueError("The top-level JSON structure must be a list/array of dialog records.")

    csv_output_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_items = len(data)
    removed_ids: List[Any] = []
    unknown_entries: List[Dict[str, Any]] = []
    cleaned_dialogs: List[Dict[str, Any]] = []

    stats: Counter = Counter()
    language_condition_counts: Counter = Counter()

    for item in data:
        if is_empty_interaction(item):
            removed_ids.append(item.get("id") if isinstance(item, dict) else None)
            continue

        # Work on a shallow copy so we do not mutate the loaded object unexpectedly.
        dialog = dict(item)

        condition, condition_source = classify_condition(dialog)
        language = detect_language(dialog)

        dialog["condition"] = condition
        dialog["condition_source"] = condition_source
        dialog["language"] = language
        dialog["topic_main"] = get_topic_main(dialog)

        cleaned_dialogs.append(dialog)

        stats["retained_interactions"] += 1
        stats[f"condition::{condition}"] += 1
        stats[f"condition_source::{condition_source}"] += 1
        stats[f"language::{language}"] += 1
        stats["system_turns_total"] += count_role(dialog, "system")
        language_condition_counts[(language, condition)] += 1

        if condition == CONDITION_UNKNOWN:
            messages = dialog.get("messages") or []
            first_assistant = get_first_assistant_message(messages)
            leading_system = get_leading_system_message(messages)
            unknown_entries.append(
                {
                    "id": dialog.get("id"),
                    "language": language,
                    "assistant_snippet": build_message_snippet(first_assistant),
                    "system_snippet": build_message_snippet(leading_system),
                }
            )

    en_dialogs = [dialog for dialog in cleaned_dialogs if dialog.get("language") == LANG_EN]
    de_dialogs = [dialog for dialog in cleaned_dialogs if dialog.get("language") == LANG_DE]

    original_fieldnames = [
        "dialog_id",
        "timestamp",
        "language",
        "condition",
        "condition_source",
        "topic_main",
        "topics_json",
        "feedback",
        "n_messages_total",
        "n_system_turns",
        "n_robot_turns",
        "n_visitor_turns",
        "messages_json",
        "dialogue_original_text",
    ]

    annotation_fieldnames = [
        "dialog_id",
        "language",
        "condition_hidden",
        "topic_main",
        "topics_json",
        "feedback_existing",
        "n_turns_visible",
        "n_robot_turns",
        "n_visitor_turns",
        "n_system_turns_removed",
        "dialogue_for_annotation",
        "robot_only_text",
        "visitor_only_text",
    ]

    write_csv(ORIGINAL_EN_CSV, [build_original_row(d, LANG_EN) for d in en_dialogs], original_fieldnames)
    write_csv(ORIGINAL_DE_CSV, [build_original_row(d, LANG_DE) for d in de_dialogs], original_fieldnames)
    write_csv(ANNOTATION_EN_CSV, [build_annotation_row(d, LANG_EN) for d in en_dialogs], annotation_fieldnames)
    write_csv(ANNOTATION_DE_CSV, [build_annotation_row(d, LANG_DE) for d in de_dialogs], annotation_fieldnames)

    save_json(CLEANED_JSON_FILE, cleaned_dialogs)
    save_json(REMOVED_IDS_FILE, removed_ids)
    save_json(UNKNOWN_ENTRIES_FILE, unknown_entries)

    report = {
        "input_file": str(input_file),
        "csv_output_dir": str(csv_output_dir),
        "report_dir": str(REPORT_DIR),
        "total_interactions_in_input": total_items,
        "retained_interactions": len(cleaned_dialogs),
        "removed_empty_or_malformed_interactions": len(removed_ids),
        "english_dialogs": len(en_dialogs),
        "german_dialogs": len(de_dialogs),
        "total_system_turns_removed_from_annotation_text": stats["system_turns_total"],
        "condition_counts": {
            CONDITION_A: stats[f"condition::{CONDITION_A}"],
            CONDITION_B: stats[f"condition::{CONDITION_B}"],
            CONDITION_UNKNOWN: stats[f"condition::{CONDITION_UNKNOWN}"],
        },
        "condition_source_counts": {
            "assistant": stats["condition_source::assistant"],
            "system_fallback": stats["condition_source::system_fallback"],
            "unknown": stats["condition_source::unknown"],
        },
        "language_condition_counts": {
            f"{language} / {condition}": count
            for (language, condition), count in sorted(language_condition_counts.items())
        },
        "output_files": [
            str(ORIGINAL_EN_CSV),
            str(ORIGINAL_DE_CSV),
            str(ANNOTATION_EN_CSV),
            str(ANNOTATION_DE_CSV),
            str(CLEANED_JSON_FILE),
            str(REMOVED_IDS_FILE),
            str(UNKNOWN_ENTRIES_FILE),
            str(REPORT_FILE),
        ],
    }
    save_json(REPORT_FILE, report)

    print("\n=== PREPROCESSING REPORT ===")
    print(f"Input file: {input_file}")
    print(f"CSV output folder: {csv_output_dir}")
    print(f"Report folder: {REPORT_DIR}")
    print(f"Total interactions in input: {total_items}")
    print(f"Retained interactions: {len(cleaned_dialogs)}")
    print(f"Removed empty/malformed interactions: {len(removed_ids)}")
    print(f"English dialogs: {len(en_dialogs)}")
    print(f"German dialogs: {len(de_dialogs)}")
    print(f"System turns removed from annotation text: {stats['system_turns_total']}")

    print("\nCondition counts:")
    for condition in [CONDITION_A, CONDITION_B, CONDITION_UNKNOWN]:
        print(f"  {condition}: {stats[f'condition::{condition}']}")

    print("\nCondition source counts:")
    for source in ["assistant", "system_fallback", "unknown"]:
        print(f"  {source}: {stats[f'condition_source::{source}']}")

    print("\nSaved files:")
    for path in report["output_files"]:
        print(f"  {path}")


if __name__ == "__main__":
    preprocess_dialogs()
