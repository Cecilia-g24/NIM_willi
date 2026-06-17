import json
from pathlib import Path

# -----------------------------------------------------------------------------
# Preprocessing script for dialogs.json
#
# Logic implemented in this file:
#
# 1. Load the original dialogs JSON file.
# 2. Remove empty interactions.
#    An interaction is treated as empty if:
#    - it is not a dictionary, or
#    - "messages" is missing, or
#    - "messages" is not a list, or
#    - "messages" is an empty list.
#
# 3. Assign the condition BEFORE removing the leading system prompt.
#    Condition assignment follows this order:
#
#    A) First assistant greeting (preferred signal)
#       - If the first assistant message contains "Willi",
#         assign: "Condition A (Willi)"
#       - If the first assistant message contains "WV-34",
#         assign: "Condition B (WV-34)"
#
#       Why this is the preferred signal:
#       - It is short and easy to parse.
#       - It reflects the persona actually shown to the participant.
#       - In this dataset it is a strong practical proxy for condition.
#
#    B) Fallback to the leading system prompt (backup signal)
#       This fallback is used only if the first assistant message does not
#       clearly identify the persona.
#
#       - If the leading system prompt indicates the Willi persona,
#         assign: "Condition A (Willi)"
#       - If the leading system prompt indicates the WV-34 persona,
#         assign: "Condition B (WV-34)"
#
#       This makes the classification more robust in case the assistant
#       greeting is missing, malformed, or does not explicitly mention the
#       persona name.
#
#    C) If neither source identifies the persona, assign:
#       "Condition Unknown"
#
# 4. Remove the leading system prompt from retained interactions.
#    This keeps the dataset smaller and cleaner for condition-level analysis.
#
# 5. Save:
#    - dialogs_preprocessed.json : cleaned interactions
#    - removed_ids.json          : IDs of removed empty interactions
#    - unknown_entries.json      : entries whose condition could not be inferred
#
# -----------------------------------------------------------------------------

INPUT_FILE = "dialogs-1771498506071_raw.json"
OUTPUT_FILE = "../dialogs.json"
REMOVED_IDS_FILE = "removed_ids.json"
UNKNOWN_ENTRIES_FILE = "unknown_entries.json"

CONDITION_A = "Condition A (Willi)"
CONDITION_B = "Condition B (WV-34)"
CONDITION_UNKNOWN = "Condition Unknown"


def load_json(filepath: str):
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filepath: str, data) -> None:
    path = Path(filepath)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_empty_interaction(item) -> bool:
    if not isinstance(item, dict):
        return True
    messages = item.get("messages")
    if not isinstance(messages, list):
        return True
    if len(messages) == 0:
        return True
    return False


def get_first_assistant_message(messages: list[dict]) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message.get("content", "") or ""
    return ""


def get_leading_system_message(messages: list[dict]) -> str:
    if (
        isinstance(messages, list)
        and len(messages) > 0
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "system"
    ):
        return messages[0].get("content", "") or ""
    return ""


def classify_from_assistant_message(message: str) -> str | None:
    if "WV-34" in message:
        return CONDITION_B
    if "Willi" in message:
        return CONDITION_A
    return None


def classify_from_system_prompt(prompt: str) -> str | None:
    prompt_lower = prompt.lower()

    # Strong WV-34 cues
    if "wv-34" in prompt_lower:
        return CONDITION_B
    if "ohne mimik" in prompt_lower and "deine stimme klingt blechern" in prompt_lower:
        return CONDITION_B
    if "keine mimik oder augenbewegungen" in prompt_lower:
        return CONDITION_B

    # Strong Willi cues
    if "du bist willi" in prompt_lower:
        return CONDITION_A
    if "mit mimik" in prompt_lower:
        return CONDITION_A
    if "augenbewegung und stimme stimmung vermitteln" in prompt_lower:
        return CONDITION_A

    return None


def classify_condition(dialog: dict) -> tuple[str, str]:
    """
    Returns:
        (condition, source)
        source is one of:
        - "assistant"
        - "system_fallback"
        - "unknown"
    """
    messages = dialog.get("messages", [])
    first_assistant = get_first_assistant_message(messages)
    condition = classify_from_assistant_message(first_assistant)
    if condition is not None:
        return condition, "assistant"

    leading_system = get_leading_system_message(messages)
    condition = classify_from_system_prompt(leading_system)
    if condition is not None:
        return condition, "system_fallback"

    return CONDITION_UNKNOWN, "unknown"


def remove_leading_system_message(item: dict) -> tuple[dict, bool]:
    cleaned_item = dict(item)
    messages = cleaned_item.get("messages", [])

    had_leading_system = (
        isinstance(messages, list)
        and len(messages) > 0
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "system"
    )

    if had_leading_system:
        cleaned_item["messages"] = messages[1:]

    return cleaned_item, had_leading_system


def build_message_snippet(message: str) -> str:
    if not message:
        return "[No Assistant Message found]"
    return message[:150].replace("\n", " ")


def preprocess_dialogs(input_file: str, output_file: str) -> None:
    data = load_json(input_file)

    if not isinstance(data, list):
        raise ValueError("The top-level JSON structure must be a list/array.")

    total_items = len(data)
    cleaned_data = []
    removed_ids = []
    unknown_entries = []

    leading_system_removed_count = 0
    no_leading_system_count = 0

    stats = {
        CONDITION_A: 0,
        CONDITION_B: 0,
        CONDITION_UNKNOWN: 0,
        "assistant_classification": 0,
        "system_fallback_classification": 0,
        "unknown_classification": 0,
    }

    for item in data:
        if is_empty_interaction(item):
            if isinstance(item, dict) and "id" in item:
                removed_ids.append(item["id"])
            else:
                removed_ids.append(None)
            continue

        condition, source = classify_condition(item)

        if source == "assistant":
            stats["assistant_classification"] += 1
        elif source == "system_fallback":
            stats["system_fallback_classification"] += 1
        else:
            stats["unknown_classification"] += 1

        stats[condition] += 1

        if condition == CONDITION_UNKNOWN:
            messages = item.get("messages", [])
            first_assistant = get_first_assistant_message(messages)
            leading_system = get_leading_system_message(messages)
            unknown_entries.append(
                {
                    "id": item.get("id"),
                    "condition": condition,
                    "assistant_snippet": build_message_snippet(first_assistant),
                    "system_snippet": build_message_snippet(leading_system),
                }
            )

        cleaned_item, had_leading_system = remove_leading_system_message(item)
        cleaned_item["condition"] = condition

        if had_leading_system:
            leading_system_removed_count += 1
        else:
            no_leading_system_count += 1

        cleaned_data.append(cleaned_item)

    print("\n=== PREPROCESSING REPORT ===")
    print(f"Total interactions: {total_items}")
    print(f"Retained interactions: {len(cleaned_data)}")
    print(f"Empty interactions removed: {len(removed_ids)}")
    print(f"Leading system prompts removed: {leading_system_removed_count}")
    print(f"Retained interactions without leading system prompt: {no_leading_system_count}")
    print(f"\nCondition counts:")
    print(f"  {CONDITION_A}: {stats[CONDITION_A]}")
    print(f"  {CONDITION_B}: {stats[CONDITION_B]}")
    print(f"  {CONDITION_UNKNOWN}: {stats[CONDITION_UNKNOWN]}")
    print(f"\nClassification source counts:")
    print(f"  First assistant greeting: {stats['assistant_classification']}")
    print(f"  System prompt fallback: {stats['system_fallback_classification']}")
    print(f"  Unknown after both checks: {stats['unknown_classification']}")

    print("\nRemoved interaction IDs:")
    print(removed_ids)

    save_json(REMOVED_IDS_FILE, removed_ids)
    save_json(UNKNOWN_ENTRIES_FILE, unknown_entries)
    save_json(output_file, cleaned_data)

    print(f"\nSaved removed IDs to: {REMOVED_IDS_FILE}")
    print(f"Saved unknown entries to: {UNKNOWN_ENTRIES_FILE}")
    print(f"Saved preprocessed interactions to: {output_file}")


if __name__ == "__main__":
    preprocess_dialogs(INPUT_FILE, OUTPUT_FILE)
