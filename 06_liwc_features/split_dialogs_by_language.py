"""
Split dialogs.json into dialogs_de.json and dialogs_en.json.

Language detection mirrors 05_prolific/create_batch_seperate.py:detect_language
(system-message hints first, then German/English marker scoring) so language
labels stay consistent across the project.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

INPUT_PATH = Path(__file__).resolve().parent.parent / "dialogs.json"
OUTPUT_DE_PATH = Path(__file__).resolve().parent / "dialogs_de.json"
OUTPUT_EN_PATH = Path(__file__).resolve().parent / "dialogs_en.json"


def detect_language(dialog: Dict[str, Any]) -> str:
    """Detect German vs. English using system hints first, then text markers."""
    all_text = "\n".join(
        str(msg.get("content", "")) for msg in (dialog.get("messages") or [])
    )
    lower = f" {all_text.lower()} "

    if "remember to only reply in english" in lower:
        return "English"
    if "antworte immer nur auf deutsch" in lower:
        return "German"

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


def main() -> None:
    dialogs: List[Dict[str, Any]] = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    de_dialogs: List[Dict[str, Any]] = []
    en_dialogs: List[Dict[str, Any]] = []

    condition_counts: Counter = Counter()

    for dialog in dialogs:
        language = detect_language(dialog)
        condition_counts[(language, dialog.get("condition"))] += 1

        if language == "German":
            de_dialogs.append(dialog)
        else:
            en_dialogs.append(dialog)

    OUTPUT_DE_PATH.write_text(
        json.dumps(de_dialogs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUTPUT_EN_PATH.write_text(
        json.dumps(en_dialogs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Total dialogs: {len(dialogs)}")
    print(f"German dialogs: {len(de_dialogs)} -> {OUTPUT_DE_PATH.name}")
    print(f"English dialogs: {len(en_dialogs)} -> {OUTPUT_EN_PATH.name}")

    print("\nCounts by language and condition:")
    for (language, condition) in sorted(condition_counts):
        print(f"  {language:7s} / {condition!s:25s}: {condition_counts[(language, condition)]}")


if __name__ == "__main__":
    main()
