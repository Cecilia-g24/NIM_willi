#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


CONDITION_A = "Condition A (Willi)"
CONDITION_B = "Condition B (WV-34)"
VALID_CONDITIONS = {CONDITION_A, CONDITION_B}

TOPIC_MAP = {
    "frühstück": "Breakfast",
    "breakfast": "Breakfast",
    "uhren": "Watches",
    "watches": "Watches",
    "urlaub": "Vacation",
    "vacation": "Vacation",
}

GERMAN_PRONOUNS_FIRST = {"ich", "mich", "mir", "mein", "meine", "meiner", "meinem", "meinen", "wir", "uns", "unser", "unsere"}
GERMAN_PRONOUNS_SECOND = {"du", "dich", "dir", "dein", "deine", "deiner", "deinem", "deinen", "sie", "ihnen", "ihr", "ihre"}
ENGLISH_PRONOUNS_FIRST = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
ENGLISH_PRONOUNS_SECOND = {"you", "your", "yours"}

POLITENESS_MARKERS = {
    "thanks", "thank", "thankyou", "thank-you", "danke", "bitte", "please",
    "hallo", "hello", "hi", "tschüss", "tschuess", "ciao", "bye", "goodbye",
}
GREETING_MARKERS = {"hallo", "hello", "hi", "hey", "guten", "servus", "willkommen", "welcome"}
FAREWELL_MARKERS = {"tschüss", "tschuess", "ciao", "bye", "goodbye", "auf wiedersehen", "bis bald", "gute nacht"}
ACK_MARKERS = {"okay", "ok", "ja", "yes", "genau", "klar", "alright", "super", "gut"}
REFUSAL_MARKERS = {"nein", "no", "nope", "möchte nicht", "will nicht", "don't want", "do not want", "kein interesse"}

CLARIFICATION_PATTERNS = [
    r"i didn[’']?t get that",
    r"can you repeat",
    r"entschuldigung,\s*das habe ich nicht gehört",
    r"kannst du es nochmal sagen",
]
OFFTOPIC_META_PATTERNS = [
    r"let[’']?s talk about something else",
    r"anderes thema",
    r"wir reden am thema vorbei",
    r"off topic",
    r"what can we do in nuremberg",
    r"recommend.*nuremberg",
]
QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "which", "who",
    "was", "warum", "wie", "wann", "wo", "welche", "wer",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_markup(text: str) -> str:
    text = re.sub(r"<<[^>]+>>", " ", text)
    text = re.sub(r"\?0+\d+", " ", text)
    return normalize_whitespace(text)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÄÖÜäöüß]+(?:'[A-Za-z]+)?", text.lower())


def split_sentences(text: str) -> list[str]:
    text = strip_markup(text)
    parts = re.split(r"[.!?]+(?:\s+|$)", text)
    return [p.strip() for p in parts if p.strip()]


def word_count(text: str) -> int:
    return len(tokenize(text))


def sentence_count(text: str) -> int:
    return len(split_sentences(text))


def lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return math.nan
    return len(set(tokens)) / len(tokens)


def contains_pattern(text: str, patterns: list[str]) -> bool:
    text_l = strip_markup(text).lower()
    return any(re.search(p, text_l) for p in patterns)


def count_markers(tokens: list[str], marker_set: set[str]) -> int:
    return sum(1 for t in tokens if t in marker_set)


def normalize_topic(raw_topic: Any) -> str | None:
    if raw_topic is None:
        return None
    topic = str(raw_topic).strip().lower()
    if not topic or topic in {"none", "n/a", "keine auswahl", "general", "abschluss", "waiting_for_choice"}:
        return None
    return TOPIC_MAP.get(topic)


def infer_chosen_topic(dialog: dict[str, Any]) -> str | None:
    topics = dialog.get("topics", []) or []
    cleaned = [normalize_topic(t) for t in topics]
    cleaned = [t for t in cleaned if t is not None]
    if cleaned:
        return cleaned[0]

    for msg in dialog.get("messages", []):
        t = normalize_topic(msg.get("topic"))
        if t is not None:
            return t

    user_text = " ".join(
        strip_markup(m.get("content", "")) for m in dialog.get("messages", []) if m.get("role") == "user"
    ).lower()
    for raw, mapped in TOPIC_MAP.items():
        if re.search(rf"\b{re.escape(raw)}\b", user_text):
            return mapped
    return None


def distinct_topics(dialog: dict[str, Any]) -> list[str]:
    found = []
    for t in dialog.get("topics", []) or []:
        nt = normalize_topic(t)
        if nt is not None:
            found.append(nt)
    for msg in dialog.get("messages", []):
        nt = normalize_topic(msg.get("topic"))
        if nt is not None:
            found.append(nt)
    return sorted(set(found))


def simple_dialog_act(text: str, role: str) -> str:
    clean = strip_markup(text)
    lower = clean.lower().strip()
    tokens = tokenize(clean)

    if not lower:
        return "other"
    if any(g in lower for g in GREETING_MARKERS):
        return "greeting"
    if any(f in lower for f in FAREWELL_MARKERS):
        return "farewell"
    if contains_pattern(clean, CLARIFICATION_PATTERNS):
        return "clarification_request"
    if role == "user":
        if any(re.search(rf"\b{re.escape(t)}\b", lower) for t in ["breakfast", "frühstück", "watches", "uhren", "vacation", "urlaub"]) and len(tokens) <= 4:
            return "topic_selection"
        if any(x in lower for x in REFUSAL_MARKERS):
            return "refusal"
    if "?" in clean or (tokens and tokens[0] in QUESTION_WORDS):
        return "question"
    if lower in ACK_MARKERS:
        return "acknowledgment"
    return "answer"


def extract_conversation_metrics(dialog: dict[str, Any]) -> dict[str, Any]:
    messages = dialog.get("messages", []) or []
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    user_messages = [m for m in messages if m.get("role") == "user"]
    system_messages = [m for m in messages if m.get("role") == "system"]

    assistant_texts = [strip_markup(m.get("content", "")) for m in assistant_messages]
    user_texts = [strip_markup(m.get("content", "")) for m in user_messages]

    assistant_tokens = [tok for text in assistant_texts for tok in tokenize(text)]
    user_tokens = [tok for text in user_texts for tok in tokenize(text)]
    all_tokens = assistant_tokens + user_tokens

    assistant_sentence_counts = [sentence_count(t) for t in assistant_texts]
    user_sentence_counts = [sentence_count(t) for t in user_texts]
    user_word_counts = [word_count(t) for t in user_texts]
    assistant_word_counts = [word_count(t) for t in assistant_texts]

    chosen_topic = infer_chosen_topic(dialog)
    topic_list = distinct_topics(dialog)
    feedback = dialog.get("feedback")
    feedback = float(feedback) if feedback is not None else np.nan

    assistant_acts = [simple_dialog_act(m.get("content", ""), "assistant") for m in assistant_messages]
    user_acts = [simple_dialog_act(m.get("content", ""), "user") for m in user_messages]
    act_counts = Counter(assistant_acts + user_acts)

    assistant_full = " ".join(assistant_texts).lower()
    assistant_repeated_fallback_count = (
        assistant_full.count("entschuldigung, das habe ich nicht gehört")
        + assistant_full.count("can you repeat")
    )
    seen_assistant_messages = Counter(assistant_texts)
    assistant_repetition_count = sum(v - 1 for v in seen_assistant_messages.values() if v > 1)

    natural_close = int(any(a == "farewell" for a in assistant_acts[-2:])) if assistant_acts else 0
    num_exchanges = min(len(user_messages), len(assistant_messages))
    completion_proxy = int(natural_close == 1 or num_exchanges >= 6 or len(user_messages) >= 6)
    early_dropoff = int(len(user_messages) <= 2)

    return {
        "id": dialog.get("id"),
        "timestamp": dialog.get("timestamp"),
        "date": pd.to_datetime(dialog.get("timestamp")).date() if dialog.get("timestamp") else pd.NaT,
        "condition": dialog.get("condition"),
        "feedback": feedback,
        "chosen_topic": chosen_topic if chosen_topic is not None else "Unknown",

        "num_distinct_topics": len(topic_list),
        "multi_topic_conversation": int(len(topic_list) > 1),

        "num_exchanges": num_exchanges,
        "total_messages": len(messages),
        "num_user_messages": len(user_messages),
        "num_assistant_messages": len(assistant_messages),
        "num_system_messages": len(system_messages),
        "turn_balance_user_over_assistant": len(user_messages) / len(assistant_messages) if assistant_messages else np.nan,

        "total_words_conversation": len(all_tokens),
        "total_user_words": len(user_tokens),
        "total_assistant_words": len(assistant_tokens),
        "avg_user_words_per_message": float(np.mean(user_word_counts)) if user_word_counts else np.nan,
        "avg_assistant_words_per_message": float(np.mean(assistant_word_counts)) if assistant_word_counts else np.nan,
        "avg_user_sentences_per_message": float(np.mean(user_sentence_counts)) if user_sentence_counts else np.nan,
        "avg_assistant_sentences_per_message": float(np.mean(assistant_sentence_counts)) if assistant_sentence_counts else np.nan,
        "avg_user_words_per_sentence": len(user_tokens) / sum(user_sentence_counts) if sum(user_sentence_counts) else np.nan,
        "avg_assistant_words_per_sentence": len(assistant_tokens) / sum(assistant_sentence_counts) if sum(assistant_sentence_counts) else np.nan,

        "lexical_diversity_user": lexical_diversity(user_tokens),
        "lexical_diversity_conversation": lexical_diversity(all_tokens),
        "first_person_pronouns_user": count_markers(user_tokens, GERMAN_PRONOUNS_FIRST | ENGLISH_PRONOUNS_FIRST),
        "second_person_pronouns_user": count_markers(user_tokens, GERMAN_PRONOUNS_SECOND | ENGLISH_PRONOUNS_SECOND),
        "assistant_question_rate": float(np.mean(["?" in t for t in assistant_texts])) if assistant_texts else np.nan,
        "user_question_rate": float(np.mean(["?" in t for t in user_texts])) if user_texts else np.nan,
        "assistant_exclamation_rate": float(np.mean(["!" in t for t in assistant_texts])) if assistant_texts else np.nan,
        "user_exclamation_rate": float(np.mean(["!" in t for t in user_texts])) if user_texts else np.nan,
        "politeness_marker_count_conversation": count_markers(all_tokens, POLITENESS_MARKERS),

        "assistant_clarification_count": sum(1 for t in assistant_texts if contains_pattern(t, CLARIFICATION_PATTERNS)),
        "assistant_repair_count": sum(1 for t in assistant_texts if contains_pattern(t, CLARIFICATION_PATTERNS)),
        "assistant_repeated_fallback_count": assistant_repeated_fallback_count,
        "assistant_repetition_count": assistant_repetition_count,

        "user_divergence_marker_count": sum(1 for t in user_texts if contains_pattern(t, OFFTOPIC_META_PATTERNS)),
        "topic_switching_frequency": max(len(topic_list) - 1, 0),

        "dialogact_greeting_count": act_counts.get("greeting", 0),
        "dialogact_farewell_count": act_counts.get("farewell", 0),
        "dialogact_question_count": act_counts.get("question", 0),
        "dialogact_answer_count": act_counts.get("answer", 0),
        "dialogact_clarification_count": act_counts.get("clarification_request", 0),
        "dialogact_refusal_count": act_counts.get("refusal", 0),
        "dialogact_topic_selection_count": act_counts.get("topic_selection", 0),
        "dialogact_acknowledgment_count": act_counts.get("acknowledgment", 0),

        "completion_proxy": completion_proxy,
        "early_dropoff": early_dropoff,
    }


def build_metrics_dataframe(dialogs_path: Path) -> pd.DataFrame:
    with dialogs_path.open("r", encoding="utf-8") as f:
        dialogs = json.load(f)
    rows = [extract_conversation_metrics(d) for d in dialogs if d.get("condition") in VALID_CONDITIONS]
    df = pd.DataFrame(rows)
    return df.sort_values(["timestamp", "id"]).reset_index(drop=True)


def detect_metric_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = {"id", "timestamp", "date", "condition", "chosen_topic"}
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    continuous, binary = [], []
    for col in numeric_cols:
        vals = set(df[col].dropna().unique().tolist())
        if vals.issubset({0, 1}):
            binary.append(col)
        else:
            continuous.append(col)
    return continuous, binary


def run_condition_tests(df: pd.DataFrame) -> pd.DataFrame:
    continuous_cols, binary_cols = detect_metric_types(df)
    g1 = df[df["condition"] == CONDITION_A]
    g2 = df[df["condition"] == CONDITION_B]
    rows = []

    for metric in continuous_cols:
        x = g1[metric].dropna()
        y = g2[metric].dropna()
        if len(x) < 2 or len(y) < 2:
            continue
        welch = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({
            "metric": metric,
            "metric_type": "continuous",
            "condition_a_mean": float(x.mean()),
            "condition_b_mean": float(y.mean()),
            "welch_t_pvalue": float(welch.pvalue),
            "mannwhitney_pvalue": float(mwu.pvalue),
        })

    for metric in binary_cols:
        ct = pd.crosstab(df["condition"], df[metric])
        if ct.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(ct)
            rows.append({
                "metric": metric,
                "metric_type": "binary",
                "condition_a_mean": float(g1[metric].mean()),
                "condition_b_mean": float(g2[metric].mean()),
                "chi2_pvalue": float(p),
            })

    topic_table = pd.crosstab(df["condition"], df["chosen_topic"])
    if topic_table.shape[0] == 2 and topic_table.shape[1] >= 2:
        chi2, p, _, _ = stats.chi2_contingency(topic_table)
        rows.append({
            "metric": "chosen_topic",
            "metric_type": "categorical",
            "chi2_pvalue": float(p),
        })

    return pd.DataFrame(rows)


def main() -> None:
    here = Path(__file__).resolve().parent
    dialogs_path = here.parent / "data" / "data_clean" / "dialogs_full.json"

    df = build_metrics_dataframe(dialogs_path)

    df.to_csv(here / "rule_based_metrics_by_conversation.csv", index=False)
    run_condition_tests(df).to_csv(here / "rule_based_condition_tests.csv", index=False)

    (
        df.groupby(["condition", "chosen_topic"])
        .size()
        .reset_index(name="n_conversations")
        .to_csv(here / "rule_based_topic_distribution_by_condition.csv", index=False)
    )

    print("Saved:")
    print(" - rule_based_metrics_by_conversation.csv")
    print(" - rule_based_condition_tests.csv")
    print(" - rule_based_topic_distribution_by_condition.csv")


if __name__ == "__main__":
    main()