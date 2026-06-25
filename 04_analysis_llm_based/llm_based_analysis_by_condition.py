#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional but recommended for loading .env files
    load_dotenv = None

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional but recommended for cleaner progress reporting
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]

with (REPO_ROOT / "configs" / "paths.json").open("r", encoding="utf-8") as _f:
    PATHS = {k: REPO_ROOT / v for k, v in json.load(_f).items() if not k.startswith("_")}

if load_dotenv is not None:
    load_dotenv(dotenv_path=PATHS["env_file"], override=True)


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

VALID_CHOSEN_TOPICS = {"Breakfast", "Watches", "Vacation"}
SUBSTANTIVE_DISCLOSURE_THRESHOLD = 3
OFFTOPIC_SCORE_MAP = {
    "on_topic": 0.0,
    "weakly_related": 0.5,
    "off_topic": 1.0,
}

DIALOG_ACTS = [
    "greeting",
    "topic_selection",
    "question",
    "answer",
    "clarification_request",
    "acknowledgment",
    "refusal_rejection",
    "self_disclosure",
    "opinion_evaluation",
    "off_topic_move",
    "repair_correction",
    "closing_farewell",
    "other",
]

SYSTEM_PROMPT = """
You are a careful research annotator for museum robot conversations.
You will annotate exactly one conversation.
Important rules:
- Annotate this conversation independently.
- Do not compare robot conditions.
- Treat the model as a structured annotator, not as the final analyst.
- Ignore system reminders about language and ignore markup/noise such as <<smile>>, <<browraise>>, ?001, ?007.
- Annotate every assistant and user message exactly once.
- Preserve message_index and role exactly as given.
- If a field does not apply, return null.

Primary dialog-act taxonomy:
- greeting: opening social hello / welcome
- topic_selection: explicit topic choice or asking to choose among Breakfast / Watches / Vacation
- question: information-seeking question
- answer: direct answer or factual reply without strong disclosure/opinion as primary function
- clarification_request: asks for repetition or clarification, or signals non-understanding
- acknowledgment: short confirmation / backchannel / acceptance
- refusal_rejection: rejects the topic, declines answering, resists continuing, says stop, or pushes back
- self_disclosure: reveals a personal habit, preference, experience, routine, family situation, or feeling
- opinion_evaluation: gives a judgment, stance, or evaluation without disclosure being the main function
- off_topic_move: semantically shifts away from the active topic/question
- repair_correction: repairs misunderstanding, corrects a previous interpretation, or tries to recover the conversation
- closing_farewell: thanks, goodbye, or explicit ending / wrap-up
- other: anything else

Disclosure rubric for USER messages:
0 = no personal information, no interpretable content, or only noise
1 = minimal factual answer with almost no personal content
2 = mild personal preference, routine, or simple habit
3 = meaningful personal opinion, experience, or habit
4 = strong personal narrative or emotionally revealing disclosure

Topic relevance rubric for USER messages, judged relative to the chosen topic and the immediately preceding assistant question:
- on_topic: clearly answers or relates to the chosen topic and current question
- weakly_related: partially related, ambiguous, or meta but still loosely connected
- off_topic: clearly diverges to another topic or another meta-conversation

Assistant response quality rubric for ASSISTANT messages:
1 = very poor, irrelevant, incoherent, or clearly unnatural
2 = weak, partially relevant, generic, or awkward
3 = adequate but limited
4 = good and relevant
5 = excellent, highly relevant, coherent, specific, and natural

Assistant repair quality rubric for ASSISTANT messages that handle confusion, refusal, or off-topic input:
1 = makes the breakdown worse or ignores the problem
2 = weak recovery
3 = acceptable recovery
4 = good recovery
5 = excellent recovery

Conversation-level ratings:
- engagement_score: 1 = minimal / forced engagement, 5 = genuinely engaged
- frustration_score: 1 = no clear frustration, 5 = strong frustration / annoyance / impatience
- naturalness_score: 1 = very unnatural interaction, 5 = very natural interaction
""".strip()


DialogAct = Literal[
    "greeting",
    "topic_selection",
    "question",
    "answer",
    "clarification_request",
    "acknowledgment",
    "refusal_rejection",
    "self_disclosure",
    "opinion_evaluation",
    "off_topic_move",
    "repair_correction",
    "closing_farewell",
    "other",
]
TopicRelevance = Literal["on_topic", "weakly_related", "off_topic"]
Role = Literal["assistant", "user"]


class MessageAnnotation(BaseModel):
    message_index: int = Field(ge=0)
    role: Role
    primary_dialog_act: DialogAct
    disclosure_level: int | None = Field(default=None, ge=0, le=4)
    topic_relevance: TopicRelevance | None = None
    response_quality: int | None = Field(default=None, ge=1, le=5)
    repair_quality: int | None = Field(default=None, ge=1, le=5)


class ConversationAnnotation(BaseModel):
    engagement_score: int = Field(ge=1, le=5)
    frustration_score: int = Field(ge=1, le=5)
    naturalness_score: int = Field(ge=1, le=5)
    message_annotations: list[MessageAnnotation]


def normalize_whitespace(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text).strip()


def strip_markup(text: str) -> str:
    import re

    text = re.sub(r"<<[^>]+>>", " ", text)
    text = re.sub(r"\?0+\d+", " ", text)
    return normalize_whitespace(text)


def normalize_topic(raw_topic: Any) -> str | None:
    if raw_topic is None:
        return None
    topic = str(raw_topic).strip().lower()
    if not topic or topic in {"none", "n/a", "keine auswahl", "general", "abschluss", "waiting_for_choice"}:
        return None
    return TOPIC_MAP.get(topic)


def infer_chosen_topic(dialog: dict[str, Any]) -> str | None:
    import re

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


def prepare_turns(dialog: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for msg in dialog.get("messages", []) or []:
        role = msg.get("role")
        if role not in {"assistant", "user"}:
            continue
        content = strip_markup(msg.get("content", ""))
        if not content:
            continue
        turns.append({
            "message_index": len(turns),
            "role": role,
            "content": content,
        })
    return turns


def load_dialogs(dialogs_path: Path) -> list[dict[str, Any]]:
    with dialogs_path.open("r", encoding="utf-8") as f:
        dialogs = json.load(f)
    return [d for d in dialogs if d.get("condition") in VALID_CONDITIONS]


def load_cache(cache_path: Path) -> dict[int, dict[str, Any]]:
    if not cache_path.exists():
        return {}

    cache: dict[int, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[int(row["id"])] = row["annotation"]
    return cache


def append_cache(cache_path: Path, dialog_id: int, annotation: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": dialog_id, "annotation": annotation}, ensure_ascii=False) + "\n")


def validate_alignment(annotation: ConversationAnnotation, turns: list[dict[str, Any]]) -> None:
    expected = [(t["message_index"], t["role"]) for t in turns]
    received = sorted((m.message_index, m.role) for m in annotation.message_annotations)
    if received != expected:
        raise ValueError(
            "Structured annotation does not align with the provided turns. "
            f"Expected {expected}, received {received}."
        )


def call_openai_annotation(
    client: Any,
    dialog: dict[str, Any],
    turns: list[dict[str, Any]],
    *,
    model: str,
    max_retries: int,
    retry_base_seconds: float,
) -> dict[str, Any]:
    chosen_topic = infer_chosen_topic(dialog) or "Unknown"
    payload = {
        "dialog_id": dialog.get("id"),
        "condition": dialog.get("condition"),
        "topics_field": dialog.get("topics", []),
        "chosen_topic_rule_based": chosen_topic,
        "messages": turns,
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.parse(
                model=model,
                store=False,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                text_format=ConversationAnnotation,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("No parsed structured output returned by the API.")
            validate_alignment(parsed, turns)
            return parsed.model_dump()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            sleep_seconds = retry_base_seconds * (2 ** (attempt - 1))
            message = (
                f"Dialog {dialog.get('id')}: annotation attempt {attempt} failed "
                f"({type(exc).__name__}: {exc}). Retrying in {sleep_seconds:.1f}s..."
            )
            if tqdm is not None:
                tqdm.write(message)
            else:
                print(message)
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


def collect_annotations(
    dialogs: list[dict[str, Any]],
    *,
    cache_path: Path,
    model: str,
    max_retries: int,
    retry_base_seconds: float,
    request_pause_seconds: float,
    overwrite_cache: bool,
    max_dialogs: int | None,
) -> dict[int, dict[str, Any]]:
    cache = {} if overwrite_cache else load_cache(cache_path)
    selected_dialogs = dialogs[:max_dialogs] if max_dialogs is not None else dialogs

    client = None
    if any(int(dialog.get("id")) not in cache for dialog in selected_dialogs if dialog.get("id") is not None):
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to a .env file or export it in your shell before running the script."
            )

        client = OpenAI()

    iterator = selected_dialogs
    progress = None
    if tqdm is not None:
        progress = tqdm(selected_dialogs, total=len(selected_dialogs), desc="Annotating dialogs", unit="dialog")
        iterator = progress

    for dialog in iterator:
        dialog_id = int(dialog.get("id"))
        if dialog_id in cache:
            continue

        turns = prepare_turns(dialog)
        if not turns:
            continue

        assert client is not None
        annotation = call_openai_annotation(
            client,
            dialog,
            turns,
            model=model,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
        )
        cache[dialog_id] = annotation
        append_cache(cache_path, dialog_id, annotation)

        if progress is not None:
            progress.set_postfix_str(f"last_id={dialog_id}", refresh=False)

        if request_pause_seconds > 0:
            time.sleep(request_pause_seconds)

    if progress is not None:
        progress.close()

    return cache


def aggregate_conversation_metrics(dialog: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    turns = prepare_turns(dialog)
    turn_lookup = {t["message_index"]: t for t in turns}
    message_annotations = annotation.get("message_annotations", [])

    act_counter = Counter(a["primary_dialog_act"] for a in message_annotations)

    user_annotations = [a for a in message_annotations if a["role"] == "user"]
    assistant_annotations = [a for a in message_annotations if a["role"] == "assistant"]

    disclosure_values = [a["disclosure_level"] for a in user_annotations if a.get("disclosure_level") is not None]
    topic_relevance_values = [a["topic_relevance"] for a in user_annotations if a.get("topic_relevance") is not None]
    response_quality_values = [a["response_quality"] for a in assistant_annotations if a.get("response_quality") is not None]
    repair_quality_values = [a["repair_quality"] for a in assistant_annotations if a.get("repair_quality") is not None]

    chosen_topic = infer_chosen_topic(dialog) or "Unknown"
    feedback = dialog.get("feedback")
    feedback = float(feedback) if feedback is not None else np.nan

    user_turns = [t for t in turns if t["role"] == "user"]
    assistant_turns = [t for t in turns if t["role"] == "assistant"]

    off_topic_count = sum(1 for x in topic_relevance_values if x == "off_topic")
    weakly_related_count = sum(1 for x in topic_relevance_values if x == "weakly_related")

    return {
        "id": dialog.get("id"),
        "timestamp": dialog.get("timestamp"),
        "date": pd.to_datetime(dialog.get("timestamp")).date() if dialog.get("timestamp") else pd.NaT,
        "condition": dialog.get("condition"),
        "feedback": feedback,
        "chosen_topic": chosen_topic,
        "n_annotated_turns": len(message_annotations),
        "n_user_turns": len(user_turns),
        "n_assistant_turns": len(assistant_turns),
        "llm_engagement_score": annotation.get("engagement_score"),
        "llm_frustration_score": annotation.get("frustration_score"),
        "llm_naturalness_score": annotation.get("naturalness_score"),
        "llm_mean_disclosure": float(np.mean(disclosure_values)) if disclosure_values else np.nan,
        "llm_max_disclosure": float(np.max(disclosure_values)) if disclosure_values else np.nan,
        "llm_any_substantive_disclosure": int(any(x >= SUBSTANTIVE_DISCLOSURE_THRESHOLD for x in disclosure_values)),
        "llm_user_disclosure_turn_count": int(sum(1 for x in disclosure_values if x >= 2)),
        "llm_mean_offtopic_score": float(np.mean([OFFTOPIC_SCORE_MAP[x] for x in topic_relevance_values])) if topic_relevance_values else np.nan,
        "llm_offtopic_user_turn_proportion": off_topic_count / len(topic_relevance_values) if topic_relevance_values else np.nan,
        "llm_weakly_related_user_turn_proportion": weakly_related_count / len(topic_relevance_values) if topic_relevance_values else np.nan,
        "llm_any_offtopic_divergence": int(off_topic_count > 0),
        "llm_assistant_response_quality_mean": float(np.mean(response_quality_values)) if response_quality_values else np.nan,
        "llm_assistant_response_quality_min": float(np.min(response_quality_values)) if response_quality_values else np.nan,
        "llm_assistant_repair_quality_mean": float(np.mean(repair_quality_values)) if repair_quality_values else np.nan,
        "llm_assistant_repair_turn_count": int(len(repair_quality_values)),
        "llm_user_refusal_count": int(sum(1 for a in user_annotations if a["primary_dialog_act"] == "refusal_rejection")),
        "llm_user_offtopic_move_count": int(sum(1 for a in user_annotations if a["primary_dialog_act"] == "off_topic_move")),
        "llm_assistant_clarification_count": int(sum(1 for a in assistant_annotations if a["primary_dialog_act"] == "clarification_request")),
        "llm_assistant_repair_count": int(sum(1 for a in assistant_annotations if a["primary_dialog_act"] == "repair_correction")),
        "llm_closing_count": act_counter.get("closing_farewell", 0),
        **{f"llm_dialogact_{act}_count": act_counter.get(act, 0) for act in DIALOG_ACTS},
    }


def build_turn_annotations_dataframe(
    dialogs_path: Path,
    *,
    cache_path: Path,
    model: str,
    max_retries: int,
    retry_base_seconds: float,
    request_pause_seconds: float,
    overwrite_cache: bool = False,
    max_dialogs: int | None = None,
) -> pd.DataFrame:
    dialogs = load_dialogs(dialogs_path)
    annotations = collect_annotations(
        dialogs,
        cache_path=cache_path,
        model=model,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        request_pause_seconds=request_pause_seconds,
        overwrite_cache=overwrite_cache,
        max_dialogs=max_dialogs,
    )

    rows: list[dict[str, Any]] = []
    processed = 0
    for dialog in dialogs:
        if max_dialogs is not None and processed >= max_dialogs:
            break
        processed += 1

        dialog_id = int(dialog.get("id"))
        if dialog_id not in annotations:
            continue

        turn_lookup = {t["message_index"]: t for t in prepare_turns(dialog)}
        chosen_topic = infer_chosen_topic(dialog) or "Unknown"
        for ann in annotations[dialog_id].get("message_annotations", []):
            content = turn_lookup.get(ann["message_index"], {}).get("content", "")
            topic_relevance = ann.get("topic_relevance")
            rows.append({
                "id": dialog_id,
                "timestamp": dialog.get("timestamp"),
                "condition": dialog.get("condition"),
                "chosen_topic": chosen_topic,
                "message_index": ann["message_index"],
                "role": ann["role"],
                "content": content,
                "primary_dialog_act": ann["primary_dialog_act"],
                "disclosure_level": ann.get("disclosure_level"),
                "topic_relevance": topic_relevance,
                "offtopic_score": OFFTOPIC_SCORE_MAP.get(topic_relevance, np.nan) if topic_relevance is not None else np.nan,
                "response_quality": ann.get("response_quality"),
                "repair_quality": ann.get("repair_quality"),
            })

    return pd.DataFrame(rows).sort_values(["timestamp", "id", "message_index"]).reset_index(drop=True)


def build_metrics_dataframe(
    dialogs_path: Path,
    *,
    cache_path: Path,
    model: str,
    max_retries: int,
    retry_base_seconds: float,
    request_pause_seconds: float,
    overwrite_cache: bool = False,
    max_dialogs: int | None = None,
) -> pd.DataFrame:
    dialogs = load_dialogs(dialogs_path)
    annotations = collect_annotations(
        dialogs,
        cache_path=cache_path,
        model=model,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        request_pause_seconds=request_pause_seconds,
        overwrite_cache=overwrite_cache,
        max_dialogs=max_dialogs,
    )

    rows: list[dict[str, Any]] = []
    processed = 0
    for dialog in dialogs:
        if max_dialogs is not None and processed >= max_dialogs:
            break
        processed += 1

        dialog_id = int(dialog.get("id"))
        if dialog_id not in annotations:
            continue
        rows.append(aggregate_conversation_metrics(dialog, annotations[dialog_id]))

    return pd.DataFrame(rows).sort_values(["timestamp", "id"]).reset_index(drop=True)


def detect_metric_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = {"id", "timestamp", "date", "condition", "feedback", "chosen_topic"}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based conversation annotation by condition.")
    parser.add_argument(
        "--dialogs",
        type=Path,
        default=PATHS["dialogs_full"],
        help="Path to dialogs_full.json",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        help="OpenAI model ID to use for annotation.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PATHS["llm_annotation_cache"],
        help="JSONL cache for raw structured LLM annotations.",
    )
    parser.add_argument("--max-dialogs", type=int, default=None, help="Optional cap for test runs.")
    parser.add_argument("--max-retries", type=int, default=5, help="API retries per dialog.")
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=2.0,
        help="Base seconds for exponential backoff.",
    )
    parser.add_argument(
        "--request-pause-seconds",
        type=float,
        default=0.0,
        help="Optional pause after each successful request to avoid bursts.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Ignore any existing cache and annotate again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    conversation_df = build_metrics_dataframe(
        args.dialogs,
        cache_path=args.cache,
        model=args.model,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        request_pause_seconds=args.request_pause_seconds,
        overwrite_cache=args.overwrite_cache,
        max_dialogs=args.max_dialogs,
    )

    turn_df = build_turn_annotations_dataframe(
        args.dialogs,
        cache_path=args.cache,
        model=args.model,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        request_pause_seconds=args.request_pause_seconds,
        overwrite_cache=False,
        max_dialogs=args.max_dialogs,
    )

    conversation_df.to_csv(PATHS["llm_metrics_by_conversation"], index=False)
    turn_df.to_csv(PATHS["llm_turn_annotations"], index=False)
    run_condition_tests(conversation_df).to_csv(PATHS["llm_condition_tests"], index=False)

    (
        conversation_df.groupby(["condition", "chosen_topic"])
        .size()
        .reset_index(name="n_conversations")
        .to_csv(PATHS["llm_topic_distribution_by_condition"], index=False)
    )

    print("Saved:")
    print(" - llm_metrics_by_conversation.csv")
    print(" - llm_turn_annotations.csv")
    print(" - llm_condition_tests.csv")
    print(" - llm_topic_distribution_by_condition.csv")
    print(f" - raw cache: {args.cache}")


if __name__ == "__main__":
    main()
