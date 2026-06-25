from __future__ import annotations

import hashlib
import html
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

# ============================================================
# Configuration
# ============================================================

# This file should be located inside: 05_prolific/
# Dialog input CSV is produced by create_test_samples.py into this same
# folder, named "streamlit_test_sample_en_<N>.csv" (N = however many
# conversations were requested). The most recently generated file is used
# automatically. Only English dialogs are loaded by this app.
BASE_DIR = Path(__file__).resolve().parent

DIALOG_INPUT_FILENAME_PATTERN = "streamlit_test_sample_en_*.csv"

RESPONSES_DIR = BASE_DIR / "responses"
DB_PATH = RESPONSES_DIR / "survey_responses_en.sqlite"
CSV_EXPORT_PATH = RESPONSES_DIR / "survey_responses_en.csv"

# Prolific completion URL.
PROLIFIC_COMPLETION_URL = "https://app.prolific.com/submissions/complete?cc=C6JV4KGN"

# Number of dialogs one participant should annotate.
DIALOGS_PER_PARTICIPANT = 3

# Optional quota per dialog. Set to None if you do not want a cap.
TARGET_RATINGS_PER_DIALOG: Optional[int] = None

# Picture shown between Block A and Block B questions, chosen by the dialog's condition.
# TODO (dummy placeholder): replace with the real image files for Block B.
ASSETS_DIR = BASE_DIR.parent / "data" / "assets" / "robot_images"
BLOCK_B_IMAGE_PATHS_BY_CONDITION = {
    "Condition A (Willi)": ASSETS_DIR / "block_b_image_condition_a_willi.png",
    "Condition B (WV-34)": ASSETS_DIR / "block_b_image_condition_b_wv34.png",
}

REQUIRED_COLUMNS = ["META_dialog_id", "META_condition", "language", "subject", "dialog_text"]

ANNOTATION_INSTRUCTION_BLOCK_A = (
    "Please rate the observable quality of the human-robot conversation based only on the transcript. "
    "Do not rate whether you personally like the robot, whether you would enjoy using it, or how you "
    "imagine the robot looks. Focus on how well the interaction works as a conversation: whether the "
    "robot responds appropriately, whether the conversation flows, whether the human participant appears "
    "engaged, and whether misunderstandings or social difficulties are handled well."
)

QUESTIONS_BLOCK_A = [
    {
        "key": "user_engagement_enjoyment",
        "label": "1. User engagement / enjoyment",
        "help": "How engaged and willing to continue did the human participant appear during the conversation?",
        "definition": (
            "User engagement / enjoyment refers to how involved, interested, and willing to "
            "continue the human participant appears to be during the conversation. This should be "
            "judged only from observable cues in the transcript, such as the participant's answers, "
            "reactions, questions, cooperation, reluctance, or attempts to end the exchange."
        ),
        "rate_higher": (
            "The human participant gives meaningful or detailed answers, reacts to the robot, "
            "asks questions, plays along with the conversation, continues the exchange voluntarily, "
            "or appears interested in the topic."
        ),
        "rate_lower": (
            "The human participant gives minimal answers, appears passive, ignores the robot's "
            "questions, resists the topic, shows boredom or irritation, or tries to end the conversation."
        ),
        "anchors": {
            1: "Very low engagement/enjoyment: the human participant barely participates, gives minimal or resistant answers, or clearly wants to stop.",
            3: "Moderate engagement/enjoyment: the human participant answers the robot but with limited detail, mixed interest, or occasional signs of disengagement.",
            5: "Very high engagement/enjoyment: the human participant actively participates, gives meaningful answers, reacts to the robot, and appears willing to continue.",
        },
        "likert_labels": {
            1: "1 — Very low engagement/enjoyment",
            2: "2 — Low engagement/enjoyment",
            3: "3 — Moderate engagement/enjoyment",
            4: "4 — High engagement/enjoyment",
            5: "5 — Very high engagement/enjoyment",
        },
    },
    {
        "key": "conversation_flow_coherence",
        "label": "2. Conversation flow / coherence",
        "help": "How smoothly and coherently did the conversation progress across turns?",
        "definition": (
            "Conversation flow / coherence refers to whether the conversation progresses in a "
            "logical, understandable, and smooth way across turns. It captures whether the robot "
            "and the human participant remain connected to each other, whether the topic develops "
            "naturally, and whether transitions or follow-up questions make sense."
        ),
        "rate_higher": (
            "The conversation follows a clear and logical sequence, the robot's follow-up questions "
            "connect to the human participant's previous answers, the topic develops naturally, and "
            "the exchange is easy to follow."
        ),
        "rate_lower": (
            "The conversation feels broken, repetitive, abrupt, confusing, or difficult to follow. "
            "The robot may jump between topics, repeat itself unnecessarily, ignore prior context, "
            "or create conversational dead ends."
        ),
        "anchors": {
            1: "Very poor flow/coherence: the conversation is hard to follow, repetitive, disconnected, or logically broken.",
            3: "Moderate flow/coherence: the general topic is understandable, but there are noticeable awkward transitions, repetitions, or weak connections between turns.",
            5: "Very good flow/coherence: the conversation progresses smoothly, stays logically connected, and each turn follows naturally from the previous one.",
        },
        "likert_labels": {
            1: "1 — Very poor flow/coherence",
            2: "2 — Poor flow/coherence",
            3: "3 — Moderate flow/coherence",
            4: "4 — Good flow/coherence",
            5: "5 — Very good flow/coherence",
        },
    },
    {
        "key": "interaction_clarity_habitability",
        "label": "3. Interaction clarity / habitability",
        "help": "How clear was it what the human participant could say or do next?",
        "definition": (
            "Interaction clarity / habitability refers to whether the robot makes the conversational "
            "task understandable for the human participant. It captures whether the available topics, "
            "expected answers, next steps, and interaction rules are clear enough for the participant "
            "to know how to continue."
        ),
        "rate_higher": (
            "The robot gives clear prompts, makes the available choices or next steps understandable, "
            "asks questions that are easy to answer, and helps the human participant understand how "
            "to continue the conversation."
        ),
        "rate_lower": (
            "The robot gives unclear, ambiguous, overly broad, or confusing prompts. The human "
            "participant appears unsure what to say, what the robot expects, or how to continue."
        ),
        "anchors": {
            1: "Very poor clarity/habitability: it is unclear what the human participant is expected to say or do, and the interaction format is confusing.",
            3: "Moderate clarity/habitability: the interaction is partly understandable, but some prompts, choices, or next steps are unclear.",
            5: "Very good clarity/habitability: the robot clearly guides the conversation and makes it easy for the human participant to know how to respond.",
        },
        "likert_labels": {
            1: "1 — Very poor clarity",
            2: "2 — Poor clarity",
            3: "3 — Moderate clarity",
            4: "4 — Good clarity",
            5: "5 — Very good clarity",
        },
    },
    {
        "key": "repair_recovery_quality",
        "label": "4. Repair / recovery quality",
        "help": "When misunderstandings or interaction problems occurred, how well did the robot recover?",
        "definition": (
            "Repair / recovery quality refers to how well the robot handles misunderstandings, "
            "unclear user input, off-topic answers, resistance, or other interaction problems. "
            "It captures whether the robot notices the problem, responds appropriately, and helps "
            "move the conversation forward without making the situation worse."
        ),
        "note": (
            "Use N/A if there is no visible misunderstanding, confusion, off-topic input, resistance, "
            "or other repair situation in the transcript."
        ),
        "rate_higher": (
            "The robot acknowledges the problem, asks for clarification when needed, adapts to the "
            "human participant's correction or resistance, repairs the topic smoothly, and continues "
            "the conversation in an appropriate way."
        ),
        "rate_lower": (
            "The robot ignores the problem, repeats the same failed prompt, misunderstands the "
            "human participant again, continues as if nothing happened, or makes the interaction "
            "more awkward or frustrating."
        ),
        "anchors": {
            1: "Very poor repair/recovery: the robot fails to handle the problem and the conversation becomes more confusing, repetitive, or frustrating.",
            3: "Moderate repair/recovery: the robot makes some attempt to recover, but the repair is only partly successful or remains awkward.",
            5: "Very good repair/recovery: the robot handles the problem appropriately, clarifies or adapts when needed, and moves the conversation forward smoothly.",
        },
        "likert_labels": {
            1: "1 — Very poor repair/recovery",
            2: "2 — Poor repair/recovery",
            3: "3 — Moderate repair/recovery",
            4: "4 — Good repair/recovery",
            5: "5 — Very good repair/recovery",
            "NA": "N/A — No repair situation occurred",
        },
    },
    {
        "key": "response_appropriateness",
        "label": "5. Response appropriateness",
        "help": "How appropriate were the robot's responses to the human participant's previous turns?",
        "definition": (
            "Response appropriateness refers to how well the robot's replies fit the human "
            "participant's immediately preceding turns. It captures whether the robot understands "
            "the user's apparent meaning, acknowledges relevant information, asks suitable follow-up "
            "questions, and avoids irrelevant or mismatched responses."
        ),
        "rate_higher": (
            "The robot's responses are relevant to what the human participant just said, acknowledge "
            "important information, ask suitable follow-up questions, and fit the participant's intent, "
            "tone, or level of cooperation."
        ),
        "rate_lower": (
            "The robot's responses are irrelevant, generic, mismatched, overly scripted, or based on "
            "a misunderstanding of what the human participant said. The robot may ignore important "
            "information or respond as if the participant had said something else."
        ),
        "anchors": {
            1: "Very poor appropriateness: the robot's responses are mostly irrelevant, mismatched, or inappropriate to the human participant's turns.",
            3: "Moderate appropriateness: the robot responds appropriately in some places but also misses, ignores, or misinterprets important parts of the participant's input.",
            5: "Very good appropriateness: the robot consistently gives relevant, fitting, and context-sensitive responses to the human participant's turns.",
        },
        "likert_labels": {
            1: "1 — Very poor appropriateness",
            2: "2 — Poor appropriateness",
            3: "3 — Moderate appropriateness",
            4: "4 — Good appropriateness",
            5: "5 — Very good appropriateness",
        },
    },
    {
        "key": "social_interaction_quality",
        "label": "6. Social interaction quality",
        "help": "How socially appropriate was the robot as a conversational partner?",
        "definition": (
            "Social interaction quality refers to how well the robot behaves as a socially appropriate "
            "conversational partner in the transcript. It captures whether the robot's tone, wording, "
            "acknowledgments, humor, politeness, and handling of the participant's reactions are "
            "suitable for the interaction."
        ),
        "rate_higher": (
            "The robot responds politely and appropriately, acknowledges the human participant, "
            "uses a suitable tone, handles reluctance or frustration respectfully, and behaves like "
            "a socially appropriate conversational partner."
        ),
        "rate_lower": (
            "The robot feels socially awkward, insensitive, overly pushy, dismissive, repetitive, "
            "too cheerful despite user frustration, or otherwise inappropriate for the participant's "
            "responses."
        ),
        "anchors": {
            1: "Very poor social interaction quality: the robot behaves in a socially inappropriate, insensitive, awkward, or pushy way.",
            3: "Moderate social interaction quality: the robot is generally acceptable, but some responses feel socially awkward, poorly timed, or insufficiently adapted to the participant.",
            5: "Very good social interaction quality: the robot behaves politely, appropriately, and adaptively as a conversational partner.",
        },
        "likert_labels": {
            1: "1 — Very poor social interaction quality",
            2: "2 — Poor social interaction quality",
            3: "3 — Moderate social interaction quality",
            4: "4 — Good social interaction quality",
            5: "5 — Very good social interaction quality",
        },
    },
]


ANNOTATION_INSTRUCTION_BLOCK_B = (
    "Please look at the robot image shown above and rate how the robot appears as an embodied "
    "social agent. These questions are about the robot's perceived appearance and persona, not "
    "about how well the specific transcript worked as a conversation. Use only the information "
    "available in this survey page."
)

QUESTIONS_BLOCK_B = [
    {
        "key": "robot_anthropomorphism",
        "label": "7. Robot anthropomorphism / human-likeness",
        "help": "How human-like did the robot appear?",
        "definition": (
            "Robot anthropomorphism / human-likeness refers to the extent to which the robot appears "
            "human-like rather than machine-like. This includes whether the robot gives the impression "
            "of a human-like social agent, rather than a purely mechanical device."
        ),
        "rate_higher": (
            "The robot appears more human-like, natural, socially expressive, or person-like."
        ),
        "rate_lower": (
            "The robot appears more machine-like, mechanical, artificial, or device-like."
        ),
        "anchors": {
            1: "Very machine-like: the robot appears purely mechanical, artificial, or device-like.",
            3: "Moderately human-like: the robot has some human-like or social qualities, but still appears clearly robotic.",
            5: "Very human-like: the robot appears strongly human-like, natural, or person-like as a social agent.",
        },
        "likert_labels": {
            1: "1 — Very machine-like",
            2: "2 — Mostly machine-like",
            3: "3 — Moderately human-like",
            4: "4 — Mostly human-like",
            5: "5 — Very human-like",
        },
    },
    {
        "key": "robot_animacy",
        "label": "8. Robot animacy / lifelikeness",
        "help": "How lifelike or animated did the robot appear?",
        "definition": (
            "Robot animacy / lifelikeness refers to the extent to which the robot appears alive, "
            "animated, lively, or capable of social presence, rather than inert or lifeless."
        ),
        "rate_higher": (
            "The robot appears more lifelike, lively, expressive, animated, or socially present."
        ),
        "rate_lower": (
            "The robot appears more lifeless, inert, static, mechanical, or lacking in social presence."
        ),
        "anchors": {
            1: "Very low animacy: the robot appears lifeless, inert, static, or purely mechanical.",
            3: "Moderate animacy: the robot appears somewhat lively or expressive, but still clearly artificial.",
            5: "Very high animacy: the robot appears lively, lifelike, animated, or socially present.",
        },
        "likert_labels": {
            1: "1 — Very lifeless/static",
            2: "2 — Mostly lifeless/static",
            3: "3 — Moderately lifelike",
            4: "4 — Mostly lifelike",
            5: "5 — Very lifelike/animated",
        },
    },
    {
        "key": "robot_likeability",
        "label": "9. Robot likeability / pleasantness",
        "help": "How likeable or pleasant did the robot appear?",
        "definition": (
            "Robot likeability / pleasantness refers to the extent to which the robot appears pleasant, "
            "friendly, nice, and approachable as an interaction partner."
        ),
        "rate_higher": (
            "The robot appears friendly, pleasant, nice, approachable, and generally likeable."
        ),
        "rate_lower": (
            "The robot appears unfriendly, unpleasant, unapproachable, awkward, or unlikeable."
        ),
        "anchors": {
            1: "Very unlikeable: the robot appears unpleasant, unfriendly, or unapproachable.",
            3: "Moderately likeable: the robot appears acceptable or neutral, but not especially friendly or pleasant.",
            5: "Very likeable: the robot appears friendly, pleasant, nice, and approachable.",
        },
        "likert_labels": {
            1: "1 — Very unlikeable/unpleasant",
            2: "2 — Mostly unlikeable/unpleasant",
            3: "3 — Moderately likeable",
            4: "4 — Mostly likeable/pleasant",
            5: "5 — Very likeable/pleasant",
        },
    },
    {
        "key": "robot_perceived_intelligence",
        "label": "10. Robot perceived intelligence / competence",
        "help": "How intelligent or competent did the robot appear?",
        "definition": (
            "Robot perceived intelligence / competence refers to the extent to which the robot appears "
            "capable, sensible, competent, and intelligent as a conversational social agent."
        ),
        "rate_higher": (
            "The robot appears competent, sensible, intelligent, knowledgeable, or capable as an interaction partner."
        ),
        "rate_lower": (
            "The robot appears incompetent, unintelligent, foolish, limited, or incapable as an interaction partner."
        ),
        "anchors": {
            1: "Very low perceived intelligence: the robot appears incompetent, unintelligent, foolish, or incapable.",
            3: "Moderate perceived intelligence: the robot appears somewhat competent, but with limited signs of intelligence or capability.",
            5: "Very high perceived intelligence: the robot appears competent, sensible, intelligent, and capable as an interaction partner.",
        },
        "likert_labels": {
            1: "1 — Very incompetent/unintelligent",
            2: "2 — Mostly incompetent/unintelligent",
            3: "3 — Moderately competent/intelligent",
            4: "4 — Mostly competent/intelligent",
            5: "5 — Very competent/intelligent",
        },
    },
]

QUESTIONS = QUESTIONS_BLOCK_A + QUESTIONS_BLOCK_B


# ============================================================
# Basic setup
# ============================================================

st.set_page_config(
    page_title="Observer-rated conversational HRI quality survey",
    page_icon="📝",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .qd-dialog-box {
        border: 1px solid #d9dee7;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        max-height: 560px;
        overflow-y: auto;
        margin-bottom: 1.5rem;
        color: #111827 !important;
    }
    .qd-turn {
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin: 0.75rem 0;
        line-height: 1.55;
        white-space: pre-wrap;
        color: #111827 !important;
        font-size: 1rem;
    }
    .qd-turn-robot {
        background: #eef4ff;
        border-left: 5px solid #3b6fd8;
    }
    .qd-turn-user {
        background: #f5f5f5;
        border-left: 5px solid #808891;
    }
    .qd-turn-context {
        background: #ffffff;
        border-left: 5px solid #c4c9d1;
    }
    .qd-role-label {
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #111827 !important;
    }
    .qd-turn-text {
        color: #111827 !important;
    }
    .meta-line {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }
    .continue-button {
        display: inline-block;
        background: #ff4b4b;
        color: #ffffff !important;
        padding: 0.65rem 1rem;
        border-radius: 0.5rem;
        text-decoration: none !important;
        font-weight: 600;
        margin-top: 0.75rem;
    }
    .continue-button:hover {
        background: #e04343;
        color: #ffffff !important;
        text-decoration: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helper functions
# ============================================================

def get_query_param(key: str, default: str = "") -> str:
    """Read query parameters in a way that works across Streamlit versions."""
    try:
        value = st.query_params.get(key, default)  # Streamlit >= 1.30 style
    except Exception:
        params = st.experimental_get_query_params()  # older Streamlit style
        value = params.get(key, [default])

    if isinstance(value, list):
        return str(value[0]) if value else default
    if value is None:
        return default
    return str(value)


def find_latest_dialog_csv() -> Optional[Path]:
    """Find the most recently generated create_test_samples.py output for English."""
    matches = sorted(BASE_DIR.glob(DIALOG_INPUT_FILENAME_PATTERN), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


@st.cache_data(show_spinner=False)
def load_dialogs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = df.copy()
    df["META_dialog_id"] = df["META_dialog_id"].astype(str)
    df["dialog_text"] = df["dialog_text"].fillna("").astype(str)
    df["language"] = df["language"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)
    return df


def init_db() -> None:
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    question_columns = ",\n".join([f"{q['key']} INTEGER NOT NULL" for q in QUESTIONS])

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS responses (
                participant_id TEXT NOT NULL,
                study_id TEXT,
                session_id TEXT,
                dialog_id TEXT NOT NULL,
                language TEXT,
                subject TEXT,
                condition TEXT,
                submitted_at_utc TEXT NOT NULL,
                free_comment TEXT,
                {question_columns},
                PRIMARY KEY (participant_id, dialog_id)
            )
            """
        )

        # Migrations for databases created before these columns were added.
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(responses)").fetchall()
        }
        if "free_comment" not in existing_columns:
            conn.execute("ALTER TABLE responses ADD COLUMN free_comment TEXT")
        if "condition" not in existing_columns:
            conn.execute("ALTER TABLE responses ADD COLUMN condition TEXT")

        conn.commit()


def export_responses_to_csv() -> None:
    ordered_columns = [
        "participant_id",
        "study_id",
        "session_id",
        "dialog_id",
        "language",
        "subject",
        "condition",
        "submitted_at_utc",
        "free_comment",
    ] + [q["key"] for q in QUESTIONS]

    column_sql = ", ".join(ordered_columns)
    with sqlite3.connect(DB_PATH) as conn:
        responses = pd.read_sql_query(
            f"SELECT {column_sql} FROM responses ORDER BY submitted_at_utc, participant_id, dialog_id",
            conn,
        )
    responses.to_csv(CSV_EXPORT_PATH, index=False, encoding="utf-8-sig")


def participant_completed_count(participant_id: str) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM responses WHERE participant_id = ?",
            (participant_id,),
        ).fetchone()
    return int(row[0]) if row else 0


def participant_answered_dialog_ids(participant_id: str) -> set[str]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT dialog_id FROM responses WHERE participant_id = ?",
            (participant_id,),
        ).fetchall()
    return {str(row[0]) for row in rows}


def stable_tie_breaker(participant_id: str, dialog_id: str) -> int:
    text = f"{participant_id}::{dialog_id}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def assign_dialog(df: pd.DataFrame, participant_id: str, requested_dialog_id: str = "") -> Optional[pd.Series]:
    """
    Assign one dialog to the participant.

    Priority:
    1. If ?DIALOG_ID=... is present and not already rated by this participant, use that dialog.
    2. Otherwise, choose among dialogs not yet rated by this participant.
       The assignment is balanced by current rating count per dialog.
    """
    answered_ids = participant_answered_dialog_ids(participant_id)

    requested_dialog_id = str(requested_dialog_id).strip()
    if requested_dialog_id and requested_dialog_id not in answered_ids:
        chosen = df[df["META_dialog_id"] == requested_dialog_id]
        if not chosen.empty:
            return chosen.iloc[0]

    with sqlite3.connect(DB_PATH) as conn:
        counts = pd.read_sql_query(
            "SELECT dialog_id, COUNT(*) AS n_ratings FROM responses GROUP BY dialog_id",
            conn,
        )

    count_map = (
        dict(zip(counts["dialog_id"].astype(str), counts["n_ratings"].astype(int)))
        if not counts.empty
        else {}
    )

    candidates = df[~df["META_dialog_id"].isin(answered_ids)].copy()
    if candidates.empty:
        return None

    candidates["n_ratings"] = candidates["META_dialog_id"].map(count_map).fillna(0).astype(int)

    if TARGET_RATINGS_PER_DIALOG is not None:
        candidates = candidates[candidates["n_ratings"] < TARGET_RATINGS_PER_DIALOG]
        if candidates.empty:
            return None

    candidates["tie_breaker"] = candidates["META_dialog_id"].apply(
        lambda dialog_id: stable_tie_breaker(participant_id, str(dialog_id))
    )
    candidates = candidates.sort_values(["n_ratings", "tie_breaker"])
    return candidates.iloc[0]


def save_response(
    participant_id: str,
    study_id: str,
    session_id: str,
    dialog_row: pd.Series,
    answers: dict[str, int],
    free_comment: str = "",
) -> bool:
    submitted_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    columns = [
        "participant_id",
        "study_id",
        "session_id",
        "dialog_id",
        "language",
        "subject",
        "condition",
        "submitted_at_utc",
        "free_comment",
    ] + [q["key"] for q in QUESTIONS]

    values = [
        participant_id,
        study_id,
        session_id,
        str(dialog_row["META_dialog_id"]),
        str(dialog_row.get("language", "")),
        str(dialog_row.get("subject", "")),
        str(dialog_row.get("META_condition", "")),
        submitted_at_utc,
        free_comment.strip(),
    ] + [int(answers[q["key"]]) for q in QUESTIONS]

    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(columns)

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        cursor = conn.execute(
            f"INSERT OR IGNORE INTO responses ({column_sql}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        inserted = cursor.rowcount == 1

    export_responses_to_csv()
    return inserted


def make_continue_url(participant_id: str, prolific_pid: str, study_id: str, session_id: str) -> str:
    """Create a browser navigation URL and jump to the top anchor on the next page.

    This avoids the deprecated st.components.v1.html scroll hack and works
    by changing the URL query string plus adding #page-top.
    """
    params: dict[str, str] = {}

    if prolific_pid:
        params["PROLIFIC_PID"] = prolific_pid
        if study_id:
            params["STUDY_ID"] = study_id
        if session_id:
            params["SESSION_ID"] = session_id
    else:
        # Preserve manually entered test IDs when using the app outside Prolific.
        params["TEST_PID"] = participant_id

    # Cache-buster: makes the browser perform an actual navigation.
    # The #page-top fragment asks the browser to open the next page at the top.
    params["reload"] = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return "?" + urlencode(params) + "#page-top"


def parse_dialog_to_blocks(dialog_text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_role = "Context"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_role
        content = "\n".join(buffer).strip()
        if content:
            blocks.append((current_role, content))
        buffer = []

    for raw_line in dialog_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            if buffer:
                buffer.append("")
            continue

        lower = stripped.lower()
        if lower.startswith("robot:"):
            flush()
            current_role = "Robot"
            buffer = [stripped.split(":", 1)[1].strip()]
        elif lower.startswith("user:") or lower.startswith("visitor:"):
            flush()
            current_role = "User"
            buffer = [stripped.split(":", 1)[1].strip()]
        else:
            buffer.append(stripped)

    flush()
    return blocks


def render_dialog(dialog_text: str) -> None:
    """Render the dialog as clean Robot/User cards.

    Important: build compact HTML strings without leading indentation.
    Otherwise Streamlit's Markdown parser may display the HTML as a code block.
    """
    blocks = parse_dialog_to_blocks(dialog_text)
    html_blocks: list[str] = []

    for role, content in blocks:
        if role == "Robot":
            css_class = "qd-turn-robot"
        elif role == "User":
            css_class = "qd-turn-user"
        else:
            css_class = "qd-turn-context"

        safe_role = html.escape(role)
        safe_content = html.escape(content)

        html_blocks.append(
            f'<div class="qd-turn {css_class}">'
            f'<div class="qd-role-label">{safe_role}</div>'
            f'<div class="qd-turn-text">{safe_content}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="qd-dialog-box">{"".join(html_blocks)}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Main app
# ============================================================

st.markdown('<a id="page-top" name="page-top"></a>', unsafe_allow_html=True)
st.title("Conversational HRI quality survey (English)")
st.caption(
    f"Please read each dialog carefully and answer all {len(QUESTIONS)} questions: "
    f"Part A: Rate the Human-Robot Dialog ({len(QUESTIONS_BLOCK_A)} questions) and "
    f"Part B: Rate the Robot ({len(QUESTIONS_BLOCK_B)} questions)."
)

en_csv_path = find_latest_dialog_csv()

if en_csv_path is None:
    st.error(
        "Could not find any dialog CSV in this folder matching "
        f"`{DIALOG_INPUT_FILENAME_PATTERN}`. "
        "Run create_test_samples.py first to generate it."
    )
    st.stop()

try:
    dialogs = load_dialogs(str(en_csv_path))
except Exception as exc:
    st.error(str(exc))
    st.stop()

init_db()

prolific_pid = get_query_param("PROLIFIC_PID")
study_id = get_query_param("STUDY_ID")
session_id = get_query_param("SESSION_ID")
requested_dialog_id = get_query_param("DIALOG_ID")
test_pid = get_query_param("TEST_PID")

if prolific_pid:
    participant_id = prolific_pid
else:
    st.sidebar.header("Testing only")
    participant_id = st.sidebar.text_input(
        "Participant ID",
        value=test_pid,
        placeholder="Enter a test ID",
    ).strip()

    st.sidebar.divider()
    st.sidebar.write(f"Input rows: **{len(dialogs)}**")
    st.sidebar.write(f"Loaded: `{en_csv_path.name}`")
    st.sidebar.write(f"Responses file: `{CSV_EXPORT_PATH}`")

if not participant_id:
    st.warning("Please enter a participant ID to start.")
    st.stop()

if st.session_state.pop("saved_previous_dialog", False):
    st.success("Previous dialog saved. Please continue with the next dialog.")

completed_count = participant_completed_count(participant_id)

if completed_count >= DIALOGS_PER_PARTICIPANT:
    st.success("All required dialog ratings have been recorded. Thank you.")
    if PROLIFIC_COMPLETION_URL:
        st.markdown(f"[Return to Prolific]({PROLIFIC_COMPLETION_URL})")
    st.stop()

st.progress(completed_count / DIALOGS_PER_PARTICIPANT)
st.markdown(f"**Progress:** Dialog {completed_count + 1} of {DIALOGS_PER_PARTICIPANT}")

dialog_row = assign_dialog(dialogs, participant_id, requested_dialog_id)
if dialog_row is None:
    st.warning("No dialog is currently available for annotation.")
    st.stop()
assert dialog_row is not None

st.subheader("Dialog to rate")
st.markdown(
    f"""
    <div class="meta-line">
    Dialog ID: <b>{html.escape(str(dialog_row['META_dialog_id']))}</b> &nbsp; | &nbsp;
    Language: <b>{html.escape(str(dialog_row.get('language', '')))}</b> &nbsp; | &nbsp;
    Subject: <b>{html.escape(str(dialog_row.get('subject', '')))}</b>
    </div>
    """,
    unsafe_allow_html=True,
)

GENERIC_LIKERT_LABELS = {
    1: "1 — Very low / very poor",
    2: "2 — Low / poor",
    3: "3 — Moderate / mixed",
    4: "4 — High / good",
    5: "5 — Very high / very good",
}


def render_question(
    question: dict,
    answers: dict[str, Optional[int]],
    question_number: int,
) -> None:
    # Show only the concrete question to annotators.
    # Keep question["key"] and question["label"] only for internal storage/analysis.
    visible_question = f"{question_number}. {question['help']}"

    st.markdown(f"**{visible_question}**")

    with st.expander("Rating guidance", expanded=False):
        # Avoid showing the internal dimension label here.
        # Use rating guidance instead of construct names.
        st.markdown(f"**Rate higher when:** {question['rate_higher']}")
        st.markdown(f"**Rate lower when:** {question['rate_lower']}")

        if question.get("note"):
            st.markdown(f"**Important:** {question['note']}")

        st.markdown("**Scale guidance:**")
        st.markdown("- **1:** The quality described in the question is very low or very poor.")
        st.markdown("- **3:** The quality described in the question is mixed or moderate.")
        st.markdown("- **5:** The quality described in the question is very high or very good.")

    answers[question["key"]] = st.radio(
        label=visible_question,
        options=[1, 2, 3, 4, 5],
        format_func=lambda value: GENERIC_LIKERT_LABELS[value],
        index=None,
        horizontal=False,
        label_visibility="collapsed",
        key=f"radio_{question['key']}_{dialog_row['META_dialog_id']}",
    )
    st.write("")


with st.form("rating_form", clear_on_submit=False):
    answers: dict[str, Optional[int]] = {}

    with st.container(border=True):
        st.header("Part A: Rate the Human-Robot Dialog")
        st.caption(f"{len(QUESTIONS_BLOCK_A)} questions")
        st.markdown(ANNOTATION_INSTRUCTION_BLOCK_A)

        render_dialog(str(dialog_row["dialog_text"]))

        for i, question in enumerate(QUESTIONS_BLOCK_A, start=1):
            render_question(question, answers, i)

    with st.container(border=True):
        st.header("Part B: Rate the Robot")
        st.caption(f"{len(QUESTIONS_BLOCK_B)} questions")
        dialog_condition = str(dialog_row.get("META_condition", ""))
        block_b_image_path = BLOCK_B_IMAGE_PATHS_BY_CONDITION.get(dialog_condition)
        if block_b_image_path is not None and block_b_image_path.exists():
            st.image(str(block_b_image_path))
        elif block_b_image_path is not None:
            st.info(f"DUMMY PLACEHOLDER: image not found at `{block_b_image_path.name}`.")
        else:
            st.info(f"DUMMY PLACEHOLDER: no image configured for condition `{dialog_condition}`.")
        st.markdown(ANNOTATION_INSTRUCTION_BLOCK_B)

        for i, question in enumerate(QUESTIONS_BLOCK_B, start=len(QUESTIONS_BLOCK_A) + 1):
            render_question(question, answers, i)

    st.markdown("**Optional free comment**")
    free_comment = st.text_area(
        "If you have any additional comments about this dialog or the rating task, you can write them here.",
        placeholder="Optional: write any additional feedback here...",
        key=f"free_comment_{dialog_row['META_dialog_id']}",
    )

    submitted = st.form_submit_button(
        "Submit ratings",
        type="primary",
    )

if submitted:
    missing_questions = [
        question["label"] for question in QUESTIONS if answers.get(question["key"]) is None
    ]

    if missing_questions:
        st.error(f"Please answer all {len(QUESTIONS)} questions before submitting.")
    else:
        inserted = save_response(
            participant_id=participant_id,
            study_id=study_id,
            session_id=session_id,
            dialog_row=dialog_row,
            answers={key: int(value) for key, value in answers.items() if value is not None},
            free_comment=free_comment,
        )

        if inserted:
            new_count = participant_completed_count(participant_id)
            if new_count >= DIALOGS_PER_PARTICIPANT:
                st.success("All required dialog ratings have been recorded. Thank you.")
                if PROLIFIC_COMPLETION_URL:
                    st.markdown(f"[Return to Prolific]({PROLIFIC_COMPLETION_URL})")
            else:
                continue_url = make_continue_url(participant_id, prolific_pid, study_id, session_id)
                st.success("Previous dialog saved. Please continue with the next dialog.")
                st.markdown(
                    f'<a class="continue-button" href="{html.escape(continue_url)}" target="_self">Continue to next dialog</a>',
                    unsafe_allow_html=True,
                )
                st.stop()
        else:
            continue_url = make_continue_url(participant_id, prolific_pid, study_id, session_id)
            st.info("Your response for this dialog was already saved earlier. Please continue with the next available dialog.")
            st.markdown(
                f'<a class="continue-button" href="{html.escape(continue_url)}" target="_self">Continue to next dialog</a>',
                unsafe_allow_html=True,
            )
            st.stop()
