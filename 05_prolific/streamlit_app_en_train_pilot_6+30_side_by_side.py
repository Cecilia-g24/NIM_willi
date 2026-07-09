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
# Dialog input CSV is produced by create_train_set.py into this same folder,
# named "streamlit_train_sample_en_<N>.csv" (N = 6 fixed + 30 random). The
# most recently generated file is used automatically. Only English dialogs
# are loaded by this app.
#
# Flow:
# - Part 1 (training): the 6 dialogs marked META_selection == "fixed" are
#   shown first, in the fixed order they appear in the CSV, one at a time.
#   Every participant rates all 6.
# - Once all 6 training dialogs are rated, a readiness checkbox is shown
#   ("Do you feel ready now to continue with the actual annotations?").
#   Part 2 only starts once the participant confirms.
# - Part 2 (real annotations): dialogs marked META_selection == "random"
#   are assigned the same way create_test_samples.py-based apps do it -
#   DIALOGS_PER_PARTICIPANT dialogs, balanced by current rating count.
BASE_DIR = Path(__file__).resolve().parent

DIALOG_INPUT_FILENAME_PATTERN = "streamlit_train_sample_en_*.csv"

RESPONSES_DIR = BASE_DIR / "responses"
DB_PATH = RESPONSES_DIR / "survey_responses_en_train_pilot.sqlite"
CSV_EXPORT_PATH = RESPONSES_DIR / "survey_responses_en_train_pilot.csv"

# Prolific completion URL.
PROLIFIC_COMPLETION_URL = "https://app.prolific.com/submissions/complete?cc=C6JV4KGN"

# Number of real (part 2) dialogs one participant should annotate.
# All training (part 1) dialogs are mandatory and are not subject to a quota.
DIALOGS_PER_PARTICIPANT = 30

# Optional quota per real dialog. Set to None if you do not want a cap.
TARGET_RATINGS_PER_DIALOG: Optional[int] = None

REQUIRED_COLUMNS = [
    "META_dialog_id",
    "META_condition",
    "META_selection",
    "language",
    "subject",
    "dialog_text",
]

ANNOTATION_INSTRUCTION_BLOCK_A = (
    "Please read the whole dialog and rate only the human participant's behavior as visible in the transcript. "
    "Do not judge the robot's quality directly, and do not base your rating on whether you personally like the robot. "
    "Focus only on what the human participant says, asks, reveals, and appears to express during the conversation."
)

QUESTIONS_BLOCK_A = [
    {
        "key": "user_engagement_enjoyment",
        "label": "1. User engagement / enjoyment",
        "headline": "User engagement / enjoyment",
        "help": "To what extent does the user appear engaged, interested, and motivated to keep the conversation going?",
        "definition": (
            "User engagement / enjoyment refers to how involved, interested, and motivated the human "
            "participant appears to be to keep the conversation going. Judge this only from visible cues in the transcript, "
            "such as active participation, reactions, questions, cooperation, enthusiasm, reluctance, "
            "or attempts to end the exchange."
        ),
        "rate_higher": (
            "The user participates actively, gives meaningful answers, reacts to the robot, asks questions, "
            "shows interest in the topic, or appears motivated to keep the conversation going."
        ),
        "rate_lower": (
            "The user gives minimal answers, appears passive or bored, ignores the robot's questions, resists "
            "the topic, shows little interest, or tries to end the conversation."
        ),
        "anchors": {
            1: "Very low/absent: the user barely participates, gives minimal or resistant answers, or clearly wants to stop.",
            3: "Moderate: the user answers the robot, but with limited detail, mixed interest, or occasional disengagement.",
            5: "Very high: the user actively participates, reacts to the robot, and appears motivated to keep the conversation going.",
        },
    },
    {
        "key": "user_self_disclosure",
        "label": "2. User self-disclosure",
        "headline": "User self-disclosure",
        "help": "To what extent does the user reveal personal information, experiences, preferences, opinions, or emotions?",
        "definition": (
            "User self-disclosure refers to how much the user reveals about themselves, including personal "
            "experiences, preferences, feelings, evaluations, memories, opinions, or everyday habits. "
            "This is about personal openness, not simply the number of words."
        ),
        "rate_higher": (
            "The user shares personal experiences, preferences, feelings, opinions, memories, or details about "
            "their own life, rather than only giving generic or factual answers."
        ),
        "rate_lower": (
            "The user gives impersonal, generic, factual, or very short answers and reveals little or nothing "
            "about themselves."
        ),
        "anchors": {
            1: "Very low/absent: the user reveals almost nothing personal.",
            3: "Moderate: the user reveals some personal preferences, opinions, or experiences, but only briefly or occasionally.",
            5: "Very high: the user openly shares personal experiences, preferences, emotions, or opinions in meaningful detail.",
        },
    },
    {
        "key": "user_topical_alignment",
        "label": "3. User topical alignment",
        "headline": "User topical alignment",
        "help": "To what extent does the user stay aligned with the topic introduced or developed by the robot?",
        "definition": (
            "User topical alignment refers to whether the user stays connected to the robot-proposed topic "
            "or develops it in a relevant way. It captures cooperation with the ongoing topic, rather than "
            "whether the topic itself is interesting."
        ),
        "rate_higher": (
            "The user answers in a way that is relevant to the robot's question or topic, follows the current "
            "topic, and develops the same conversational thread."
        ),
        "rate_lower": (
            "The user gives off-topic, unrelated, evasive, or mismatched answers, redirects the conversation "
            "without clear connection, or does not cooperate with the current topic."
        ),
        "anchors": {
            1: "Very low/absent: the user is mostly off-topic, evasive, or disconnected from the robot's topic.",
            3: "Moderate: the user is partly aligned, but sometimes gives weakly related, unclear, or redirected answers.",
            5: "Very high: the user consistently stays relevant to the robot's topic and develops the same thread.",
        },
    },
    {
        "key": "user_elaboration_informativeness",
        "label": "4. User elaboration / informativeness",
        "headline": "User elaboration / informativeness",
        "help": "To what extent does the user provide meaningful detail beyond short or minimal answers?",
        "definition": (
            "User elaboration / informativeness refers to how much meaningful content the user provides. "
            "It captures whether the user's answers add useful detail, explanation, examples, or context, "
            "rather than only minimal responses."
        ),
        "rate_higher": (
            "The user gives informative answers with details, reasons, examples, explanations, or context that "
            "make their contribution meaningful."
        ),
        "rate_lower": (
            "The user gives short, vague, repetitive, or minimal answers, such as yes/no responses or brief fragments, "
            "with little meaningful detail."
        ),
        "anchors": {
            1: "Very low/absent: the user provides almost no meaningful detail beyond minimal answers.",
            3: "Moderate: the user provides some useful detail, but many answers remain brief or only partly informative.",
            5: "Very high: the user provides rich, meaningful, and informative detail across the conversation.",
        },
    },
    {
        "key": "user_initiative_active_contribution",
        "label": "5. User initiative / active contribution",
        "headline": "User initiative / active contribution",
        "help": "To what extent does the user actively contribute to developing the conversation, for example by asking questions, adding topics, giving opinions, or steering the exchange?",
        "definition": (
            "User initiative / active contribution refers to whether the user does more than simply answer "
            "the robot's prompts. It captures active participation in shaping the exchange, such as asking "
            "questions, introducing related ideas, giving opinions, or steering the direction of the conversation."
        ),
        "rate_higher": (
            "The user asks questions, introduces or develops topics, gives opinions, reacts proactively, "
            "pushes the conversation forward, or otherwise helps shape the exchange."
        ),
        "rate_lower": (
            "The user only responds when prompted, gives passive or minimal answers, rarely adds anything new, "
            "and does not help develop the conversation."
        ),
        "anchors": {
            1: "Very low/absent: the user is almost entirely passive and only gives minimal prompted answers.",
            3: "Moderate: the user occasionally adds something new or gives an opinion, but mostly follows the robot's lead.",
            5: "Very high: the user actively shapes the exchange by asking questions, adding topics, or steering the conversation.",
        },
    },
    {
        "key": "user_politeness",
        "label": "6. User politeness",
        "headline": "User politeness",
        "help": "To what extent does the user address the robot in a polite, respectful, or socially affiliative way?",
        "definition": (
            "User politeness refers to whether the user treats the robot as a socially addressable interaction "
            "partner. This can include polite wording, respectful responses, greetings, thanks, friendly comments, "
            "or other affiliative language."
        ),
        "rate_higher": (
            "The user uses polite, respectful, friendly, appreciative, or socially affiliative language toward the robot, "
            "such as greetings, thanks, softening phrases, or cooperative wording."
        ),
        "rate_lower": (
            "The user is blunt, dismissive, rude, disrespectful, socially cold, or gives no signs of treating the robot "
            "as a social interaction partner."
        ),
        "anchors": {
            1: "Very low/absent: the user is rude, dismissive, or shows no polite or affiliative behavior.",
            3: "Moderate: the user is generally neutral or minimally polite, with limited social warmth.",
            5: "Very high: the user is clearly polite, respectful, friendly, or socially affiliative toward the robot.",
        },
    },
    {
        "key": "user_frustration_dissatisfaction",
        "label": "7. User frustration / dissatisfaction",
        "headline": "User frustration / dissatisfaction",
        "help": "To what extent does the user appear frustrated, annoyed, impatient, confused, disappointed, or unwilling to continue the conversation?",
        "definition": (
            "User frustration / dissatisfaction refers to visible negative reactions from the user during the dialog. "
            "This includes annoyance, impatience, confusion, disappointment, irritation, resistance, or signals that "
            "the user does not want to continue."
        ),
        "rate_higher": (
            "The user appears annoyed, impatient, confused, disappointed, irritated, resistant, dissatisfied, or unwilling "
            "to continue the conversation."
        ),
        "rate_lower": (
            "The user shows little or no visible frustration, impatience, confusion, dissatisfaction, resistance, or desire "
            "to stop the conversation."
        ),
        "anchors": {
            1: "Very low/absent: the user shows little or no frustration, dissatisfaction, or unwillingness to continue.",
            3: "Moderate: the user shows some confusion, impatience, dissatisfaction, or reluctance, but not consistently.",
            5: "Very high: the user clearly appears frustrated, annoyed, dissatisfied, or unwilling to continue.",
        },
    },
    {
        "key": "overall_conversational_interaction_quality",
        "label": "8. Overall conversational interaction quality",
        "headline": "Overall conversational interaction quality",
        "help": "Overall, based only on the user's visible behavior in the transcript, how high is the quality of the user's contribution to the conversation with the robot?",
        "definition": (
            "Overall conversational interaction quality refers to the general quality of the user's visible "
            "contribution to the human-robot conversation. It captures the overall impression of whether the user "
            "helps create a coherent, smooth, meaningful, and socially appropriate exchange with the robot. "
            "This is a holistic judgment based on the whole dialog, not a simple average of the previous ratings."
        ),
        "rate_higher": (
            "The user contributes to a coherent, smooth, meaningful, and socially positive conversation. "
            "They respond in ways that make the exchange feel successful, cooperative, and interactionally rich."
        ),
        "rate_lower": (
            "The user's contribution makes the conversation feel weak, minimal, disconnected, one-sided, awkward, "
            "or unsuccessful, even if some individual answers are understandable."
        ),
        "note": (
            "Do not judge the robot's technical quality, the attractiveness of the topic, or whether you personally "
            "like the robot. Focus on the observable quality of the user's contribution to the interaction."
        ),
        "anchors": {
            1: "Very low: the user's contribution makes the conversation feel largely unsuccessful, disconnected, or minimal.",
            3: "Moderate: the user's contribution supports the conversation to some extent, but the interaction remains partly limited, uneven, or weak.",
            5: "Very high: the user's contribution makes the conversation feel coherent, smooth, meaningful, and socially successful.",
        },
    },
]


QUESTIONS = QUESTIONS_BLOCK_A


# ============================================================
# Basic setup
# ============================================================

st.set_page_config(
    page_title="Observer-rated user behavior in HRI survey - Training Pilot",
    page_icon="📝",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .qd-dialog-box {
        border: 1px solid #d9dee7;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        max-height: calc(100vh - 160px);
        overflow-y: auto;
        margin-bottom: 1.5rem;
        color: #111827 !important;
        position: sticky;
        top: 1rem;
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
    div[role="radiogroup"] {
        gap: 0.6rem 2.2rem;
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
    """Find the most recently generated create_train_set.py output for English."""
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
    df["META_selection"] = df["META_selection"].fillna("").astype(str)
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
                phase TEXT NOT NULL,
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_confirmations (
                participant_id TEXT PRIMARY KEY,
                confirmed_at_utc TEXT NOT NULL
            )
            """
        )

        # Migrations for databases created before these columns were added.
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(responses)").fetchall()
        }
        if "free_comment" not in existing_columns:
            conn.execute("ALTER TABLE responses ADD COLUMN free_comment TEXT")
            existing_columns.add("free_comment")
        if "condition" not in existing_columns:
            conn.execute("ALTER TABLE responses ADD COLUMN condition TEXT")
            existing_columns.add("condition")
        if "phase" not in existing_columns:
            conn.execute("ALTER TABLE responses ADD COLUMN phase TEXT NOT NULL DEFAULT 'main'")
            existing_columns.add("phase")

        # Add columns for any newly added or renamed questions.
        # Existing response databases from earlier survey versions may not contain
        # the current question keys.
        for question in QUESTIONS:
            question_key = question["key"]
            if question_key not in existing_columns:
                conn.execute(f"ALTER TABLE responses ADD COLUMN {question_key} INTEGER")
                existing_columns.add(question_key)

        conn.commit()

        # If old question columns were removed from QUESTIONS but still exist in the DB
        # as NOT NULL, INSERT OR IGNORE silently drops every new row. Rebuild the table
        # to make those stale columns nullable so inserts work again.
        col_rows = conn.execute("PRAGMA table_info(responses)").fetchall()
        non_question = {
            "participant_id", "study_id", "session_id", "dialog_id", "phase",
            "language", "subject", "condition", "submitted_at_utc", "free_comment",
        }
        current_keys = {q["key"] for q in QUESTIONS}
        stale_notnull = [r for r in col_rows if r[3] and r[1] not in current_keys and r[1] not in non_question]

        if stale_notnull:
            required_notnull = {"participant_id", "dialog_id", "phase", "submitted_at_utc"}
            col_names = [r[1] for r in col_rows]
            defs = []
            for r in col_rows:
                name, typ, notnull = r[1], r[2], r[3]
                nn = " NOT NULL" if (notnull and name in required_notnull) else ""
                defs.append(f"    {name} {typ}{nn}")
            col_list = ", ".join(col_names)
            defs_sql = ",\n".join(defs)
            conn.executescript(
                f"CREATE TABLE responses_new (\n{defs_sql},\n    PRIMARY KEY (participant_id, dialog_id)\n);\n"
                f"INSERT OR IGNORE INTO responses_new ({col_list}) SELECT {col_list} FROM responses;\n"
                f"DROP TABLE responses;\n"
                f"ALTER TABLE responses_new RENAME TO responses;\n"
            )


def export_responses_to_csv() -> None:
    ordered_columns = [
        "participant_id",
        "study_id",
        "session_id",
        "dialog_id",
        "phase",
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


def participant_completed_count(participant_id: str, phase: str) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM responses WHERE participant_id = ? AND phase = ?",
            (participant_id, phase),
        ).fetchone()
    return int(row[0]) if row else 0


def participant_answered_dialog_ids(participant_id: str, phase: str) -> set[str]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT dialog_id FROM responses WHERE participant_id = ? AND phase = ?",
            (participant_id, phase),
        ).fetchall()
    return {str(row[0]) for row in rows}


def is_training_confirmed(participant_id: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM training_confirmations WHERE participant_id = ?",
            (participant_id,),
        ).fetchone()
    return row is not None


def confirm_training(participant_id: str) -> None:
    confirmed_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO training_confirmations (participant_id, confirmed_at_utc) VALUES (?, ?)",
            (participant_id, confirmed_at_utc),
        )
        conn.commit()


def stable_tie_breaker(participant_id: str, dialog_id: str) -> int:
    text = f"{participant_id}::{dialog_id}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def next_training_dialog(training_df: pd.DataFrame, answered_ids: set[str]) -> Optional[pd.Series]:
    """Return the next un-answered training dialog, in fixed CSV order."""
    remaining = training_df[~training_df["META_dialog_id"].isin(answered_ids)]
    if remaining.empty:
        return None
    return remaining.iloc[0]


def assign_main_dialog(main_df: pd.DataFrame, participant_id: str, requested_dialog_id: str = "") -> Optional[pd.Series]:
    """
    Assign one real (part 2) dialog to the participant.

    Priority:
    1. If ?DIALOG_ID=... is present and not already rated by this participant, use that dialog.
    2. Otherwise, choose among dialogs not yet rated by this participant.
       The assignment is balanced by current rating count per dialog.
    """
    answered_ids = participant_answered_dialog_ids(participant_id, phase="main")

    requested_dialog_id = str(requested_dialog_id).strip()
    if requested_dialog_id and requested_dialog_id not in answered_ids:
        chosen = main_df[main_df["META_dialog_id"] == requested_dialog_id]
        if not chosen.empty:
            return chosen.iloc[0]

    with sqlite3.connect(DB_PATH) as conn:
        counts = pd.read_sql_query(
            "SELECT dialog_id, COUNT(*) AS n_ratings FROM responses WHERE phase = 'main' GROUP BY dialog_id",
            conn,
        )

    count_map = (
        dict(zip(counts["dialog_id"].astype(str), counts["n_ratings"].astype(int)))
        if not counts.empty
        else {}
    )

    candidates = main_df[~main_df["META_dialog_id"].isin(answered_ids)].copy()
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
    phase: str,
    free_comment: str = "",
) -> bool:
    submitted_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    columns = [
        "participant_id",
        "study_id",
        "session_id",
        "dialog_id",
        "phase",
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
        phase,
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
            current_role = "Visitor"
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
        elif role == "Visitor":
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


GENERIC_LIKERT_LABELS = {
    1: "1 = Very low",
    2: "2 = Low",
    3: "3 = Moderate",
    4: "4 = High",
    5: "5 = Very high",
}


def show_pending_continue(pending: dict) -> None:
    """Show the saved-confirmation message and the Continue link."""
    if pending["kind"] == "success":
        st.success(pending["message"])
    else:
        st.info(pending["message"])
    st.markdown(
        f'<a class="continue-button" href="{html.escape(pending["url"])}" target="_self">Continue</a>',
        unsafe_allow_html=True,
    )


def render_rating_form(
    dialog_row: pd.Series,
    phase: str,
    participant_id: str,
    study_id: str,
    session_id: str,
    prolific_pid: str,
) -> None:
    """Render one dialog + all questions, save the response on submit, then
    show a continue link below it on the same page. Used for both training
    and real dialogs."""
    answers: dict[str, Optional[int]] = {}

    # True when this dialog was just submitted and we are waiting for the
    # participant to click Continue. The form is re-rendered read-only with
    # the Continue link below the submit button.
    pending = st.session_state.get("pending_continue")
    pending_here = bool(
        pending
        and pending.get("dialog_id") == str(dialog_row["META_dialog_id"])
        and pending.get("phase") == phase
    )

    def render_question(question: dict, question_number: int) -> None:
        # Show only the concrete question to annotators.
        # Keep question["key"] and question["label"] only for internal storage/analysis.
        visible_question = f"{question_number}. {question['help']}"

        st.markdown(f"**{question_number}. {question['headline']}**")
        st.markdown(f"*{question['help']}*")

        with st.expander("Rating guidance", expanded=False):
            # Avoid showing the internal dimension label here.
            # Use rating guidance instead of construct names.
            st.markdown(f"**Rate higher when:** {question['rate_higher']}")
            st.markdown(f"**Rate lower when:** {question['rate_lower']}")

            if question.get("note"):
                st.markdown(f"**Important:** {question['note']}")

            st.markdown("**Scale guidance:**")
            st.markdown("- **1:** The behavior described in the question is very low or absent.")
            st.markdown("- **3:** The behavior described in the question is moderate.")
            st.markdown("- **5:** The behavior described in the question is very high.")

        answers[question["key"]] = st.radio(
            label=visible_question,
            options=[1, 2, 3, 4, 5],
            format_func=lambda value: GENERIC_LIKERT_LABELS[value],
            index=None,
            horizontal=True,
            label_visibility="collapsed",
            disabled=pending_here,
            key=f"radio_{phase}_{question['key']}_{dialog_row['META_dialog_id']}",
        )
        st.write("")

    with st.form(f"rating_form_{phase}_{dialog_row['META_dialog_id']}", clear_on_submit=False):
        st.markdown(ANNOTATION_INSTRUCTION_BLOCK_A)

        dialog_col, questions_col = st.columns([1, 1], gap="large")

        with dialog_col:
            render_dialog(str(dialog_row["dialog_text"]))

        with questions_col:
            for i, question in enumerate(QUESTIONS_BLOCK_A, start=1):
                render_question(question, i)

            st.markdown("**Optional free comment**")
            free_comment = st.text_area(
                "If you have any additional comments about this dialog or the rating task, you can write them here.",
                placeholder="Optional: write any additional feedback here...",
                disabled=pending_here,
                key=f"free_comment_{phase}_{dialog_row['META_dialog_id']}",
            )

            submitted = st.form_submit_button(
                "Submit ratings",
                type="primary",
                disabled=pending_here,
            )

            if pending_here:
                show_pending_continue(pending)

    if pending_here:
        st.stop()

    if not submitted:
        return

    missing_questions = [
        question["label"] for question in QUESTIONS if answers.get(question["key"]) is None
    ]

    if missing_questions:
        st.error(f"Please answer all {len(QUESTIONS)} questions before submitting.")
        return

    inserted = save_response(
        participant_id=participant_id,
        study_id=study_id,
        session_id=session_id,
        dialog_row=dialog_row,
        answers={key: int(value) for key, value in answers.items() if value is not None},
        phase=phase,
        free_comment=free_comment,
    )

    continue_url = make_continue_url(participant_id, prolific_pid, study_id, session_id)

    if inserted:
        message, kind = "Response saved. Please continue.", "success"
    else:
        message, kind = (
            "Your response for this dialog was already saved earlier. Please continue with the next available dialog.",
            "info",
        )

    # Remember that we are waiting for the participant to click Continue.
    # Without this flag, any further rerun (e.g. a double click on the submit
    # button) would immediately render the next dialog, because dialog
    # selection is derived from the database on every rerun. The Continue
    # link performs a real browser navigation, which starts a fresh session
    # and thereby clears this flag. The rerun re-renders this same dialog
    # (read-only) with the Continue link below the submit button.
    st.session_state["pending_continue"] = {
        "message": message,
        "kind": kind,
        "url": continue_url,
        "dialog_id": str(dialog_row["META_dialog_id"]),
        "phase": phase,
    }
    st.rerun()


# ============================================================
# Main app
# ============================================================

st.markdown('<a id="page-top" name="page-top"></a>', unsafe_allow_html=True)
st.title("Observer-rated user behavior in HRI survey (English) - Training Pilot")
st.caption(
    "Part 1: a few training dialogs to help you get familiar with the rating task. "
    f"Part 2: {DIALOGS_PER_PARTICIPANT} real dialogs to annotate."
)

en_csv_path = find_latest_dialog_csv()

if en_csv_path is None:
    st.error(
        "Could not find any dialog CSV in this folder matching "
        f"`{DIALOG_INPUT_FILENAME_PATTERN}`. "
        "Run create_train_set.py first to generate it."
    )
    st.stop()

try:
    dialogs = load_dialogs(str(en_csv_path))
except Exception as exc:
    st.error(str(exc))
    st.stop()

training_df = dialogs[dialogs["META_selection"] == "fixed"].reset_index(drop=True)
main_df = dialogs[dialogs["META_selection"] != "fixed"].reset_index(drop=True)

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
    st.sidebar.write(f"Training rows: **{len(training_df)}**")
    st.sidebar.write(f"Real dialog rows: **{len(main_df)}**")
    st.sidebar.write(f"Loaded: `{en_csv_path.name}`")
    st.sidebar.write(f"Responses file: `{CSV_EXPORT_PATH}`")

if not participant_id:
    st.warning("Please enter a participant ID to start.")
    st.stop()

# If a response was just saved, keep showing that same dialog page (with the
# Continue link below the submit button) until the participant actually clicks
# Continue, which navigates and starts a fresh session. This prevents double
# clicks or stray reruns from skipping ahead.
pending_continue = st.session_state.get("pending_continue")
if pending_continue:
    pending_dialog_id = pending_continue.get("dialog_id")
    if pending_dialog_id:
        pending_rows = dialogs[dialogs["META_dialog_id"] == pending_dialog_id]
        if not pending_rows.empty:
            # render_rating_form shows the Continue link and stops the script.
            render_rating_form(
                dialog_row=pending_rows.iloc[0],
                phase=pending_continue["phase"],
                participant_id=participant_id,
                study_id=study_id,
                session_id=session_id,
                prolific_pid=prolific_pid,
            )
    # Fallback (e.g. training confirmation, or dialog no longer in the CSV):
    # show the confirmation message and Continue link on their own.
    show_pending_continue(pending_continue)
    st.stop()

training_answered_ids = participant_answered_dialog_ids(participant_id, phase="training")
training_total = len(training_df)
training_completed_count = int(training_df["META_dialog_id"].isin(training_answered_ids).sum())
training_done = training_total == 0 or training_completed_count >= training_total

if not training_done:
    st.info(
        "**Part 1 of 2: training dialogs**\n\n"
        "These practice dialogs help you get familiar with the rating task "
        "before you start the real annotations. Every participant rates all "
        f"{training_total} training dialogs."
    )
    st.progress(training_completed_count / training_total)
    st.markdown(f"**Training progress:** Dialog {training_completed_count + 1} of {training_total}")

    dialog_row = next_training_dialog(training_df, training_answered_ids)
    if dialog_row is None:
        st.warning("No training dialog is currently available.")
        st.stop()

    render_rating_form(
        dialog_row=dialog_row,
        phase="training",
        participant_id=participant_id,
        study_id=study_id,
        session_id=session_id,
        prolific_pid=prolific_pid,
    )

elif not is_training_confirmed(participant_id):
    st.success(f"You have completed all {training_total} training dialogs.")
    st.markdown("### Ready to continue?")

    ready = st.checkbox("Do you feel ready now to continue with the actual annotations?")

    if st.button("Continue to real annotations", type="primary", disabled=not ready):
        confirm_training(participant_id)
        continue_url = make_continue_url(participant_id, prolific_pid, study_id, session_id)
        st.session_state["pending_continue"] = {
            "message": "Training confirmed. Please continue with the real annotations.",
            "kind": "success",
            "url": continue_url,
        }
        st.rerun()

else:
    main_completed_count = participant_completed_count(participant_id, phase="main")

    if main_completed_count >= DIALOGS_PER_PARTICIPANT:
        st.success("All required dialog ratings have been recorded. Thank you.")
        if PROLIFIC_COMPLETION_URL:
            st.markdown(f"[Return to Prolific]({PROLIFIC_COMPLETION_URL})")
        st.stop()

    st.info("**Part 2 of 2: real annotations**")
    st.progress(main_completed_count / DIALOGS_PER_PARTICIPANT)
    st.markdown(f"**Progress:** Dialog {main_completed_count + 1} of {DIALOGS_PER_PARTICIPANT}")

    dialog_row = assign_main_dialog(main_df, participant_id, requested_dialog_id)
    if dialog_row is None:
        st.warning("No dialog is currently available for annotation.")
        st.stop()

    render_rating_form(
        dialog_row=dialog_row,
        phase="main",
        participant_id=participant_id,
        study_id=study_id,
        session_id=session_id,
        prolific_pid=prolific_pid,
    )
