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
# Input CSV should be located in the same folder: 05_prolific/pilot_batch_en.csv
BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "pilot_batch_en.csv"
RESPONSES_DIR = BASE_DIR / "responses"
DB_PATH = RESPONSES_DIR / "survey_responses_final_metrics.sqlite"
CSV_EXPORT_PATH = RESPONSES_DIR / "survey_responses_final_metrics.csv"

# Prolific completion URL.
PROLIFIC_COMPLETION_URL = "https://app.prolific.com/submissions/complete?cc=C6JV4KGN"

# Number of dialogs one participant should annotate.
DIALOGS_PER_PARTICIPANT = 3

# Optional quota per dialog. Set to None if you do not want a cap.
TARGET_RATINGS_PER_DIALOG: Optional[int] = None

REQUIRED_COLUMNS = ["META_dialog_id", "language", "subject", "dialog_text"]

QUESTIONS = [
    {
        "key": "overall_dialog_quality",
        "label": "1. Overall dialog quality",
        "help": "How would you rate the overall quality of the conversation?",
        "definition": (
            "Overall dialog quality refers to how good the conversation feels as a complete "
            "interaction. Consider whether it is understandable, smooth, coherent, and useful or "
            "pleasant as a robot-user exchange."
        ),
        "rate_higher": (
            "The conversation flows well, the robot responds appropriately, the user can "
            "participate easily, and the interaction feels coherent overall."
        ),
        "rate_lower": (
            "The conversation feels broken, repetitive, confusing, awkward, or difficult to continue."
        ),
        "anchors": {
            1: "Very poor conversation: mostly confusing, broken, repetitive, or hard to follow.",
            3: "Acceptable but mixed: some parts work, but there are noticeable problems such as awkward transitions, misunderstandings, or weak flow.",
            5: "Very good conversation: coherent, easy to follow, responsive, and effective as a complete dialog.",
        },
        "likert_labels": {
            1: "1 — Very low quality",
            2: "2 — Low quality",
            3: "3 — Moderate quality",
            4: "4 — High quality",
            5: "5 — Very high quality",
        },
    },
    {
        "key": "robot_understanding_responsiveness",
        "label": "2. Robot understanding and responsiveness",
        "help": "How well did the robot understand and respond to the user?",
        "definition": (
            "Robot understanding and responsiveness measures how well the robot seems to understand "
            "what the user says and how appropriately it responds."
        ),
        "rate_higher": (
            "The robot acknowledges the user’s actual answer, asks relevant follow-up questions, "
            "adapts to corrections, and responds appropriately to confusion or resistance."
        ),
        "rate_lower": (
            "The robot misunderstands the user, gives generic replies, ignores user input, repeats "
            "scripted questions, or continues as if the user had said something else."
        ),
        "anchors": {
            1: "The robot mostly fails to understand or respond to the user; replies are irrelevant or clearly mismatched.",
            3: "The robot understands some user input but also misses, ignores, or misinterprets important parts.",
            5: "The robot consistently understands the user and gives relevant, appropriate, and helpful responses.",
        },
        "likert_labels": {
            1: "1 — Very poor",
            2: "2 — Poor",
            3: "3 — Moderate",
            4: "4 — Good",
            5: "5 — Very good",
        },
    },
    {
        "key": "topic_coherence",
        "label": "3. Topic coherence",
        "help": "How coherent and on-topic was the conversation?",
        "definition": (
            "Topic coherence refers to whether the conversation stays logically connected to the "
            "selected topic and whether transitions between turns make sense."
        ),
        "rate_higher": (
            "The conversation remains focused on the selected topic, follow-up questions connect "
            "to previous answers, and any topic changes are clear."
        ),
        "rate_lower": (
            "The conversation jumps between topics, introduces unrelated content, gets stuck in "
            "irrelevant loops, or fails to recover from off-topic user input."
        ),
        "anchors": {
            1: "Very incoherent: the conversation often goes off-topic or becomes difficult to connect logically.",
            3: "Somewhat coherent: the main topic is visible, but there are noticeable irrelevant or confusing parts.",
            5: "Highly coherent: the conversation stays on-topic and each turn follows naturally from the previous one.",
        },
        "likert_labels": {
            1: "1 — Very low coherence",
            2: "2 — Low coherence",
            3: "3 — Moderate coherence",
            4: "4 — High coherence",
            5: "5 — Very high coherence",
        },
    },
    {
        "key": "user_engagement",
        "label": "4. User engagement",
        "help": "How engaged did the user seem during the conversation?",
        "definition": (
            "User engagement refers to how involved, interested, and willing to participate the "
            "user appears to be."
        ),
        "rate_higher": (
            "The user gives meaningful answers, expands on experiences, asks questions, reacts "
            "to the robot, or continues the exchange voluntarily."
        ),
        "rate_lower": (
            "The user gives very short answers, seems passive, avoids the task, says they do not "
            "want to continue, or provides only minimal responses."
        ),
        "anchors": {
            1: "Very low engagement: the user barely participates, gives minimal answers, or tries to end the interaction.",
            3: "Moderate engagement: the user answers the questions but with limited detail or inconsistent interest.",
            5: "Very high engagement: the user actively participates, gives detailed answers, and appears interested in continuing.",
        },
        "likert_labels": {
            1: "1 — Very low engagement",
            2: "2 — Low engagement",
            3: "3 — Moderate engagement",
            4: "4 — High engagement",
            5: "5 — Very high engagement",
        },
    },
    {
        "key": "user_frustration_resistance",
        "label": "5. User frustration or resistance",
        "help": "How much frustration, annoyance, resistance, or rejection did the user show?",
        "definition": (
            "User frustration or resistance measures how much annoyance, reluctance, rejection, "
            "or discomfort the user shows toward the robot or the conversation."
        ),
        "note": "This is a negative item: a higher score means more frustration or resistance.",
        "rate_higher": (
            "The user complains, rejects the topic, challenges the robot, expresses annoyance, "
            "refuses to answer, asks to stop, or shows irritation."
        ),
        "rate_lower": (
            "The user appears comfortable, cooperative, neutral, or positive."
        ),
        "anchors": {
            1: "No visible frustration or resistance: the user seems cooperative or neutral.",
            3: "Some frustration or resistance: the user shows mild annoyance, confusion, reluctance, or impatience.",
            5: "Very high frustration or resistance: the user clearly rejects the robot or conversation, complains strongly, or wants to stop.",
        },
        "likert_labels": {
            1: "1 — No frustration/resistance",
            2: "2 — Low frustration/resistance",
            3: "3 — Moderate frustration/resistance",
            4: "4 — High frustration/resistance",
            5: "5 — Very high frustration/resistance",
        },
    },
    {
        "key": "user_disclosure_specificity",
        "label": "6. User disclosure or specificity",
        "help": "How much personal, specific, or detailed information did the user share?",
        "definition": (
            "User disclosure or specificity refers to how much concrete, personal, or detailed "
            "information the user shares."
        ),
        "rate_higher": (
            "The user gives specific examples, personal habits, preferences, experiences, reasons, "
            "or contextual details."
        ),
        "rate_lower": (
            "The user gives vague, very short, generic, or unclear answers."
        ),
        "anchors": {
            1: "Very low disclosure/specificity: the user gives almost no useful detail.",
            3: "Moderate disclosure/specificity: the user gives some information, but it remains brief or general.",
            5: "Very high disclosure/specificity: the user shares concrete, personal, and detailed information.",
        },
        "likert_labels": {
            1: "1 — Very low disclosure/specificity",
            2: "2 — Low disclosure/specificity",
            3: "3 — Moderate disclosure/specificity",
            4: "4 — High disclosure/specificity",
            5: "5 — Very high disclosure/specificity",
        },
    },
    {
        "key": "robot_human_likeness_social_presence",
        "label": "7. Robot human-likeness / social presence",
        "help": "How human-like or socially present did the robot seem in the conversation?",
        "definition": (
            "Robot human-likeness / social presence measures how much the robot comes across as "
            "a socially present interaction partner rather than a purely mechanical question-answer system."
        ),
        "rate_higher": (
            "The robot uses natural language, acknowledges the user personally, shows warmth or "
            "personality, uses humor appropriately, reacts socially, or creates the impression of a conversational partner."
        ),
        "rate_lower": (
            "The robot feels mechanical, scripted, impersonal, repetitive, emotionally flat, or disconnected from the user."
        ),
        "anchors": {
            1: "Very low human-likeness/social presence: the robot feels purely mechanical or scripted.",
            3: "Moderate human-likeness/social presence: the robot shows some social cues but still feels partly scripted or artificial.",
            5: "Very high human-likeness/social presence: the robot feels socially present, personable, and responsive as an interaction partner.",
        },
        "likert_labels": {
            1: "1 — Very low",
            2: "2 — Low",
            3: "3 — Moderate",
            4: "4 — High",
            5: "5 — Very high",
        },
    },
    {
        "key": "robot_trustworthiness_comfort",
        "label": "8. Robot trustworthiness / comfort",
        "help": "How trustworthy and comfortable did the robot seem as an interaction partner?",
        "definition": (
            "Robot trustworthiness / comfort measures whether the robot seems like a reliable, "
            "appropriate, and comfortable interaction partner."
        ),
        "rate_higher": (
            "The robot seems respectful, understandable, predictable, non-judgmental, and safe to interact with."
        ),
        "rate_lower": (
            "The robot seems confusing, intrusive, dismissive, inappropriate, overly pushy, unreliable, or uncomfortable."
        ),
        "anchors": {
            1: "Very low trust/comfort: the robot feels unreliable, inappropriate, uncomfortable, or difficult to trust.",
            3: "Moderate trust/comfort: the robot is mostly acceptable, but some responses feel awkward, confusing, or uncomfortable.",
            5: "Very high trust/comfort: the robot feels respectful, reliable, comfortable, and appropriate as a conversation partner.",
        },
        "likert_labels": {
            1: "1 — Very low",
            2: "2 — Low",
            3: "3 — Moderate",
            4: "4 — High",
            5: "5 — Very high",
        },
    },
]


# ============================================================
# Basic setup
# ============================================================

st.set_page_config(
    page_title="Dialog Rating Survey",
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
    .attention-box {
        border: 1px solid #d1d5db;
        background: #f9fafb;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin: 0.75rem 0 1rem 0;
        color: #111827 !important;
    }
    .attention-box b {
        color: #111827 !important;
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
                submitted_at_utc TEXT NOT NULL,
                free_comment TEXT,
                {question_columns},
                PRIMARY KEY (participant_id, dialog_id)
            )
            """
        )

        # Migration for databases created before the optional comment field was added.
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(responses)").fetchall()
        }
        if "free_comment" not in existing_columns:
            conn.execute("ALTER TABLE responses ADD COLUMN free_comment TEXT")

        conn.commit()


def export_responses_to_csv() -> None:
    ordered_columns = [
        "participant_id",
        "study_id",
        "session_id",
        "dialog_id",
        "language",
        "subject",
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
        elif lower.startswith("user:"):
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
st.title("Dialog Rating Survey")
st.caption("Please read each dialog carefully and answer all 8 questions.")

try:
    dialogs = load_dialogs(str(INPUT_CSV))
except FileNotFoundError:
    st.error(f"Could not find the input CSV: `{INPUT_CSV}`")
    st.stop()
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

current_dialog_id = str(dialog_row["META_dialog_id"])

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

st.markdown(
    """
    <div class="attention-box">
        <b>Attention check</b><br>
        Please start from the top of this dialog and read the full exchange before answering the rating questions.
    </div>
    """,
    unsafe_allow_html=True,
)
attention_checked = st.checkbox(
    "I understand that I should read the full dialog before rating it.",
    key=f"attention_check_{current_dialog_id}",
)

render_dialog(str(dialog_row["dialog_text"]))

st.subheader("Your ratings")
st.markdown(
    """
    Please judge the **full conversation**, not just one turn. Use **1** for a very low level,
    **3** for a moderate or mixed level, and **5** for a very high level of the described feature.
    Do not judge the user personally; focus on the interaction. Treat transcription errors leniently
    if the intended meaning is still understandable.
    """
)
if not attention_checked:
    st.info("Please tick the attention check above the dialog before submitting your ratings.")

with st.form("rating_form", clear_on_submit=False):
    answers: dict[str, Optional[int]] = {}

    for question in QUESTIONS:
        st.markdown(f"**{question['label']}**")
        st.caption(question["help"])

        with st.expander("Definition and anchor examples", expanded=False):
            st.markdown(f"**Definition:** {question['definition']}")
            if question.get("note"):
                st.markdown(f"**Important:** {question['note']}")
            st.markdown(f"**Rate higher when:** {question['rate_higher']}")
            st.markdown(f"**Rate lower when:** {question['rate_lower']}")
            st.markdown("**Anchor examples:**")
            for score in sorted(question["anchors"]):
                st.markdown(f"- **{score}:** {question['anchors'][score]}")

        answers[question["key"]] = st.radio(
            label=question["label"],
            options=[1, 2, 3, 4, 5],
            format_func=lambda value, labels=question["likert_labels"]: labels[value],
            index=None,
            horizontal=False,
            label_visibility="collapsed",
            key=f"radio_{question['key']}_{dialog_row['META_dialog_id']}",
            disabled=not attention_checked,
        )
        st.write("")

    st.markdown("**Optional free comment**")
    free_comment = st.text_area(
        "If you have any additional comments about this dialog or the rating task, you can write them here.",
        placeholder="Optional: write any additional feedback here...",
        key=f"free_comment_{dialog_row['META_dialog_id']}",
        disabled=not attention_checked,
    )

    submitted = st.form_submit_button(
        "Submit ratings",
        type="primary",
        disabled=not attention_checked,
    )

if submitted:
    missing_questions = [
        question["label"] for question in QUESTIONS if answers.get(question["key"]) is None
    ]

    if not attention_checked:
        st.error("Please tick the attention check before submitting.")
    elif missing_questions:
        st.error("Please answer all 8 questions before submitting.")
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
