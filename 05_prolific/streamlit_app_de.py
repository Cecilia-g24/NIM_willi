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
# folder, named "streamlit_test_sample_de_<N>.csv". The most recently
# generated file is used automatically. Only German dialogs are loaded.
BASE_DIR = Path(__file__).resolve().parent

DIALOG_INPUT_FILENAME_PATTERN = "streamlit_test_sample_de_*.csv"

RESPONSES_DIR = BASE_DIR / "responses"
DB_PATH = RESPONSES_DIR / "survey_responses_de.sqlite"
CSV_EXPORT_PATH = RESPONSES_DIR / "survey_responses_de.csv"

# Prolific completion URL for the German study.
PROLIFIC_COMPLETION_URL = "https://app.prolific.com/submissions/complete?cc=C6JV4KGN"

# Number of dialogs one participant should annotate.
DIALOGS_PER_PARTICIPANT = 3

# Optional quota per dialog. Set to None if you do not want a cap.
TARGET_RATINGS_PER_DIALOG: Optional[int] = None

REQUIRED_COLUMNS = ["META_dialog_id", "META_condition", "language", "subject", "dialog_text"]

ANNOTATION_INSTRUCTION_BLOCK_A = (
    "Bitte lesen Sie den gesamten Dialog und bewerten Sie ausschließlich das Verhalten der "
    "menschlichen Gesprächsperson, wie es im Transkript sichtbar ist. Beurteilen Sie nicht die "
    "Qualität des Roboters direkt und lassen Sie Ihre persönliche Meinung über den Roboter außer "
    "Acht. Konzentrieren Sie sich allein auf das, was die menschliche Gesprächsperson sagt, fragt, "
    "preisgibt und im Verlauf des Gesprächs zum Ausdruck zu bringen scheint."
)

QUESTIONS_BLOCK_A = [
    {
        "key": "user_engagement_enjoyment",
        "label": "1. Nutzerbeteiligung / Gesprächsfreude",
        "help": "Inwieweit wirkt die Person engagiert, interessiert und bereit, das Gespräch fortzusetzen?",
        "definition": (
            "Nutzerbeteiligung / Gesprächsfreude bezieht sich darauf, wie stark beteiligt, interessiert "
            "und bereit die menschliche Gesprächsperson erscheint, das Gespräch fortzusetzen. Beurteilen "
            "Sie dies nur anhand sichtbarer Hinweise im Transkript, wie aktive Beteiligung, Reaktionen, "
            "Fragen, Kooperation, Begeisterung, Zögern oder Versuche, den Austausch zu beenden."
        ),
        "rate_higher": (
            "Die Person beteiligt sich aktiv, gibt bedeutsame Antworten, reagiert auf den Roboter, "
            "stellt Fragen, zeigt Interesse am Thema oder scheint bereit, das Gespräch weiterzuführen."
        ),
        "rate_lower": (
            "Die Person gibt minimale Antworten, wirkt passiv oder gelangweilt, ignoriert die Fragen des "
            "Roboters, weicht dem Thema aus, zeigt wenig Interesse oder versucht, das Gespräch zu beenden."
        ),
        "anchors": {
            1: "Sehr gering/nicht vorhanden: Die Person beteiligt sich kaum, gibt minimale oder ablehnende Antworten oder möchte das Gespräch erkennbar beenden.",
            3: "Mittel: Die Person antwortet dem Roboter, jedoch mit wenig Details, gemischtem Interesse oder gelegentlichem Desengagement.",
            5: "Sehr hoch: Die Person beteiligt sich aktiv, reagiert auf den Roboter und scheint bereit, das Gespräch fortzusetzen.",
        },
    },
    {
        "key": "user_self_disclosure",
        "label": "2. Selbstoffenbarung",
        "help": "Inwieweit gibt die Person persönliche Informationen, Erlebnisse, Vorlieben, Meinungen oder Gefühle preis?",
        "definition": (
            "Selbstoffenbarung bezieht sich darauf, wie viel die Person über sich selbst preisgibt, "
            "einschließlich persönlicher Erlebnisse, Vorlieben, Gefühle, Bewertungen, Erinnerungen, "
            "Meinungen oder alltäglicher Gewohnheiten. Es geht um persönliche Offenheit, nicht um die "
            "bloße Anzahl der Wörter."
        ),
        "rate_higher": (
            "Die Person teilt persönliche Erlebnisse, Vorlieben, Gefühle, Meinungen, Erinnerungen oder "
            "Details aus dem eigenen Leben mit, anstatt nur allgemeine oder sachliche Antworten zu geben."
        ),
        "rate_lower": (
            "Die Person gibt unpersönliche, allgemeine, sachliche oder sehr kurze Antworten und offenbart "
            "wenig oder nichts über sich selbst."
        ),
        "anchors": {
            1: "Sehr gering/nicht vorhanden: Die Person gibt fast nichts Persönliches preis.",
            3: "Mittel: Die Person offenbart einige persönliche Vorlieben, Meinungen oder Erlebnisse, jedoch nur kurz oder gelegentlich.",
            5: "Sehr hoch: Die Person teilt offen persönliche Erlebnisse, Vorlieben, Gefühle oder Meinungen in bedeutsamer Tiefe mit.",
        },
    },
    {
        "key": "user_topical_alignment",
        "label": "3. Thematische Ausrichtung",
        "help": "Inwieweit bleibt die Person beim Thema, das der Roboter einführt oder entwickelt?",
        "definition": (
            "Thematische Ausrichtung bezieht sich darauf, ob die Person mit dem vom Roboter vorgeschlagenen "
            "Thema verbunden bleibt oder es auf relevante Weise weiterentwickelt. Es erfasst die Kooperation "
            "beim laufenden Thema, nicht ob das Thema selbst interessant ist."
        ),
        "rate_higher": (
            "Die Person antwortet in einer Weise, die relevant für die Frage oder das Thema des Roboters "
            "ist, folgt dem aktuellen Thema und entwickelt denselben Gesprächsfaden."
        ),
        "rate_lower": (
            "Die Person gibt themenfremde, unzusammenhängende, ausweichende oder nicht passende Antworten, "
            "lenkt das Gespräch ohne klaren Bezug um oder kooperiert nicht beim aktuellen Thema."
        ),
        "anchors": {
            1: "Sehr gering/nicht vorhanden: Die Person ist überwiegend am falschen Thema, weicht aus oder ist vom Thema des Roboters losgelöst.",
            3: "Mittel: Die Person ist teilweise thematisch ausgerichtet, gibt aber manchmal schwach zusammenhängende, unklare oder umgeleitete Antworten.",
            5: "Sehr hoch: Die Person bleibt durchgehend relevant beim Thema des Roboters und entwickelt denselben Gesprächsfaden.",
        },
    },
    {
        "key": "user_elaboration_informativeness",
        "label": "4. Ausführlichkeit / Informationsgehalt",
        "help": "Inwieweit liefert die Person bedeutsame Details über kurze oder minimale Antworten hinaus?",
        "definition": (
            "Ausführlichkeit / Informationsgehalt bezieht sich darauf, wie viel bedeutsamen Inhalt die "
            "Person liefert. Es erfasst, ob die Antworten der Person nützliche Details, Erklärungen, "
            "Beispiele oder Kontext hinzufügen, anstatt nur minimale Antworten zu geben."
        ),
        "rate_higher": (
            "Die Person gibt informative Antworten mit Details, Begründungen, Beispielen, Erklärungen "
            "oder Kontext, die ihren Beitrag bedeutsam machen."
        ),
        "rate_lower": (
            "Die Person gibt kurze, vage, repetitive oder minimale Antworten, wie Ja/Nein-Antworten oder "
            "kurze Fragmente, mit wenig bedeutsamen Details."
        ),
        "anchors": {
            1: "Sehr gering/nicht vorhanden: Die Person liefert kaum bedeutsame Details über minimale Antworten hinaus.",
            3: "Mittel: Die Person liefert einige nützliche Details, viele Antworten bleiben jedoch kurz oder nur teilweise informativ.",
            5: "Sehr hoch: Die Person liefert reiche, bedeutsame und informative Details im gesamten Gespräch.",
        },
    },
    {
        "key": "user_initiative_active_contribution",
        "label": "5. Initiative / Aktiver Beitrag",
        "help": "Inwieweit trägt die Person aktiv zur Entwicklung des Gesprächs bei, zum Beispiel durch Fragen, neue Themen, Meinungsäußerungen oder die Steuerung des Austauschs?",
        "definition": (
            "Initiative / Aktiver Beitrag bezieht sich darauf, ob die Person mehr tut, als nur auf die "
            "Aufforderungen des Roboters zu reagieren. Es erfasst die aktive Beteiligung an der Gestaltung "
            "des Austauschs, wie das Stellen von Fragen, das Einführen verwandter Ideen, das Äußern von "
            "Meinungen oder das Lenken des Gesprächs."
        ),
        "rate_higher": (
            "Die Person stellt Fragen, führt Themen ein oder entwickelt sie, äußert Meinungen, reagiert "
            "proaktiv, treibt das Gespräch voran oder hilft anderweitig, den Austausch zu gestalten."
        ),
        "rate_lower": (
            "Die Person reagiert nur auf direkte Aufforderungen, gibt passive oder minimale Antworten, "
            "fügt selten etwas Neues hinzu und hilft nicht, das Gespräch zu entwickeln."
        ),
        "anchors": {
            1: "Sehr gering/nicht vorhanden: Die Person ist fast vollständig passiv und gibt nur minimale Antworten auf Aufforderungen.",
            3: "Mittel: Die Person fügt gelegentlich etwas Neues hinzu oder äußert eine Meinung, folgt aber überwiegend der Führung des Roboters.",
            5: "Sehr hoch: Die Person gestaltet den Austausch aktiv, indem sie Fragen stellt, Themen einbringt oder das Gespräch lenkt.",
        },
    },
    {
        "key": "user_politeness",
        "label": "6. Höflichkeit",
        "help": "Inwieweit spricht die Person den Roboter auf höfliche, respektvolle oder sozial zugewandte Weise an?",
        "definition": (
            "Höflichkeit bezieht sich darauf, ob die Person den Roboter als sozial ansprechbaren "
            "Interaktionspartner behandelt. Dazu können höfliche Formulierungen, respektvolle Antworten, "
            "Begrüßungen, Dankesworte, freundliche Kommentare oder andere zugewandte Sprache gehören."
        ),
        "rate_higher": (
            "Die Person verwendet höfliche, respektvolle, freundliche, wertschätzende oder sozial "
            "zugewandte Sprache gegenüber dem Roboter, wie Begrüßungen, Dankesworte, abschwächende "
            "Formulierungen oder kooperative Ausdrücke."
        ),
        "rate_lower": (
            "Die Person ist direkt, abweisend, unhöflich, respektlos, sozial distanziert oder zeigt "
            "keine Anzeichen dafür, den Roboter als sozialen Interaktionspartner zu behandeln."
        ),
        "anchors": {
            1: "Sehr gering/nicht vorhanden: Die Person ist unhöflich, abweisend oder zeigt kein höfliches oder sozial zugewandtes Verhalten.",
            3: "Mittel: Die Person ist im Allgemeinen neutral oder minimal höflich, mit begrenzter sozialer Wärme.",
            5: "Sehr hoch: Die Person ist klar höflich, respektvoll, freundlich oder sozial zugewandt gegenüber dem Roboter.",
        },
    },
    {
        "key": "user_frustration_dissatisfaction",
        "label": "7. Frustration / Unzufriedenheit",
        "help": "Inwieweit wirkt die Person frustriert, gereizt, ungeduldig, verwirrt, enttäuscht oder unwillig, das Gespräch fortzusetzen?",
        "definition": (
            "Frustration / Unzufriedenheit bezieht sich auf sichtbare negative Reaktionen der Person "
            "während des Dialogs. Dazu gehören Ärger, Ungeduld, Verwirrung, Enttäuschung, Gereiztheit, "
            "Widerstand oder Signale, dass die Person das Gespräch nicht fortsetzen möchte."
        ),
        "rate_higher": (
            "Die Person wirkt gereizt, ungeduldig, verwirrt, enttäuscht, irritiert, widerstrebend, "
            "unzufrieden oder unwillig, das Gespräch fortzusetzen."
        ),
        "rate_lower": (
            "Die Person zeigt wenig oder keine sichtbare Frustration, Ungeduld, Verwirrung, "
            "Unzufriedenheit, Widerstand oder den Wunsch, das Gespräch zu beenden."
        ),
        "anchors": {
            1: "Sehr gering/nicht vorhanden: Die Person zeigt wenig oder keine Frustration, Unzufriedenheit oder Unwilligkeit fortzufahren.",
            3: "Mittel: Die Person zeigt etwas Verwirrung, Ungeduld, Unzufriedenheit oder Zögern, aber nicht durchgehend.",
            5: "Sehr hoch: Die Person wirkt klar frustriert, gereizt, unzufrieden oder unwillig fortzufahren.",
        },
    },
]


QUESTIONS = QUESTIONS_BLOCK_A


# ============================================================
# Basic setup
# ============================================================

st.set_page_config(
    page_title="Beobachterrating des Nutzerverhaltens in der MRI-Studie",
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
    """Find the most recently generated create_test_samples.py output for German."""
    matches = sorted(BASE_DIR.glob(DIALOG_INPUT_FILENAME_PATTERN), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


@st.cache_data(show_spinner=False)
def load_dialogs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Eingabe-CSV fehlen erforderliche Spalten: {missing}")

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
            existing_columns.add("free_comment")
        if "condition" not in existing_columns:
            conn.execute("ALTER TABLE responses ADD COLUMN condition TEXT")
            existing_columns.add("condition")

        # Add columns for any newly added or renamed questions.
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
            "participant_id", "study_id", "session_id", "dialog_id",
            "language", "subject", "condition", "submitted_at_utc", "free_comment",
        }
        current_keys = {q["key"] for q in QUESTIONS}
        stale_notnull = [r for r in col_rows if r[3] and r[1] not in current_keys and r[1] not in non_question]

        if stale_notnull:
            required_notnull = {"participant_id", "dialog_id", "submitted_at_utc"}
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
    params: dict[str, str] = {}

    if prolific_pid:
        params["PROLIFIC_PID"] = prolific_pid
        if study_id:
            params["STUDY_ID"] = study_id
        if session_id:
            params["SESSION_ID"] = session_id
    else:
        params["TEST_PID"] = participant_id

    params["reload"] = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return "?" + urlencode(params) + "#page-top"


def parse_dialog_to_blocks(dialog_text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_role = "Kontext"
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
        if lower.startswith("robot:") or lower.startswith("roboter:"):
            flush()
            current_role = "Roboter"
            buffer = [stripped.split(":", 1)[1].strip()]
        elif lower.startswith("user:") or lower.startswith("visitor:") or lower.startswith("besucher:"):
            flush()
            current_role = "Besucher"
            buffer = [stripped.split(":", 1)[1].strip()]
        else:
            buffer.append(stripped)

    flush()
    return blocks


def render_dialog(dialog_text: str) -> None:
    """Render the dialog as clean Roboter/Besucher cards."""
    blocks = parse_dialog_to_blocks(dialog_text)
    html_blocks: list[str] = []

    for role, content in blocks:
        if role == "Roboter":
            css_class = "qd-turn-robot"
        elif role == "Besucher":
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
st.title("Beobachterrating des Nutzerverhaltens in der MRI-Studie (Deutsch)")
st.caption(
    f"Bitte lesen Sie jeden Dialog sorgfältig und beantworten Sie alle {len(QUESTIONS)} Fragen "
    f"zum Verhalten der menschlichen Gesprächsperson."
)

de_csv_path = find_latest_dialog_csv()

if de_csv_path is None:
    st.error(
        "Es wurde keine Dialog-CSV in diesem Ordner gefunden, die dem Muster "
        f"`{DIALOG_INPUT_FILENAME_PATTERN}` entspricht. "
        "Bitte führen Sie zunächst create_test_samples.py aus, um diese Datei zu erstellen."
    )
    st.stop()

try:
    dialogs = load_dialogs(str(de_csv_path))
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
    st.sidebar.header("Nur für Tests")
    participant_id = st.sidebar.text_input(
        "Teilnehmer-ID",
        value=test_pid,
        placeholder="Test-ID eingeben",
    ).strip()

    st.sidebar.divider()
    st.sidebar.write(f"Eingabezeilen: **{len(dialogs)}**")
    st.sidebar.write(f"Geladen: `{de_csv_path.name}`")
    st.sidebar.write(f"Antwortdatei: `{CSV_EXPORT_PATH}`")

if not participant_id:
    st.warning("Bitte geben Sie eine Teilnehmer-ID ein, um zu beginnen.")
    st.stop()

if st.session_state.pop("saved_previous_dialog", False):
    st.success("Vorheriger Dialog gespeichert. Bitte fahren Sie mit dem nächsten Dialog fort.")

completed_count = participant_completed_count(participant_id)

if completed_count >= DIALOGS_PER_PARTICIPANT:
    st.success("Alle erforderlichen Dialogbewertungen wurden aufgezeichnet. Vielen Dank.")
    if PROLIFIC_COMPLETION_URL:
        st.markdown(f"[Zurück zu Prolific]({PROLIFIC_COMPLETION_URL})")
    st.stop()

st.progress(completed_count / DIALOGS_PER_PARTICIPANT)
st.markdown(f"**Fortschritt:** Dialog {completed_count + 1} von {DIALOGS_PER_PARTICIPANT}")

dialog_row = assign_dialog(dialogs, participant_id, requested_dialog_id)
if dialog_row is None:
    st.warning("Derzeit ist kein Dialog zur Bewertung verfügbar.")
    st.stop()
assert dialog_row is not None

GENERIC_LIKERT_LABELS = {
    1: "1 — Sehr gering / nicht vorhanden",
    2: "2 — Gering",
    3: "3 — Mittel",
    4: "4 — Hoch",
    5: "5 — Sehr hoch",
}


def render_question(
    question: dict,
    answers: dict[str, Optional[int]],
    question_number: int,
) -> None:
    visible_question = f"{question_number}. {question['help']}"

    st.markdown(f"**{visible_question}**")

    with st.expander("Bewertungshinweise", expanded=False):
        st.markdown(f"**Höher bewerten, wenn:** {question['rate_higher']}")
        st.markdown(f"**Niedriger bewerten, wenn:** {question['rate_lower']}")

        if question.get("note"):
            st.markdown(f"**Wichtig:** {question['note']}")

        st.markdown("**Skalenhinweise:**")
        st.markdown("- **1:** Das in der Frage beschriebene Verhalten ist sehr gering oder nicht vorhanden.")
        st.markdown("- **3:** Das in der Frage beschriebene Verhalten ist moderat.")
        st.markdown("- **5:** Das in der Frage beschriebene Verhalten ist sehr stark ausgeprägt.")

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

    st.markdown(ANNOTATION_INSTRUCTION_BLOCK_A)

    render_dialog(str(dialog_row["dialog_text"]))

    for i, question in enumerate(QUESTIONS_BLOCK_A, start=1):
        render_question(question, answers, i)

    st.markdown("**Optionaler Kommentar**")
    free_comment = st.text_area(
        "Wenn Sie weitere Anmerkungen zu diesem Dialog oder zur Bewertungsaufgabe haben, können Sie diese hier eintragen.",
        placeholder="Optional: Schreiben Sie hier zusätzliche Anmerkungen...",
        key=f"free_comment_{dialog_row['META_dialog_id']}",
    )

    submitted = st.form_submit_button(
        "Bewertungen absenden",
        type="primary",
    )

if submitted:
    missing_questions = [
        question["label"] for question in QUESTIONS if answers.get(question["key"]) is None
    ]

    if missing_questions:
        st.error(f"Bitte beantworten Sie alle {len(QUESTIONS)} Fragen, bevor Sie absenden.")
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
                st.success("Alle erforderlichen Dialogbewertungen wurden aufgezeichnet. Vielen Dank.")
                if PROLIFIC_COMPLETION_URL:
                    st.markdown(f"[Zurück zu Prolific]({PROLIFIC_COMPLETION_URL})")
            else:
                continue_url = make_continue_url(participant_id, prolific_pid, study_id, session_id)
                st.success("Vorheriger Dialog gespeichert. Bitte fahren Sie mit dem nächsten Dialog fort.")
                st.markdown(
                    f'<a class="continue-button" href="{html.escape(continue_url)}" target="_self">Weiter zum nächsten Dialog</a>',
                    unsafe_allow_html=True,
                )
                st.stop()
        else:
            continue_url = make_continue_url(participant_id, prolific_pid, study_id, session_id)
            st.info("Ihre Antwort für diesen Dialog wurde bereits früher gespeichert. Bitte fahren Sie mit dem nächsten verfügbaren Dialog fort.")
            st.markdown(
                f'<a class="continue-button" href="{html.escape(continue_url)}" target="_self">Weiter zum nächsten Dialog</a>',
                unsafe_allow_html=True,
            )
            st.stop()
