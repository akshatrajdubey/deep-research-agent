"""
Session management: persists sessions, conversation history, and turn history using SQLite.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "sessions.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT DEFAULT 'New Session'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                search_queries TEXT NOT NULL,
                urls_opened TEXT NOT NULL,
                context_snippets TEXT NOT NULL,
                final_answer TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
        """)


def create_session(title: str = "New Session") -> str:
    """Create a new session and return its ID."""
    init_db()
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            (session_id, now, now, title)
        )
    return session_id


def list_sessions() -> list[dict]:
    """Return all sessions ordered by most recently updated."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def update_session_title(session_id: str, title: str):
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (title, now, session_id)
        )


def delete_session(session_id: str):
    with _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


# --- Message history ---

def add_message(session_id: str, role: str, content: str):
    """Append a message to the session's conversation history."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now)
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id)
        )


def get_messages(session_id: str) -> list[dict]:
    """Return all messages for a session, ordered by insertion."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# --- Turn history ---

def save_turn(
    session_id: str,
    query: str,
    search_queries: list[str],
    urls_opened: list[str],
    context_snippets: list[dict],
    final_answer: str,
):
    """Save a complete research turn."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO turns
               (session_id, query, search_queries, urls_opened, context_snippets, final_answer, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                query,
                json.dumps(search_queries),
                json.dumps(urls_opened),
                json.dumps(context_snippets),
                final_answer,
                now,
            )
        )


def get_turns(session_id: str) -> list[dict]:
    """Return all research turns for a session."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["search_queries"] = json.loads(d["search_queries"])
        d["urls_opened"] = json.loads(d["urls_opened"])
        d["context_snippets"] = json.loads(d["context_snippets"])
        result.append(d)
    return result
