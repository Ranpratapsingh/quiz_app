"""
modules/session.py
------------------
Manages quiz sessions: creation, persistence (JSON), loading, and result summaries.
Sessions are stored in data/sessions/ as JSON files.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

SESSIONS_DIR = Path("data/sessions")


def _ensure_dir():
    if SESSIONS_DIR.exists() and not SESSIONS_DIR.is_dir():
        raise RuntimeError(
            f"Session path '{SESSIONS_DIR}' exists as a file. Delete it and create a folder with the same name."
        )
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def create_session(filename: str, num_questions: int, model: str) -> str:
    """Create a new session and return its ID."""
    _ensure_dir()
    sid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    session = {
        "id": sid,
        "filename": filename,
        "model": model,
        "num_questions": num_questions,
        "created_at": datetime.now().isoformat(),
        "status": "in_progress",
        "answers": [],
        "score_pct": 0,
    }
    path = SESSIONS_DIR / f"{sid}.json"
    path.write_text(json.dumps(session, indent=2))
    return sid


def save_session(
    session_id: str,
    filename: str,
    answers: list[dict],
    model: str,
    user_name: str = "",
    source_type: str = "",
):
    """Persist the completed session with all answers and computed score."""
    _ensure_dir()
    summary = compute_result_summary(answers)
    session = {
        "id": session_id,
        "filename": filename,
        "model": model,
        "user_name": user_name,
        "source_type": source_type,
        "created_at": datetime.now().isoformat(),
        "status": "completed",
        "answers": answers,
        "score_pct": summary["score_pct"],
        "correct": summary["correct"],
        "total": summary["total"],
        "incorrect": summary["incorrect"],
        "avg_score": summary["avg_score"],
    }
    path = SESSIONS_DIR / f"{session_id}.json"
    path.write_text(json.dumps(session, indent=2))


def load_session(session_id: str) -> dict | None:
    """Load a session by ID. Returns None if not found."""
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_all_sessions() -> list[dict]:
    """Return all completed sessions, sorted by date (oldest first)."""
    _ensure_dir()
    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            s = json.loads(path.read_text())
            if s.get("status") == "completed":
                sessions.append(s)
        except Exception:
            continue
    return sessions


def compute_result_summary(answers: list[dict]) -> dict:
    """Compute aggregate statistics from a list of answer dicts."""
    if not answers:
        return {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "score_pct": 0,
            "avg_score": 0.0,
        }

    total     = len(answers)
    correct   = sum(1 for a in answers if a.get("is_correct", False))
    scores    = [a.get("score", 0) for a in answers]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    score_pct = round((correct / total) * 100) if total else 0

    return {
        "total":     total,
        "correct":   correct,
        "incorrect": total - correct,
        "score_pct": score_pct,
        "avg_score": avg_score,
    }
