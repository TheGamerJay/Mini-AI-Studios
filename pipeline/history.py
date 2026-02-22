"""
History Manager — persists song generation history to JSON.
"""
import json
import os
from datetime import datetime

from config import HISTORY_FILE, MAX_HISTORY


def add(entry: dict):
    """Prepend a new entry to history, trim to MAX_HISTORY, save."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    history = load()
    history.insert(0, {**entry, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
    history = history[:MAX_HISTORY]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load() -> list:
    """Return full history list, newest first."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def clear():
    """Delete the history file."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


def to_rows(history: list) -> list:
    """Convert history list → list of rows for gr.Dataframe."""
    rows = []
    for e in history:
        path     = e.get("path", "")
        filename = os.path.basename(path) if path else "—"
        prompt   = e.get("prompt", "")
        display  = prompt[:55] + "…" if len(prompt) > 55 else prompt
        rows.append([
            e.get("timestamp", ""),
            display,
            e.get("genre", ""),
            f'{e.get("duration", "")}s',
            e.get("voice", ""),
            filename,
        ])
    return rows
