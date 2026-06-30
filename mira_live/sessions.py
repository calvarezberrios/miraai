"""
Session store for Mira Live — one JSON file per chat session under SESSIONS_DIR.

Powers the sidebar: each "New Chat" makes a fresh session (starts clean, persona re-applied by
the LLM layer), and old sessions stay browsable. Deliberately simple file storage — no DB, no
long-term memory. (Future: this is the hook where real memory could grow.)
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from . import config


def _dir() -> str:
    os.makedirs(config.SESSIONS_DIR, exist_ok=True)
    return config.SESSIONS_DIR


def _path(session_id: str) -> str:
    # session_id is a timestamp-based slug we generate, so it's filesystem-safe.
    return os.path.join(_dir(), f"{session_id}.json")


def new_session() -> Dict:
    sid = time.strftime("%Y%m%d-%H%M%S")
    # de-dup if two land in the same second
    if os.path.exists(_path(sid)):
        sid += f"-{int(time.time() * 1000) % 1000:03d}"
    s = {"id": sid, "title": "New chat", "created": time.time(), "messages": []}
    _write(s)
    return s


def _write(s: Dict) -> None:
    with open(_path(s["id"]), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


def load(session_id: str) -> Optional[Dict]:
    try:
        with open(_path(session_id), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete(session_id: str) -> bool:
    """Remove a session's transcript file. Returns True if it existed and was deleted."""
    p = _path(session_id)
    try:
        if os.path.exists(p):
            os.remove(p)
            return True
    except Exception:
        pass
    return False


def append(session_id: str, role: str, content: str) -> Optional[Dict]:
    s = load(session_id)
    if s is None:
        return None
    s["messages"].append({"role": role, "content": content, "ts": time.time()})
    # Title the session from the first user line so the sidebar is readable.
    if s.get("title") in (None, "", "New chat") and role == "user":
        s["title"] = (content[:48] + "…") if len(content) > 48 else content
    _write(s)
    return s


def list_sessions() -> List[Dict]:
    out = []
    for name in os.listdir(_dir()):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(_dir(), name), "r", encoding="utf-8") as f:
                s = json.load(f)
            out.append({
                "id": s["id"],
                "title": s.get("title") or "New chat",
                "created": s.get("created", 0),
                "count": len(s.get("messages", [])),
            })
        except Exception:
            continue
    out.sort(key=lambda x: x["created"], reverse=True)
    return out
