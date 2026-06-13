"""
Hypothalamus — the body clock (minimal seed; grows in Phase 7).

Right now it does one thing: persist the time Mira was last talked to, so that
after a restart she knows how long she's been away. Nothing runs while she's shut
down — this only writes a timestamp to disk as activity happens and reads it back
on startup.

State is a tiny JSON file under the (gitignored) memory_store folder.
"""

from __future__ import annotations

import json
import os
import time

_STATE_FILE = os.path.join("memory_store", "clock.json")


def _load() -> dict:
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        tmp = _STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, _STATE_FILE)   # atomic-ish swap
    except Exception:
        pass


def last_active():
    """Epoch time Mira was last talked to (across restarts), or None if never."""
    value = _load().get("last_active")
    return value if isinstance(value, (int, float)) else None


def touch(now=None):
    """Record that activity just happened, and persist it. Call on each message."""
    now = time.time() if now is None else now
    data = _load()
    data["last_active"] = now
    _save(data)
    return now


def time_since_last_active(now=None):
    """Seconds since Mira was last talked to (spanning any downtime), or None."""
    la = last_active()
    if la is None:
        return None
    now = time.time() if now is None else now
    return max(0.0, now - la)