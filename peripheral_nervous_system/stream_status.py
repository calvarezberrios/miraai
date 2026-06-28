"""
Stream status — makes Mira aware of the broadcast itself (live/offline, viewers, game).

This is the cheap, GPU-free half of "stream awareness": it polls Twitch's Helix API
(NOT the chat IRC connection — that can't see viewer counts or online state) and keeps a
small snapshot of the current broadcast. The brain reads summary() each turn and folds it
into her situation, so she knows she's live, how many people are watching, and what game
she's playing — and can react to it ("good to see 40 of you here", "this boss is rough").

It also fires an optional on_transition(kind, info) callback when the stream goes LIVE or
OFFLINE, so the app can have her say something at those moments.

Cost: one HTTP poll every POLL_SEC (default 45s), no GPU, negligible tokens. Auth uses an
APP access token (client-credentials) — enough for public data (stream up/down, viewers,
game, title). Follows/subs/raids would need EventSub + a user token; left as a follow-on.

Config (.env / env):
    TWITCH_CLIENT_ID=...        (required — create an app at dev.twitch.tv/console)
    TWITCH_CLIENT_SECRET=...    (required)
    TWITCH_CHANNEL=yourchannel  (shared with the chat reader)
    MIRA_TWITCH_POLL_SEC=45     (optional)
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Callable, Optional

ENV_FILE = ".env"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
HELIX = "https://api.twitch.tv/helix"
POLL_SEC = float(os.environ.get("MIRA_TWITCH_POLL_SEC", "45"))


def _read_env_file(path, key):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _cfg(key):
    return (os.environ.get(key, "").strip() or _read_env_file(ENV_FILE, key)).strip()


class StreamStatus:
    def __init__(self) -> None:
        self._client_id = ""
        self._secret = ""
        self._channel = ""
        self._user_id = ""
        self._token = ""
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_transition: Optional[Callable[[str, dict], None]] = None
        self._lock = threading.Lock()
        self._state = {"live": None, "viewers": 0, "game": "", "title": "", "started_at": ""}

    # ---------------- lifecycle ----------------
    def start(self, on_transition: Optional[Callable[[str, dict], None]] = None) -> bool:
        self._client_id = _cfg("TWITCH_CLIENT_ID")
        self._secret = _cfg("TWITCH_CLIENT_SECRET")
        self._channel = _cfg("TWITCH_CHANNEL").lstrip("#").lower()
        self._on_transition = on_transition
        if not (self._client_id and self._secret and self._channel):
            print("[stream-status] disabled — set TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, "
                  "TWITCH_CHANNEL in .env to let Mira see live/offline + viewer count.")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="stream-status", daemon=True)
        self._thread.start()
        print(f"[stream-status] watching #{self._channel} (every {POLL_SEC:.0f}s).")
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

    # ---------------- public read ----------------
    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def summary(self) -> str:
        """A one-line note for the brain's situation prompt (empty until the first poll)."""
        with self._lock:
            s = dict(self._state)
        if s["live"] is None:
            return ""
        if not s["live"]:
            return "Your Twitch stream is currently OFFLINE (you are not broadcasting right now)."
        bits = ["You are LIVE on Twitch right now"]
        if s["viewers"]:
            bits.append(f"with {s['viewers']} viewer{'s' if s['viewers'] != 1 else ''} watching")
        line = " ".join(bits) + "."
        if s["game"]:
            line += f" You're streaming {s['game']}."
        if s["title"]:
            line += f' The stream title is "{s["title"]}".'
        return line

    # ---------------- internals ----------------
    def _loop(self) -> None:
        backoff = POLL_SEC
        while self._running:
            try:
                if not self._token and not self._refresh_token():
                    time.sleep(min(backoff, 120)); backoff = min(backoff * 2, 120); continue
                if not self._user_id and not self._resolve_user():
                    time.sleep(min(backoff, 120)); backoff = min(backoff * 2, 120); continue
                self._poll_stream()
                backoff = POLL_SEC
            except Exception as e:
                print(f"[stream-status] poll error: {e}")
            time.sleep(POLL_SEC)

    def _api(self, url: str, headers: dict, data: Optional[bytes] = None) -> dict:
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    def _refresh_token(self) -> bool:
        body = urllib.parse.urlencode({
            "client_id": self._client_id,
            "client_secret": self._secret,
            "grant_type": "client_credentials",
        }).encode()
        try:
            j = self._api(TOKEN_URL, {"Content-Type": "application/x-www-form-urlencoded"}, body)
            self._token = j.get("access_token", "")
            return bool(self._token)
        except Exception as e:
            print(f"[stream-status] auth failed (check client id/secret): {e}")
            return False

    def _headers(self) -> dict:
        return {"Client-Id": self._client_id, "Authorization": f"Bearer {self._token}"}

    def _resolve_user(self) -> bool:
        j = self._api(f"{HELIX}/users?login={urllib.parse.quote(self._channel)}", self._headers())
        data = j.get("data") or []
        if data:
            self._user_id = data[0].get("id", "")
        return bool(self._user_id)

    def _poll_stream(self) -> None:
        j = self._api(f"{HELIX}/streams?user_id={self._user_id}", self._headers())
        data = j.get("data") or []
        if data:
            s = data[0]
            new = {"live": True,
                   "viewers": int(s.get("viewer_count", 0) or 0),
                   "game": s.get("game_name", "") or "",
                   "title": s.get("title", "") or "",
                   "started_at": s.get("started_at", "") or ""}
        else:
            new = {"live": False, "viewers": 0, "game": "", "title": "", "started_at": ""}

        with self._lock:
            was = self._state.get("live")
            self._state = new
        # Fire a transition only on a real edge (skip the very first poll, was=None).
        if was is not None and was != new["live"] and self._on_transition is not None:
            try:
                self._on_transition("live" if new["live"] else "offline", new)
            except Exception as e:
                print(f"[stream-status] transition callback error: {e}")
