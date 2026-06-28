"""
Twitch adapter — reads live Twitch chat (IRC), responds by voice/avatar.

TOKEN EFFICIENCY (the whole point of this adapter)
  Reading chat is FREE: this connects to Twitch's IRC over a plain socket with an
  anonymous (justinfan) login — no OAuth, no API tokens, no pip dependency. The LLM
  only ever runs when she actually decides to SPEAK, so the firehose is gated BEFORE
  the brain ever sees it:

    1. cheap pre-filters drop bot / "!command" / link / emote-only / tiny messages
       with NO model call (pure string work in this file).
    2. a message that NAMES her ("mira", "@mira") is answered directly (the normal
       "addressed" path) — these bypass the digest so callouts feel responsive.
    3. everything else is buffered and, every MIRA_TWITCH_DIGEST_SEC, condensed into
       ONE digest (capped to the last N lines so the prompt stays small). She reacts
       to it via consider_speaking — a single model call per window, and she may stay
       silent. So a 200-message minute costs at most ~1-2 model calls.

ON STREAM SHE ALSO HEARS YOU
  A streamer talks to her too, so the local mic (wernickes_area) runs alongside chat:
  the streamer is the local speaker (always addressed, exactly like local mode) while
  chat scrolls in the background. Set MIRA_TWITCH_NO_MIC=1 for chat-only (headless).

OUTPUT IS VOICE/AVATAR ONLY
  She speaks her replies through the same mouth as local mode (brocas_area). She does
  NOT post text back into Twitch chat — the connection is read-only, so no bot account
  or OAuth token is needed.

Threading: a single worker thread drains one inbox and feeds the brain one turn at a
time (mic finals + chat turns serialized -> one model call at a time, 6GB-VRAM safe).
Mic partials/prefill/interrupt go straight through for low-latency captions/drafting,
mirroring the Discord adapter.

Config (.env in the project root, or env vars):
    TWITCH_CHANNEL=yourchannel        (required — the channel whose chat to read)
    MIRA_TWITCH_DIGEST_SEC=40         (optional, 30-50 is a good range)
    MIRA_TWITCH_DIGEST_MAX_LINES=40   (optional — cap lines per digest, bounds tokens)
    MIRA_TWITCH_NO_MIC=1              (optional — chat only, don't open the local mic)
"""

from __future__ import annotations

import os
import queue
import re
import socket
import threading
import time
from typing import Optional

from .io_adapter import IOAdapter, InputEvent, OnEvent, FINAL, PARTIAL, INTERRUPT, PREFILL

# ---- config ----------------------------------------------------------------
ENV_FILE = ".env"
CHANNEL_ENV = "TWITCH_CHANNEL"

IRC_HOST = "irc.chat.twitch.tv"
IRC_PORT = 6667                      # plain IRC; anonymous read needs no TLS/OAuth

DIGEST_SEC = float(os.environ.get("MIRA_TWITCH_DIGEST_SEC", "40"))
DIGEST_MAX_LINES = int(os.environ.get("MIRA_TWITCH_DIGEST_MAX_LINES", "40"))
DIGEST_MIN_LINES = 3                 # don't bother her with a near-empty digest
NO_MIC = os.environ.get("MIRA_TWITCH_NO_MIC", "0").strip().lower() not in ("0", "false", "no", "")

OWNER = "You"                        # the local streamer (mic), always addressing her

# She answers to her name in chat (matches action_selector.NAME). A named line skips the
# digest and is answered directly; the brain's should_respond also detects this name.
_NAME_RE = re.compile(r"\bmira\b", re.IGNORECASE)

# Chat bots whose messages are never worth a model call.
KNOWN_BOTS = {
    "nightbot", "streamelements", "streamlabs", "moobot", "fossabot", "soundalerts",
    "wizebot", "botisimo", "commanderroot", "pretzelrocks", "streamcaptainbot",
}

_STOP = object()


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


def _load_channel():
    ch = os.environ.get(CHANNEL_ENV, "").strip()
    if not ch:
        ch = _read_env_file(ENV_FILE, CHANNEL_ENV)
    return ch.lstrip("#").strip().lower()


# A bare emote/spam line ("LUL", "Kappa Kappa", "!!!", "🐸🐸") carries no content worth a
# reply. We keep only lines with at least two word-ish tokens, unless they name her.
_WORD_RE = re.compile(r"[A-Za-z0-9']{2,}")


def _worth_keeping(user: str, text: str) -> bool:
    """Cheap pre-filter (no model call). True if this chat line is worth buffering."""
    if not text:
        return False
    if user.lower() in KNOWN_BOTS:
        return False
    if text.startswith("!"):                       # chat command (e.g. !uptime)
        return False
    if "http://" in text or "https://" in text:    # links / self-promo
        return False
    if _NAME_RE.search(text):                       # named her -> always keep
        return True
    return len(_WORD_RE.findall(text)) >= 2          # else need a little substance


# IRC PRIVMSG line: ":nick!nick@nick.tmi.twitch.tv PRIVMSG #channel :the message"
_PRIVMSG_RE = re.compile(r"^:(?P<nick>[^!]+)![^ ]+ PRIVMSG #[^ ]+ :(?P<msg>.*)$")


class TwitchAdapter(IOAdapter):
    name = "twitch"

    def __init__(self, chat_only: bool = False) -> None:
        # chat_only: run as a pure chat INPUT source — no local mic, and speak()/wait/pause
        # are no-ops. Used by the composite (Discord owns the mic and all voice output, so
        # her chat replies are heard in the VC and therefore on the stream).
        self._chat_only = chat_only
        self._on_event: Optional[OnEvent] = None
        self._channel = ""
        self._running = False
        self._sock: Optional[socket.socket] = None

        self._inbox = queue.Queue()              # serialized FINAL turns -> brain
        self._work_thread: Optional[threading.Thread] = None
        self._net_thread: Optional[threading.Thread] = None
        self._digest_thread: Optional[threading.Thread] = None

        self._buf = []                            # [(user, text)] awaiting the next digest
        self._buf_lock = threading.Lock()

        self._ears = None                         # wernickes_area (local mic), if enabled
        self._mouth = None                        # brocas_area (voice/avatar out)

    # ---------------- lifecycle ----------------
    def start(self, on_event: OnEvent) -> None:
        self._on_event = on_event
        self._channel = _load_channel()
        if not self._channel:
            print(
                f"\n[Twitch] No channel configured.\n"
                f"  Set the channel to read one of these ways:\n"
                f"    - a line in .env:   {CHANNEL_ENV}=yourchannel\n"
                f"    - the {CHANNEL_ENV} environment variable\n"
            )
            raise SystemExit(1)

        self._running = True

        # one worker: drains finals (mic + chat) one at a time -> one model call at a time
        self._work_thread = threading.Thread(target=self._process, daemon=True)
        self._work_thread.start()

        # local mic so the streamer can talk to her too (best-effort; chat works without it).
        # Skipped in chat_only mode — the composite's Discord adapter owns the mic/voice.
        if not NO_MIC and not self._chat_only:
            try:
                from brain.forebrain.cerebrum.temporal_lobe import wernickes_area as ears
                from brain.forebrain.cerebrum.frontal_lobe import brocas_area as mouth
                self._ears = ears
                self._mouth = mouth
                ears.start(
                    on_final=self._mic_final,
                    on_partial=self._mic_partial,
                    on_interrupt=self._mic_interrupt,
                    on_prefill=self._mic_prefill,
                )
            except Exception as e:
                print(f"[Twitch] local mic unavailable ({e}); running chat-only.")
                self._ears = None
        if self._mouth is None and not self._chat_only:
            try:
                from brain.forebrain.cerebrum.frontal_lobe import brocas_area as mouth
                self._mouth = mouth
            except Exception as e:
                print(f"[Twitch] voice output unavailable: {e}")

        # network + digest pump
        self._net_thread = threading.Thread(target=self._irc_loop, daemon=True)
        self._net_thread.start()
        self._digest_thread = threading.Thread(target=self._digest_loop, daemon=True)
        self._digest_thread.start()
        print(f"[Twitch] reading #{self._channel} — addressed lines answered directly, "
              f"the rest digested every {DIGEST_SEC:.0f}s (read-only, no tokens to listen).\n")

    def stop(self) -> None:
        self._running = False
        self._inbox.put(_STOP)
        if self._ears is not None:
            try:
                self._ears.stop()
            except Exception:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        for t in (self._net_thread, self._digest_thread, self._work_thread):
            if t is not None:
                t.join(timeout=5)

    def warmup(self) -> None:
        # Heat the ears (STT) up front so the streamer's first words transcribe instantly.
        # No-op if the mic is disabled / unavailable / chat-only (the composite's Discord
        # adapter warms Whisper instead). (The mouth is warmed by startup.)
        if self._chat_only:
            return
        if self._ears is None and not NO_MIC:
            try:
                from brain.forebrain.cerebrum.temporal_lobe import wernickes_area as ears
                self._ears = ears
            except Exception:
                return
        if self._ears is not None:
            try:
                self._ears.warmup()
            except Exception:
                pass

    # ---------------- mic passthrough (mirrors LocalAdapter) ----------------
    def _emit(self, ev: InputEvent) -> None:
        if self._on_event is not None:
            self._on_event(ev)

    def _mic_final(self, text: str) -> None:
        # serialize with chat turns through the single worker
        self._inbox.put(("mic", text))

    def _mic_partial(self, text: str) -> None:
        self._emit(InputEvent(text=text, speaker=OWNER, kind=PARTIAL, channel="local",
                              mentioned=True))

    def _mic_interrupt(self, text: str) -> None:
        self._emit(InputEvent(text=text, speaker=OWNER, kind=INTERRUPT, channel="local",
                              mentioned=True))

    def _mic_prefill(self, text: str) -> None:
        self._emit(InputEvent(text=text, speaker=OWNER, kind=PREFILL, channel="local",
                              mentioned=True))

    def pause_input(self) -> None:
        if self._ears is not None:
            self._ears.pause()        # don't let her transcribe her own TTS; chat keeps reading

    def resume_input(self) -> None:
        if self._ears is not None:
            self._ears.resume()

    def flush_input(self) -> None:
        if self._ears is not None:
            self._ears.flush()

    # ---------------- IRC (chat in) ----------------
    def _irc_loop(self):
        """Connect to Twitch IRC anonymously and read chat, reconnecting on drop."""
        backoff = 1.0
        nick = f"justinfan{int(time.time()) % 100000}"   # anonymous read-only login
        while self._running:
            try:
                sock = socket.create_connection((IRC_HOST, IRC_PORT), timeout=20)
                self._sock = sock
                sock.settimeout(330)   # Twitch pings ~every 5 min; this just bounds a dead read
                # anonymous login (no PASS needed); request tags for display names
                sock.sendall(b"CAP REQ :twitch.tv/tags\r\n")
                sock.sendall(f"NICK {nick}\r\n".encode())
                sock.sendall(f"JOIN #{self._channel}\r\n".encode())
                backoff = 1.0
                buf = b""
                while self._running:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break                       # server closed -> reconnect
                    buf += chunk
                    while b"\r\n" in buf:
                        line, buf = buf.split(b"\r\n", 1)
                        self._handle_irc_line(sock, line.decode("utf-8", "replace"))
            except Exception as e:
                if self._running:
                    print(f"[Twitch] connection lost ({e}); reconnecting in {backoff:.0f}s")
            finally:
                try:
                    if self._sock is not None:
                        self._sock.close()
                except Exception:
                    pass
                self._sock = None
            if self._running:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle_irc_line(self, sock, line: str):
        if line.startswith("PING"):
            sock.sendall(b"PONG :tmi.twitch.tv\r\n")
            return
        # strip IRCv3 tags prefix if present (we don't need them; nick comes from the source)
        body = line
        if body.startswith("@"):
            sp = body.find(" ")
            body = body[sp + 1:] if sp != -1 else body
        m = _PRIVMSG_RE.match(body)
        if not m:
            return
        user = m.group("nick").strip()
        text = m.group("msg").strip()
        if not _worth_keeping(user, text):
            return
        if _NAME_RE.search(text):
            # named her -> answer directly (skip the digest); single worker serializes it
            self._inbox.put(("chat_addressed", user, text))
        else:
            with self._buf_lock:
                self._buf.append((user, text))

    # ---------------- digest (batched chime-in) ----------------
    def _digest_loop(self):
        """Every DIGEST_SEC, condense buffered chat into ONE turn the brain reacts to
        (consider_speaking). Capped to the last N lines so the prompt — and the token
        cost — stays bounded no matter how fast chat is moving."""
        while self._running:
            time.sleep(DIGEST_SEC)
            if not self._running:
                break
            with self._buf_lock:
                pending = self._buf
                self._buf = []
            if len(pending) < DIGEST_MIN_LINES:
                # too little to be worth a model call — fold it back so it can still
                # accrue into the next window rather than being dropped.
                with self._buf_lock:
                    self._buf[:0] = pending
                continue
            lines = pending[-DIGEST_MAX_LINES:]
            digest = "Live Twitch chat (recent messages):\n" + "\n".join(
                f"{u}: {t}" for u, t in lines)
            self._inbox.put(("digest", digest, len(lines)))

    # ---------------- worker: one turn at a time -> brain ----------------
    def _process(self):
        while self._running:
            item = self._inbox.get()
            try:
                if item is _STOP:
                    break
                kind = item[0]
                if kind == "mic":
                    _, text = item
                    if text:
                        self._emit(InputEvent(text=text, speaker=OWNER, kind=FINAL,
                                              channel="local", mentioned=True))
                elif kind == "chat_addressed":
                    _, user, text = item
                    self._emit(InputEvent(text=text, speaker=user, kind=FINAL,
                                          channel="twitch_chat", mentioned=True))
                elif kind == "digest":
                    _, digest, _n = item
                    # mentioned=False -> brain runs the chime-in (consider_speaking) path,
                    # so she reacts to the vibe or stays silent (one model call, maybe zero).
                    self._emit(InputEvent(text=digest, speaker="chat", kind=FINAL,
                                          channel="twitch_chat", mentioned=False))
            except Exception as e:
                print(f"[Twitch] turn error: {e}")
            finally:
                self._inbox.task_done()

    # ---------------- mouth (voice/avatar only) ----------------
    def speak(self, text: str) -> None:
        if self._mouth is not None and text:
            self._mouth.say(text)

    def wait_until_done(self) -> None:
        if self._mouth is not None:
            self._mouth.wait_until_done()
