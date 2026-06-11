"""
Discord adapter — text chat (Step 1).

Connects Mira to Discord as a bot. Every human text message she can see is fed
through the SAME brain as the local loop as one InputEvent, carrying the signals
the brain needs to decide whether to reply (was she @-mentioned, is it a DM, is
it a reply to her). The brain (basal ganglia) makes the respond/stay-quiet call —
the adapter is just transport. Her reply is sent back to the originating channel.

The discord library is imported LAZILY inside start(), so merely importing this
module (or the package) never requires discord.py to be installed.

Threading model:
  - The Discord client runs on its own asyncio loop in a background thread.
  - on_message just enqueues the message (non-blocking) onto a plain Queue.
  - A single worker thread drains that queue one item at a time, so turns are
    serialized and the "reply goes here" channel can't be raced.
  - speak() sends the reply back via run_coroutine_threadsafe on the client loop.

Config:
  - Bot token from .env (DISCORD_BOT_TOKEN), the env var, or ./discord_token.txt
  - Optional ONLY_CHANNELS allowlist (channel names) to limit where she replies.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
from typing import Optional

from .io_adapter import IOAdapter, InputEvent, OnEvent, FINAL

TOKEN_ENV = "DISCORD_BOT_TOKEN"
TOKEN_FILE = "discord_token.txt"
ENV_FILE = ".env"

# If non-empty, Mira only replies in text channels whose name is in this set.
# Leave empty to reply in every text channel she can read. DMs always pass.
# Example: ONLY_CHANNELS = {"mira-chat"}
ONLY_CHANNELS = set()

_STOP = object()  # sentinel to stop the worker


def _read_env_file(path, key):
    """Minimal .env parser — returns the value for `key`, or "". No dependency."""
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


def _load_token():
    # 1) real environment variable
    token = os.environ.get(TOKEN_ENV, "").strip()
    # 2) .env file (accepts DISCORD_BOT_TOKEN, falls back to DISCORD_TOKEN)
    if not token:
        token = _read_env_file(ENV_FILE, TOKEN_ENV) or _read_env_file(ENV_FILE, "DISCORD_TOKEN")
    # 3) plain token file
    if not token and os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            token = f.read().strip()
    return token


class DiscordAdapter(IOAdapter):
    name = "discord"

    def __init__(self) -> None:
        self._on_event: Optional[OnEvent] = None
        self._client = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._net_thread: Optional[threading.Thread] = None
        self._work_thread: Optional[threading.Thread] = None
        self._inbox = queue.Queue()
        self._current_channel = None   # where the in-progress reply should go
        self._running = False

    # ----- lifecycle -----
    def start(self, on_event: OnEvent) -> None:
        self._on_event = on_event

        token = _load_token()
        if not token:
            print(
                f"\n[Discord] No bot token found.\n"
                f"  Provide it one of these ways:\n"
                f"    - a line in .env:   {TOKEN_ENV}=your_token_here\n"
                f"    - the {TOKEN_ENV} environment variable\n"
                f"    - a file '{TOKEN_FILE}' in the project root\n"
                f"  (Make sure .env / the token file are in .gitignore.)\n"
            )
            raise SystemExit(1)

        try:
            import discord  # lazy: only needed in Discord mode
        except ImportError:
            print(
                "\n[Discord] discord.py isn't installed.\n"
                "  Run:  python -m pip install -U discord.py\n"
            )
            raise SystemExit(1)

        self._discord = discord

        intents = discord.Intents.default()
        intents.message_content = True   # PRIVILEGED — must ALSO be enabled in the portal
        client = discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_ready():
            print(f"[Discord] Connected as {client.user} — Mira is listening.\n")

        @client.event
        async def on_message(message):
            # ignore herself and other bots
            if message.author == client.user or message.author.bot:
                return

            content = (message.clean_content or "").strip()
            if not content:
                return

            is_dm = message.guild is None

            # optional per-channel allowlist (guild channels only; DMs always pass)
            if ONLY_CHANNELS and not is_dm:
                ch_name = getattr(message.channel, "name", None)
                if ch_name not in ONLY_CHANNELS:
                    return

            # addressing signals the brain's action selector needs
            mentioned = (client.user in message.mentions) or is_dm
            reply_to_her = False
            ref = message.reference
            if ref is not None:
                resolved = getattr(ref, "resolved", None)
                if resolved is not None and getattr(resolved, "author", None) == client.user:
                    reply_to_her = True

            speaker = getattr(message.author, "display_name", None) or str(message.author)
            # transport only — the brain decides whether to answer
            self._inbox.put((message.channel, speaker, content, mentioned, reply_to_her))

        # single worker that feeds the brain one message at a time
        self._running = True
        self._work_thread = threading.Thread(target=self._process, daemon=True)
        self._work_thread.start()

        # run the client on its own loop in a background thread
        self._loop = asyncio.new_event_loop()

        def _runner():
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(client.start(token))
            except Exception as e:
                print(f"[Discord] connection ended: {e}")

        self._net_thread = threading.Thread(target=_runner, daemon=True)
        self._net_thread.start()

    def stop(self) -> None:
        self._running = False
        self._inbox.put(_STOP)

        # Close the Discord client on its own loop and WAIT for it to finish,
        # so we don't exit the process while close() tasks are still pending
        # (the "Task was destroyed but it is pending" warnings).
        if self._client is not None and self._loop is not None and self._loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
                fut.result(timeout=10)
            except Exception:
                pass

        # Let the network thread's run loop unwind, then join both threads.
        if self._net_thread is not None:
            self._net_thread.join(timeout=10)
        if self._work_thread is not None:
            self._work_thread.join(timeout=5)

    # ----- worker: queue -> brain -----
    def _process(self) -> None:
        while self._running:
            item = self._inbox.get()
            if item is _STOP:
                break
            channel, speaker, text, mentioned, reply_to_her = item
            self._current_channel = channel
            try:
                if self._on_event is not None:
                    self._on_event(
                        InputEvent(text=text, speaker=speaker, kind=FINAL,
                                   channel="discord_text", raw=channel,
                                   mentioned=mentioned, reply_to_her=reply_to_her)
                    )
            except Exception as e:
                print(f"[Discord] turn error: {e}")
            finally:
                self._inbox.task_done()

    # ----- mouth: send reply back to the originating channel -----
    def speak(self, text: str) -> None:
        if not text:
            return
        channel = self._current_channel
        if channel is None or self._loop is None:
            return
        text = text[:2000]  # Discord message hard limit
        try:
            fut = asyncio.run_coroutine_threadsafe(channel.send(text), self._loop)
            fut.result(timeout=30)   # block until sent (text is near-instant)
        except Exception as e:
            print(f"[Discord] send failed: {e}")

    # text send already blocks in speak(); nothing extra to wait on
    def wait_until_done(self) -> None:
        pass