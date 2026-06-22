"""
Discord adapter — text chat + voice (full loop).

TEXT: every human message is transported to the brain as an InputEvent with the
addressing signals; the brain decides whether to answer; her reply is sent back
to the channel.

VOICE (say "mira join" while you're in a voice channel):
  join a VC  ->  py-cord native recorder (vc.start_recording + discord.sinks.Sink)
             ->  per-speaker CONTINUOUS timeline (gaps filled with silence)
             ->  SAME endpointing as the local mic (wernickes_area): in-buffer VAD,
                 finalize after END_SILENCE of trailing silence
             ->  one Whisper instance (wernickes.transcribe), serial
             ->  InputEvent(channel="discord_voice", speaker=<member>)
             ->  same brain + action selector (addressed = her name spoken)
             ->  reply synthesized by brocas (Piper+RVC) and played INTO the VC

WHY THIS MATCHES LOCAL VOICE
  A microphone always produces samples — even silence — so the local path can end
  a turn by watching for trailing silence *inside* the audio buffer. Discord only
  transmits packets while you actually speak (voice-activity), so silence never
  enters the buffer on its own. We bridge that gap with a 50 ms "ticker": every
  tick each active speaker's buffer is advanced by either the audio that arrived
  or an equal slice of silence. The buffer then looks just like a mic stream, and
  we reuse the exact in-buffer VAD endpointing from wernickes_area — so a Discord
  turn ends as snappily as a local one (END_SILENCE after you stop), instead of
  waiting on wall-clock packet gaps.

Voice receive under Discord's mandatory DAVE E2EE requires py-cord built from a
revision that decrypts incoming frames (PR #3159), installed from git. py-cord
and discord.py share the `discord` namespace, so only one may be installed.

Everything heavy (discord, numpy, Whisper, TTS) is imported lazily so text-only
use and importing the package never require the voice stack.

Threading: the Discord client runs its own asyncio loop in a background thread.
on_message and the voice ticker only ENQUEUE finished turns; a single worker
thread drains the queue one item at a time, so text turns, voice turns,
transcription, thinking, and speaking are all serialized (one model call at a
time — friendly to 6GB VRAM).

Config: token from .env (DISCORD_BOT_TOKEN), env var, or ./discord_token.txt.
Voice-OUT needs ffmpeg on PATH.
"""

from __future__ import annotations

import asyncio
import io
import os
import queue
import threading
import time
from typing import Optional

from .io_adapter import IOAdapter, InputEvent, OnEvent, FINAL, PARTIAL

TOKEN_ENV = "DISCORD_BOT_TOKEN"
TOKEN_FILE = "discord_token.txt"
ENV_FILE = ".env"

# Only reply in these text channels (by name); empty = everywhere she can read. DMs always pass.
ONLY_CHANNELS = set()

# Voice control phrases (checked before normal message handling)
JOIN_COMMANDS = {"mira join", "mira, join", "!join", "mira come here"}
LEAVE_COMMANDS = {"mira leave", "mira, leave", "!leave", "mira go away"}

# ----------------------------------------------------------------------------
# Voice endpointing — mirrors the local mic path (wernickes_area) 1:1.
# A per-speaker buffer is advanced every VOICE_BLOCK_SEC by either arrived audio
# or an equal slice of silence, so trailing silence accrues exactly like a mic.
# Endpointing is then the SAME in-buffer VAD test the local path uses.
# ----------------------------------------------------------------------------
VOICE_BLOCK_SEC = 0.05      # ticker granularity (matches local BLOCK_SEC)
VOICE_REFRESH_SEC = 0.7     # how often to re-run VAD + live transcription (matches local)
VOICE_END_SILENCE = float(os.environ.get("MIRA_VOICE_END_SILENCE", "1.2"))
                            # trailing (synthesized) silence that ends a turn. Tune with
                            # MIRA_VOICE_END_SILENCE: raise toward 3.0 if your py-cord receive
                            # build delivers audio in late bursts (gaps mid-sentence can otherwise
                            # look like a stop and chop the turn); lower toward 1.0 if steady.
VOICE_MIN_SECONDS = 0.4     # ignore utterances with less speech than this (coughs, blips)
VOICE_MAX_SECONDS = 30.0    # force-finalize a monologue this long so the buffer can't grow forever
VOICE_DEBUG = os.environ.get("MIRA_VOICE_DEBUG", "0").strip().lower() not in ("0", "false", "no", "")
# ^ set MIRA_VOICE_DEBUG=1 to print receive/finalize/transcribe diagnostics (use this to tell
#   whether audio is even reaching the sink vs. an endpointing problem).

_STOP = object()  # sentinel to stop the worker


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


def _load_token():
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        token = _read_env_file(ENV_FILE, TOKEN_ENV) or _read_env_file(ENV_FILE, "DISCORD_TOKEN")
    if not token and os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            token = f.read().strip()
    return token


class DiscordAdapter(IOAdapter):
    name = "discord"

    def __init__(self) -> None:
        self._on_event: Optional[OnEvent] = None
        self._client = None
        self._discord = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._net_thread: Optional[threading.Thread] = None
        self._work_thread: Optional[threading.Thread] = None
        self._voice_thread: Optional[threading.Thread] = None
        self._inbox = queue.Queue()
        self._voice_synth_q = queue.Queue()      # sentences awaiting synthesis (synth-ahead)
        self._voice_play_q = queue.Queue()       # synthesized WAVs awaiting VC playback
        self._current_target = ("text", None)   # ("text", channel) | ("voice", None)
        self._last_text_channel = None           # most recent text channel seen (for notify())
        self._running = False

        # voice
        self._voice_client = None
        self._vbuf = {}                          # uid -> per-speaker timeline state
        self._vbuf_lock = threading.Lock()
        self._voice_ready = False
        self._dbg_seen = False                   # printed "receiving audio" yet this session
        self._np = None
        self._wernicke = None
        self._brocas = None
        self._vad = None                          # Silero VAD fn (loaded with voice deps)
        self._vad_opts = None
        self._brain_busy = threading.Event()     # set while she's transcribing/thinking/speaking;
                                                  # pauses live partials so Whisper won't fight TTS

    # ---------------- lifecycle ----------------
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
            import discord
        except ImportError:
            print("\n[Discord] py-cord isn't installed.\n"
                  "  Uninstall discord.py first, then install py-cord with voice:\n"
                  "    python -m pip uninstall -y discord.py discord-ext-voice-recv discord\n"
                  '    python -m pip install -U "py-cord[voice] @ git+https://github.com/Pycord-Development/pycord"\n')
            raise SystemExit(1)

        self._discord = discord

        intents = discord.Intents.default()
        intents.message_content = True   # PRIVILEGED — also enable in the portal
        intents.voice_states = True      # to see which VC a member is in
        # workers: brain feeder + voice ticker (loop-independent)
        self._running = True
        self._work_thread = threading.Thread(target=self._process, daemon=True)
        self._work_thread.start()
        self._voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self._voice_thread.start()
        # voice output pipeline: synthesize sentences AHEAD of playback so N+1 is made
        # while N is still playing in the VC (the overlap that makes local feel snappy).
        self._voice_synth_thread = threading.Thread(target=self._voice_synth_worker, daemon=True)
        self._voice_synth_thread.start()
        self._voice_play_thread = threading.Thread(target=self._voice_play_worker, daemon=True)
        self._voice_play_thread.start()

        # Create the loop up front (other threads schedule onto it via
        # run_coroutine_threadsafe), but build the Client INSIDE the runner
        # thread, on this loop. This dev build's rewritten voice internals bind
        # to the loop that is active when the voice objects are created;
        # constructing the client here keeps voice connect on the same loop the
        # client runs on. Otherwise join fails with "Future attached to a
        # different loop".
        self._loop = asyncio.new_event_loop()

        def _runner():
            asyncio.set_event_loop(self._loop)

            client = discord.Client(intents=intents)
            self._client = client

            @client.event
            async def on_ready():
                print(f"[Discord] Connected as {client.user} — Mira is listening.\n")

            @client.event
            async def on_message(message):
                if message.author == client.user or message.author.bot:
                    return
                content = (message.clean_content or "").strip()
                if not content:
                    return

                # remember where to post non-voice notices (note-taking confirmations/recaps)
                self._last_text_channel = message.channel

                low = content.lower()
                if low in JOIN_COMMANDS:
                    await self._join(message)
                    return
                if low in LEAVE_COMMANDS:
                    await self._leave(message)
                    return

                is_dm = message.guild is None
                if ONLY_CHANNELS and not is_dm:
                    ch_name = getattr(message.channel, "name", None)
                    if ch_name not in ONLY_CHANNELS:
                        return

                mentioned = (client.user in message.mentions) or is_dm
                reply_to_her = False
                ref = message.reference
                if ref is not None:
                    resolved = getattr(ref, "resolved", None)
                    if resolved is not None and getattr(resolved, "author", None) == client.user:
                        reply_to_her = True

                speaker = getattr(message.author, "display_name", None) or str(message.author)
                self._inbox.put(("text", message.channel, speaker, content,
                                 mentioned, reply_to_her, is_dm))

            try:
                self._loop.run_until_complete(client.start(token))
            except Exception as e:
                print(f"[Discord] connection ended: {e}")

        self._net_thread = threading.Thread(target=_runner, daemon=True)
        self._net_thread.start()

    def stop(self) -> None:
        self._running = False
        self._inbox.put(_STOP)
        self._voice_synth_q.put(_STOP)   # unblock the voice-output workers
        self._voice_play_q.put(_STOP)
        if self._client is not None and self._loop is not None and self._loop.is_running():
            try:
                if self._voice_client is not None:
                    asyncio.run_coroutine_threadsafe(
                        self._voice_client.disconnect(), self._loop
                    ).result(timeout=10)
            except Exception:
                pass
            try:
                asyncio.run_coroutine_threadsafe(self._client.close(), self._loop).result(timeout=10)
            except Exception:
                pass
        if self._net_thread is not None:
            self._net_thread.join(timeout=10)
        if self._work_thread is not None:
            self._work_thread.join(timeout=5)

    # ---------------- voice: lazy deps ----------------
    def _ensure_voice_imports(self) -> bool:
        if self._voice_ready:
            return True
        try:
            import discord.sinks  # py-cord native voice receive
            import numpy as np
            from brain.forebrain.cerebrum.temporal_lobe import wernickes_area
            from brain.forebrain.cerebrum.frontal_lobe import brocas_area
            from faster_whisper.vad import VadOptions, get_speech_timestamps
        except Exception as e:
            print(f"[Discord] voice deps missing: {e}\n"
                  f'  python -m pip install -U "py-cord[voice] @ git+https://github.com/Pycord-Development/pycord"')
            return False
        try:
            if not self._discord.opus.is_loaded():
                self._discord.opus._load_default()
            print(f"[Discord] opus loaded: {self._discord.opus.is_loaded()}")
        except Exception as e:
            print(f"[Discord] opus load warning (voice receive needs opus): {e}")
        print(f"[Discord] py-cord version: {getattr(self._discord, '__version__', '?')} "
              f"(needs the PR #3159 build to decrypt DAVE voice-receive)")
        self._np = np
        self._wernicke = wernickes_area
        self._brocas = brocas_area
        self._vad = get_speech_timestamps
        self._vad_opts = VadOptions(min_silence_duration_ms=500)
        self._voice_ready = True
        return True

    # ---------------- voice: join / leave ----------------
    async def _connect_voice(self, channel):
        """Connect (or move) to a voice channel and start listening via py-cord's
        native recorder. Under DAVE, py-cord's voice client decrypts incoming
        frames (PR #3159) and hands decoded PCM to our sink's write()."""
        if not self._ensure_voice_imports():
            return False
        try:
            existing = channel.guild.voice_client if channel.guild else None
            if existing is not None:
                await existing.move_to(channel)
                vc = existing
            else:
                vc = await channel.connect()
            self._voice_client = vc
            # Defensively probe for a DAVE/E2EE indicator the build may expose,
            # so the logs confirm the call is encrypted and the build sees it.
            for attr in ("is_e2ee", "is_dave", "e2ee", "dave"):
                if hasattr(vc, attr):
                    try:
                        v = getattr(vc, attr)
                        print(f"[Discord] voice E2EE/DAVE [{attr}]: {v() if callable(v) else v}")
                    except Exception:
                        pass
                    break
            if not getattr(vc, "recording", False):
                vc.start_recording(self._make_sink(), self._on_recording_done)
            self._brocas.warmup()   # heat Piper + RVC now, off the event loop
            print(f"[Discord] joined voice: {channel}")
            return True
        except Exception as e:
            print(f"[Discord] voice join failed: {e}")
            return False

    async def _on_recording_done(self, sink, *args):
        # Required by start_recording; fires when stop_recording() is called.
        # We process utterances in real time in the ticker, so this is a no-op.
        return

    async def _disconnect_voice(self):
        vc = self._voice_client
        if vc is None:
            return
        try:
            if getattr(vc, "recording", False):
                vc.stop_recording()
        except Exception:
            pass
        try:
            vc.stop()   # stop any TTS playback
        except Exception:
            pass
        try:
            await vc.disconnect()
        except Exception:
            pass
        self._voice_client = None
        with self._vbuf_lock:
            self._vbuf.clear()
        self._dbg_seen = False
        print("[Discord] left voice")

    async def _join(self, message):
        vc_channel = getattr(getattr(message.author, "voice", None), "channel", None)
        if vc_channel is None:
            await message.channel.send("you're not in a voice channel~")
            return
        if await self._connect_voice(vc_channel):
            await message.channel.send(f"*hops into {vc_channel.name}*")
        else:
            await message.channel.send("(couldn't join voice — check the logs)")

    async def _leave(self, message):
        if self._voice_client is None:
            return
        await self._disconnect_voice()
        try:
            await message.channel.send("*slips out of voice*")
        except Exception:
            pass

    # ---------------- voice: receive ----------------
    def _make_sink(self):
        """A real-time py-cord sink. py-cord decodes opus -> PCM before calling
        write(), so `data` is decoded 48k stereo s16 PCM (DAVE-decrypted upstream
        by the voice client). py-cord calls write(data, user); `user` is a user id."""
        discord = self._discord
        adapter = self

        class _MiraSink(discord.sinks.Sink):
            def __init__(self):
                super().__init__()

            def write(self, data, user):
                adapter._voice_callback(user, data)

            def cleanup(self):
                pass

        return _MiraSink()

    def _resolve_member(self, user):
        """py-cord may hand the sink a full Member, a bare discord.Object (id only),
        or an int id. Resolve to something with a real name for display."""
        # already a full member/user (has a name)? use it as-is
        if hasattr(user, "display_name"):
            return user
        try:
            uid = int(getattr(user, "id", user))
        except (TypeError, ValueError):
            return user
        vc = self._voice_client
        # best source for VC speakers: members currently in the voice channel
        ch = getattr(vc, "channel", None) if vc else None
        if ch is not None:
            for m in getattr(ch, "members", []):
                if getattr(m, "id", None) == uid:
                    return m
        # guild member cache
        guild = getattr(vc, "guild", None) if vc else None
        if guild is not None:
            m = guild.get_member(uid)
            if m is not None:
                return m
        # global user cache (populated from messages / on_ready)
        if self._client is not None:
            u = self._client.get_user(uid)
            if u is not None:
                return u
        return user  # last resort; str() will show the id object

    def _voice_callback(self, user, data):
        """Called from py-cord's decode thread, ~every 20ms per active speaker.
        We do NOT endpoint here — we just convert to 16k mono float32 and stash it
        as 'pending'. The ticker thread folds pending audio into the continuous
        per-speaker timeline. This keeps receive cheap and endpointing identical
        to the local mic path."""
        if user is None or data is None:
            return
        if isinstance(data, (bytes, bytearray)):
            pcm = bytes(data)
        else:  # tolerate RawData-like objects across py-cord versions
            pcm = (getattr(data, "pcm", None)
                   or getattr(data, "decoded_data", None)
                   or getattr(data, "decrypted_data", None))
        if not pcm:
            return
        f32 = self._pcm_to_f32_16k(pcm)
        if f32.size == 0:
            return
        member = self._resolve_member(user)
        uid = member.id if hasattr(member, "id") else int(user)

        if VOICE_DEBUG and not self._dbg_seen:
            self._dbg_seen = True
            print(f"[voice] receiving audio (first packet from "
                  f"{getattr(member, 'display_name', member)}, {len(pcm)} bytes)")

        with self._vbuf_lock:
            b = self._vbuf.get(uid)
            if b is None:
                b = self._new_speaker_state(member)
                self._vbuf[uid] = b
            b["pending"].append(f32)
            b["member"] = member

    def _new_speaker_state(self, member):
        np = self._np
        return {
            "member": member,
            "buf": np.zeros(0, dtype=np.float32),  # continuous timeline (worker-owned)
            "pending": [],                          # arrived-but-not-yet-folded (receive thread)
            "last_refresh": 0.0,                    # monotonic, worker-owned
            "partial_text": "",                     # worker-owned
        }

    def _voice_worker(self):
        """The mic-equivalent for Discord. Every VOICE_BLOCK_SEC, advance each
        active speaker's continuous timeline by either the audio that arrived or
        an equal slice of silence, then run the SAME endpointing as the local mic:
        VAD finds the speech span, and a turn ends after VOICE_END_SILENCE of
        trailing silence inside the buffer."""
        SR = 16000
        block_len = int(SR * VOICE_BLOCK_SEC)
        while self._running:
            time.sleep(VOICE_BLOCK_SEC)
            if self._vad is None:                # voice deps not loaded yet
                continue
            np = self._np

            with self._vbuf_lock:
                uids = list(self._vbuf.keys())

            for uid in uids:
                # 1) fold this tick's audio (or silence) into the timeline
                with self._vbuf_lock:
                    b = self._vbuf.get(uid)
                    if b is None:
                        continue
                    pending = b["pending"]
                    b["pending"] = []
                if pending:
                    chunk = np.concatenate(pending)
                else:
                    chunk = np.zeros(block_len, dtype=np.float32)   # silence, like a quiet mic
                b["buf"] = np.concatenate([b["buf"], chunk])

                # 2) endpoint on a refresh cadence (identical to local _worker)
                now = time.monotonic()
                if now - b["last_refresh"] < VOICE_REFRESH_SEC:
                    continue
                b["last_refresh"] = now
                self._endpoint_speaker(uid, b, SR)

    def _endpoint_speaker(self, uid, b, SR):
        """Mirror of wernickes_area._worker's per-refresh logic, per speaker."""
        buf = b["buf"]
        try:
            speech = self._vad(buf, self._vad_opts) if len(buf) else []
        except Exception as e:
            if VOICE_DEBUG:
                print(f"[voice] vad error: {e}")
            speech = []

        if not speech:
            # no speech yet -> keep only the last second so the buffer doesn't grow
            if len(buf) > SR:
                b["buf"] = buf[-SR:]
            return

        silence = (len(buf) - speech[-1]["end"]) / SR
        spoken = (speech[-1]["end"] - speech[0]["start"]) / SR

        # live caption: re-transcribe as you talk, but never while she's mid-reply
        # (so Whisper doesn't fight the TTS for the GPU).
        member = b["member"]
        if not self._brain_busy.is_set():
            text = self._transcribe_f32(buf)
            if text and text != b["partial_text"]:
                b["partial_text"] = text
                speaker = getattr(member, "display_name", None) or str(member)
                self._emit(InputEvent(text=text, speaker=speaker, kind=PARTIAL,
                                      channel="discord_voice", raw=None))

        end_now = silence >= VOICE_END_SILENCE or spoken >= VOICE_MAX_SECONDS
        if not end_now:
            return

        # turn over -> finalize and hand to the brain worker
        with self._vbuf_lock:
            self._vbuf.pop(uid, None)
        if spoken < VOICE_MIN_SECONDS:
            if VOICE_DEBUG:
                print(f"[voice] dropped (spoke {spoken:.1f}s < {VOICE_MIN_SECONDS}s)")
            return
        final_text = b["partial_text"]
        final_audio = buf
        if VOICE_DEBUG:
            print(f"[voice] finalized {getattr(member, 'display_name', member)} "
                  f"(silence {silence:.1f}s, spoke {spoken:.1f}s) -> {final_text!r}")
        self._inbox.put(("voice", member, final_text, final_audio))

    def _pcm_to_f32_16k(self, pcm):
        np = self._np
        a = np.frombuffer(pcm, dtype=np.int16)
        if a.size == 0:
            return np.zeros(0, dtype=np.float32)
        if a.size % 2 == 0:                       # stereo -> mono
            a = a.reshape(-1, 2).mean(axis=1)
        a = a.astype(np.float32) / 32768.0
        n = (a.size // 3) * 3                      # 48000 -> 16000
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        return a[:n].reshape(-1, 3).mean(axis=1).astype(np.float32)

    def _transcribe_f32(self, f32):
        try:
            return self._wernicke.transcribe(f32)
        except Exception as e:
            print(f"[Discord] transcription error: {e}")
            return ""

    # ---------------- worker: queue -> brain ----------------
    def _emit(self, ev):
        if self._on_event is not None:
            self._on_event(ev)

    def _process(self):
        while self._running:
            item = self._inbox.get()
            if item is _STOP:
                break
            self._brain_busy.set()       # pause live partials while she works (no GPU fight)
            try:
                kind = item[0]
                if kind == "text":
                    _, channel, speaker, text, mentioned, reply_to_her, is_dm = item
                    self._current_target = ("text", channel)
                    self._emit(InputEvent(text=text, speaker=speaker, kind=FINAL,
                                          channel="discord_text", raw=channel,
                                          mentioned=mentioned, reply_to_her=reply_to_her,
                                          is_dm=is_dm))
                elif kind == "voice":
                    _, member, pretranscribed, audio_f32 = item
                    text = pretranscribed or self._transcribe_f32(audio_f32)
                    if not text:
                        continue
                    speaker = getattr(member, "display_name", None) or str(member)
                    print(f"[voice] {speaker}: {text}")
                    vc_channel = self._voice_client.channel if self._voice_client else None
                    self._current_target = ("voice", None)
                    # Voice is NOT auto-addressed. handle_message runs a chime-in decision on
                    # un-named voice (she joins in if it's relevant or about her, else stays
                    # quiet). Saying "Mira" still forces a direct reply via name_mentioned.
                    self._emit(InputEvent(text=text, speaker=speaker, kind=FINAL,
                                          channel="discord_voice", raw=vc_channel))
            except Exception as e:
                print(f"[Discord] turn error: {e}")
            finally:
                self._brain_busy.clear()
                self._inbox.task_done()

    # ---------------- mouth ----------------
    def speak(self, text: str) -> None:
        kind, target = self._current_target
        if kind == "voice":
            self._speak_voice(text)
        else:
            self._speak_text(text, target)

    def notify(self, text: str) -> None:
        """Non-voice status channel for note-taking (confirmations / recaps / saved paths).
        Always prints to the console, and best-effort posts to the most recent text channel
        so a Discord user sees it. Never uses the voice — note-taking stays silent."""
        if not text:
            return
        print(text)
        ch = self._last_text_channel
        if ch is None or self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(ch.send(text[:2000]), self._loop)
        except Exception as e:
            print(f"[Discord] notify failed: {e}")

    def _speak_text(self, text, channel):
        if not text or channel is None or self._loop is None:
            return
        text = text[:2000]
        try:
            asyncio.run_coroutine_threadsafe(channel.send(text), self._loop).result(timeout=30)
        except Exception as e:
            print(f"[Discord] send failed: {e}")

    def _speak_voice(self, text):
        """Queue a reply for VC playback. Returns fast — the synth worker makes each
        sentence while the play worker streams the previous one into the VC (overlap),
        instead of the old synth->play->synth serial path. wait_until_done() blocks the
        turn until it's all spoken."""
        brocas = self._brocas
        vc = self._voice_client
        if not text or brocas is None or vc is None or not vc.is_connected():
            return
        cleaned = brocas._clean_for_speech(text)
        if not cleaned:
            return
        for sentence in brocas._split_sentences(cleaned):
            self._voice_synth_q.put(sentence)

    def _voice_synth_worker(self):
        """Synthesize queued sentences ahead of playback; hand each WAV to the play queue."""
        while self._running:
            sentence = self._voice_synth_q.get()
            try:
                if sentence is _STOP:
                    break
                brocas = self._brocas
                wav = brocas._synthesize(sentence) if brocas else None
                if wav:
                    self._voice_play_q.put(wav)
            except Exception as e:
                print(f"[Discord voice] synth error: {e}")
            finally:
                self._voice_synth_q.task_done()

    def _voice_play_worker(self):
        """Play synthesized sentences into the VC in order, one at a time."""
        while self._running:
            wav = self._voice_play_q.get()
            try:
                if wav is _STOP:
                    break
                self._play_wav_in_vc(wav)   # blocks until this sentence finishes
            except Exception as e:
                print(f"[Discord voice] play error: {e}")
            finally:
                self._voice_play_q.task_done()

    def wait_until_done(self) -> None:
        """Block until everything queued for the VC is synthesized AND played, so the turn
        (and pause-input) holds until she has actually finished speaking."""
        self._voice_synth_q.join()
        self._voice_play_q.join()

    def _play_wav_in_vc(self, wav_bytes):
        vc = self._voice_client
        discord = self._discord
        if vc is None or not vc.is_connected() or discord is None or self._loop is None:
            return
        done = threading.Event()

        def _after(err):
            if err:
                print(f"[Discord voice] playback error: {err}")
            done.set()

        try:
            source = discord.FFmpegPCMAudio(io.BytesIO(wav_bytes), pipe=True)
        except Exception as e:
            print(f"[Discord voice] ffmpeg source failed (is ffmpeg on PATH?): {e}")
            return

        async def _do_play():
            if vc.is_playing():
                vc.stop()
            vc.play(source, after=_after)

        lip_stop = self._brocas.lip_drive_bytes(wav_bytes)   # move the mouth in time
        try:
            asyncio.run_coroutine_threadsafe(_do_play(), self._loop).result(timeout=15)
            done.wait(timeout=120)
        except Exception as e:
            print(f"[Discord voice] play failed: {e}")
        finally:
            if lip_stop is not None:
                lip_stop.set()

    def wait_until_done(self) -> None:
        # text send blocks in _speak_text; voice playback blocks in _play_wav_in_vc
        pass
