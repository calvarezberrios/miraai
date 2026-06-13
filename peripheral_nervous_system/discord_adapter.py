"""
Discord adapter — text chat + voice (full loop).

TEXT: every human message is transported to the brain as an InputEvent with the
addressing signals; the brain decides whether to answer; her reply is sent back
to the channel.

VOICE (say "mira join" while you're in a voice channel):
  join a VC  ->  py-cord native recorder (vc.start_recording + discord.sinks.Sink)
             ->  per-user buffer, finalized on a short silence
             ->  one Whisper instance (wernickes.transcribe), serial
             ->  InputEvent(channel="discord_voice", speaker=<member>)
             ->  same brain + action selector (addressed = her name spoken)
             ->  reply synthesized by GPT-SoVITS (brocas) and played INTO the VC

Voice receive under Discord's mandatory DAVE E2EE requires py-cord built from a
revision that decrypts incoming frames (PR #3159), installed from git. py-cord
and discord.py share the `discord` namespace, so only one may be installed.

Everything heavy (discord, numpy, Whisper, GPT-SoVITS) is imported lazily so
text-only use and importing the package never require the voice stack.

Threading: the Discord client runs its own asyncio loop in a background thread.
on_message and the voice finalizer only ENQUEUE work; a single worker thread
drains the queue one item at a time, so text turns, voice turns, transcription,
thinking, and speaking are all serialized (one model call at a time — friendly
to 6GB VRAM).

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

# Voice endpointing
# Endpointing uses the SAME Silero VAD as the local mic path (wernickes_area), so
# long sentences with natural pauses aren't chopped: VAD finds where speech actually
# is, and an utterance ends only after VOICE_END_SILENCE of trailing silence past it.
VOICE_END_SILENCE = 6.0     # seconds of no audio -> turn over. Set high on purpose: the
                            # experimental #3159 receive build delivers audio in bursts with
                            # multi-second gaps (~4s seen) even mid-sentence, so a low value
                            # mistakes those delivery gaps for you stopping and chops your turn.
                            # 6s rides over the gaps. Lower it if/when the build delivers steadily.
VOICE_REFRESH_SEC = 0.7     # how often to re-run VAD + live transcription while you talk
VOICE_MIN_SECONDS = 0.4     # ignore utterances with less speech than this (coughs, blips)
VOICE_DEBUG = False          # print receive/finalize/transcribe diagnostics

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
        self._fin_thread: Optional[threading.Thread] = None
        self._inbox = queue.Queue()
        self._current_target = ("text", None)   # ("text", channel) | ("voice", None)
        self._running = False

        # voice
        self._voice_client = None
        self._vbuf = {}                          # user_id -> {"member","chunks","last"}
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
        # workers: brain feeder + voice finalizer (loop-independent)
        self._running = True
        self._work_thread = threading.Thread(target=self._process, daemon=True)
        self._work_thread.start()
        self._fin_thread = threading.Thread(target=self._voice_finalizer, daemon=True)
        self._fin_thread.start()

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
            self._brocas.warmup()   # heat GPT-SoVITS now, off the event loop
            print(f"[Discord] joined voice: {channel}")
            return True
        except Exception as e:
            print(f"[Discord] voice join failed: {e}")
            return False

    async def _on_recording_done(self, sink, *args):
        # Required by start_recording; fires when stop_recording() is called.
        # We process utterances in real time inside the sink, so this is a no-op.
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
        # called from py-cord's decode thread, ~every 20ms per active speaker
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
        member = self._resolve_member(user)
        uid = member.id if hasattr(member, "id") else int(user)
        now = time.time()
        if VOICE_DEBUG and not self._dbg_seen:
            self._dbg_seen = True
            print(f"[voice] receiving audio (first packet from "
                  f"{getattr(member, 'display_name', member)}, {len(pcm)} bytes)")

        # Buffer everything; the VAD worker decides what is speech and when it ends
        # (same approach as the local mic path). No level gating here.
        with self._vbuf_lock:
            b = self._vbuf.get(uid)
            if b is None:
                b = {"member": member, "chunks": [], "bytes": 0,
                     "last": now, "last_refresh": 0.0,
                     "partial_text": "", "partial_bytes": -1}
                self._vbuf[uid] = b
                gap = 0.0
            else:
                gap = now - b["last"]
            b["chunks"].append(pcm)
            b["bytes"] += len(pcm)
            b["last"] = now
            b["member"] = member
        # Reveal bursty delivery: if audio ARRIVAL pauses while you're still active,
        # that gap (not your silence) is what the endpointer may be misreading.
        if VOICE_DEBUG and gap > 0.5:
            print(f"[voice] delivery gap {gap:.2f}s (no packets arrived for this long)")

    def _voice_finalizer(self):
        # Endpointing for Discord: unlike a mic, Discord stops sending packets when you
        # go quiet (voice-activity), so silence never fills the buffer. We therefore time
        # the silence by how long since the last packet ARRIVED (wall-clock), not by
        # trailing silence inside the buffer. VAD is still used to (a) confirm there is
        # real speech (reject pure noise) and (b) drive the live caption.
        SR = 16000
        one_sec_raw = 48000 * 2 * 2          # ~1s of 48k stereo int16
        while self._running:
            time.sleep(0.15)
            if self._vad is None:            # voice deps not loaded yet
                continue
            now = time.time()
            jobs = []
            with self._vbuf_lock:
                for uid in list(self._vbuf.keys()):
                    b = self._vbuf[uid]
                    if not b["chunks"] or (now - b["last_refresh"]) < VOICE_REFRESH_SEC:
                        continue
                    b["last_refresh"] = now
                    jobs.append((uid, b["member"], b"".join(b["chunks"]),
                                 b["last"], b.get("partial_bytes", -1),
                                 b.get("partial_text", "")))

            for uid, member, raw, last_pkt, prev_bytes, prev_text in jobs:
                silence_gap = time.time() - last_pkt        # seconds since last packet
                f32 = self._pcm_to_f32_16k(raw)
                try:
                    speech = self._vad(f32, self._vad_opts) if len(f32) else []
                except Exception as e:
                    if VOICE_DEBUG:
                        print(f"[voice] vad error: {e}")
                    speech = []
                spoken = ((speech[-1]["end"] - speech[0]["start"]) / SR) if speech else 0.0

                # live caption: only when the buffer actually grew, there's speech,
                # and she isn't mid-reply (so Whisper doesn't fight GPT-SoVITS).
                text = prev_text
                if speech and len(raw) != prev_bytes and not self._brain_busy.is_set():
                    t = self._transcribe_f32(f32)
                    with self._vbuf_lock:
                        b = self._vbuf.get(uid)
                        if b is not None:
                            b["partial_bytes"] = len(raw)
                    if t and t != prev_text:
                        text = t
                        speaker = getattr(member, "display_name", None) or str(member)
                        self._emit(InputEvent(text=text, speaker=speaker, kind=PARTIAL,
                                              channel="discord_voice", raw=None))
                        with self._vbuf_lock:
                            b = self._vbuf.get(uid)
                            if b is not None:
                                b["partial_text"] = text
                    elif t:
                        text = t

                # endpoint: no packets for VOICE_END_SILENCE -> you've stopped talking.
                # Re-read the freshest packet time at the decision moment so a slow
                # transcribe pass (or you resuming right at the boundary) is never
                # miscounted as your silence.
                with self._vbuf_lock:
                    b = self._vbuf.get(uid)
                    fresh_gap = (time.time() - b["last"]) if b else silence_gap
                if fresh_gap >= VOICE_END_SILENCE:
                    with self._vbuf_lock:
                        b = self._vbuf.pop(uid, None)
                    if b is None:
                        continue
                    final_raw = b"".join(b["chunks"])
                    final_text = b.get("partial_text", "") or text
                    if spoken >= VOICE_MIN_SECONDS:
                        if not final_text:
                            final_text = self._transcribe_f32(self._pcm_to_f32_16k(final_raw))
                        if final_text:
                            if VOICE_DEBUG:
                                print(f"[voice] finalized {getattr(member, 'display_name', member)} "
                                      f"(silence {fresh_gap:.1f}s, spoke {spoken:.1f}s) -> {final_text!r}")
                            self._inbox.put(("voice", member, final_raw, final_text))
                            continue
                    if VOICE_DEBUG:
                        print(f"[voice] dropped (silence {fresh_gap:.1f}s, spoke {spoken:.1f}s, "
                              f"text={final_text!r})")
                elif not speech and len(raw) > one_sec_raw:
                    # packets still coming but no speech (noise / always-transmit): trim
                    with self._vbuf_lock:
                        b = self._vbuf.get(uid)
                        if b is not None:
                            self._trim_buffer(b, one_sec_raw)

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

    def _transcribe_pcm(self, pcm):
        try:
            return self._wernicke.transcribe(self._pcm_to_f32_16k(pcm))
        except Exception as e:
            print(f"[Discord] transcription error: {e}")
            return ""

    def _transcribe_f32(self, f32):
        try:
            return self._wernicke.transcribe(f32)
        except Exception as e:
            print(f"[Discord] transcription error: {e}")
            return ""

    @staticmethod
    def _trim_buffer(b, max_bytes):
        """Keep only the most recent ~max_bytes of raw audio (drop older silence)."""
        kept, acc = [], 0
        for ch in reversed(b["chunks"]):
            kept.append(ch)
            acc += len(ch)
            if acc >= max_bytes:
                break
        b["chunks"] = list(reversed(kept))
        b["bytes"] = acc

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
                    _, member, pcm, pretranscribed = item
                    text = pretranscribed or self._transcribe_pcm(pcm)
                    if not text:
                        continue
                    speaker = getattr(member, "display_name", None) or str(member)
                    print(f"[voice] {speaker}: {text}")
                    vc_channel = self._voice_client.channel if self._voice_client else None
                    self._current_target = ("voice", None)
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

    def _speak_text(self, text, channel):
        if not text or channel is None or self._loop is None:
            return
        text = text[:2000]
        try:
            asyncio.run_coroutine_threadsafe(channel.send(text), self._loop).result(timeout=30)
        except Exception as e:
            print(f"[Discord] send failed: {e}")

    def _speak_voice(self, text):
        brocas = self._brocas
        vc = self._voice_client
        if not text or brocas is None or vc is None or not vc.is_connected():
            return
        cleaned = brocas._clean_for_speech(text)
        if not cleaned:
            return
        for sentence in brocas._split_sentences(cleaned):
            wav = brocas._synthesize(sentence)
            if wav:
                self._play_wav_in_vc(wav)   # blocks until this sentence finishes

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

        try:
            asyncio.run_coroutine_threadsafe(_do_play(), self._loop).result(timeout=15)
            done.wait(timeout=120)
        except Exception as e:
            print(f"[Discord voice] play failed: {e}")

    def wait_until_done(self) -> None:
        # text send blocks in _speak_text; voice playback blocks in _play_wav_in_vc
        pass