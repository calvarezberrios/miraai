"""
Game audio — lets Mira HEAR what the stream hears (the desktop's output mix), so she can
talk about a game's dialogue, an ad read, a YouTube clip, etc.

WHY THIS IS SAFE ALONGSIDE YOUR MIC
  * It's a SEPARATE capture (WASAPI loopback of the default output device), not your mic, so
    it never blocks you talking — the two streams run in parallel.
  * It is NEVER mistaken for you: transcribed game speech is surfaced as AMBIENT context
    tagged "Game" (summary() below), not as a conversational turn under your name. She uses
    it as material to riff on, she doesn't reply to it.
  * It shares the one Whisper model with the mic, which is why wernickes._transcribe() is now
    locked. Game audio runs on a SLOW cadence (every CADENCE_SEC) and is VAD-gated (music /
    silence cost no decode), so the mic stays responsive.

  Requires HEADPHONES on the streamer's side: if game sound comes out of speakers it bleeds
  into the mic and would arrive under your name — headphones keep the mic clean.

Needs WASAPI loopback, which python-sounddevice exposes via WasapiSettings(loopback=True)
(>= 0.5.0). If unavailable, set MIRA_GAME_AUDIO_DEVICE to a "Stereo Mix"/loopback input.

Config (.env / env):
    MIRA_GAME_AUDIO_DEVICE=     (optional output/loopback device name or index; default = default output)
    MIRA_GAME_AUDIO_CADENCE=5   (optional seconds between transcription passes)
    MIRA_GAME_AUDIO_WINDOW=5    (optional seconds of audio transcribed each pass)
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Optional

TARGET_SR = 16000
CADENCE_SEC = float(os.environ.get("MIRA_GAME_AUDIO_CADENCE", "5"))
WINDOW_SEC = float(os.environ.get("MIRA_GAME_AUDIO_WINDOW", "5"))
KEEP_SEC = 45.0            # how long a heard line stays in the ambient summary before aging out
MAX_LINES = 4             # cap the summary so the prompt stays small


def _read_env_file(path, key):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


class GameAudio:
    def __init__(self) -> None:
        self._sd = None
        self._np = None
        self._wernicke = None
        self._stream = None
        self._running = False
        self._dev_sr = 48000
        self._lock = threading.Lock()
        self._pending = []                       # captured float32 mono chunks (device rate)
        self._lines = deque()                    # (timestamp, text) recently heard
        self._last_text = ""
        self._worker: Optional[threading.Thread] = None

    # ---------------- lifecycle ----------------
    def start(self) -> bool:
        try:
            import sounddevice as sd
            import numpy as np
            from brain.forebrain.cerebrum.temporal_lobe import wernickes_area
        except Exception as e:
            print(f"[game-audio] disabled — deps missing: {e}")
            return False
        self._sd, self._np, self._wernicke = sd, np, wernickes_area

        dev, channels, loopback = self.resolve_device()
        if dev is None:
            return False
        try:
            self._stream = self.open_stream(dev, channels, loopback, self._cb)
            self._stream.start()
        except Exception as e:
            print(f"[game-audio] couldn't open capture on device {dev!r}: {e}\n"
                  f"  List devices:  python tools/check_capture.py devices\n"
                  f"  Then set MIRA_GAME_AUDIO_DEVICE to a recordable output (a Voicemeeter/VB "
                  f"'Output' bus, 'Stereo Mix', or 'CABLE Output'), or update python-sounddevice "
                  f"(>=0.5.0 for WASAPI loopback).")
            return False

        self._running = True
        self._worker = threading.Thread(target=self._loop, name="game-audio", daemon=True)
        self._worker.start()
        mode = "loopback" if loopback else "direct input"
        print(f"[game-audio] hearing the output mix via {mode} (device {dev!r}, "
              f"{channels}ch @ {self._dev_sr} Hz). Tagged as 'Game' so it's never confused with you.")
        return True

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
        if self._worker is not None:
            self._worker.join(timeout=3)

    def resolve_device(self):
        """Pick what to capture and HOW. Returns (index, channels, use_loopback).

        Two capture modes, chosen automatically:
          * DIRECT INPUT — the device is already recordable (max_input_channels > 0): a
            Voicemeeter/VB 'Output' bus, 'Stereo Mix', a 'CABLE Output', etc. We open it as a
            normal input. This is the right path for a virtual-mixer setup like Voicemeeter.
          * WASAPI LOOPBACK — a pure render (speaker) device: we capture what's played to it,
            and MUST request the device's full output-channel count (downmixed in the callback)
            or WASAPI rejects it with -9998.

        MIRA_GAME_AUDIO_DEVICE (index or name substring) overrides the auto-pick.
        """
        sd = self._sd
        want = (os.environ.get("MIRA_GAME_AUDIO_DEVICE", "").strip()
                or _read_env_file(".env", "MIRA_GAME_AUDIO_DEVICE"))
        try:
            if want:
                if want.isdigit():
                    idx = int(want)
                else:
                    idx = next(i for i, d in enumerate(sd.query_devices())
                               if want.lower() in d["name"].lower())
                return (idx, *self._mode_for(idx))
            # auto: prefer the default OUTPUT via loopback (with the right channel count)
            out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
            if out is None or out < 0:
                out = sd.query_hostapis(sd.default.hostapi)["default_output_device"]
            return (out, *self._mode_for(out))
        except Exception as e:
            print(f"[game-audio] couldn't resolve a capture device ({e}); set MIRA_GAME_AUDIO_DEVICE.")
            return None, None, None

    def _mode_for(self, idx):
        """(channels, use_loopback) for a device: record it directly if it's already an input,
        else loopback the output with its full channel count."""
        info = self._sd.query_devices(idx)
        self._dev_sr = int(info.get("default_samplerate", 48000) or 48000)
        in_ch = int(info.get("max_input_channels", 0) or 0)
        out_ch = int(info.get("max_output_channels", 0) or 0)
        if in_ch > 0:
            return max(1, min(2, in_ch)), False             # directly recordable -> normal input
        return max(1, out_ch or 2), True                    # render device -> WASAPI loopback

    def open_stream(self, idx, channels, loopback, callback):
        """Open the capture InputStream — WASAPI loopback only when capturing a render device."""
        sd = self._sd
        extra = None
        if loopback:
            try:
                extra = sd.WasapiSettings(loopback=True)
            except TypeError:
                extra = None                                 # older sounddevice; rely on direct input
        return sd.InputStream(
            device=idx, channels=channels, samplerate=self._dev_sr, dtype="float32",
            blocksize=int(self._dev_sr * 0.1), callback=callback, extra_settings=extra,
        )

    # ---------------- capture ----------------
    def _cb(self, indata, frames, time_info, status):
        np = self._np
        a = indata if indata.ndim == 1 else indata.mean(axis=1)   # downmix to mono
        with self._lock:
            self._pending.append(np.asarray(a, dtype=np.float32).copy())

    def _drain_window(self):
        """Concatenate pending audio, keep only the last WINDOW_SEC, resample to 16 kHz mono."""
        np = self._np
        with self._lock:
            if not self._pending:
                return None
            buf = np.concatenate(self._pending)
            keep = int(self._dev_sr * WINDOW_SEC)
            self._pending = [buf[-keep:]] if len(buf) > keep else [buf]
            win = self._pending[0]
        if self._dev_sr == TARGET_SR:
            return win.astype(np.float32)
        n = max(1, int(len(win) * TARGET_SR / self._dev_sr))   # linear resample to 16k
        x = np.linspace(0, 1, num=len(win), endpoint=False, dtype=np.float32)
        xi = np.linspace(0, 1, num=n, endpoint=False, dtype=np.float32)
        return np.interp(xi, x, win).astype(np.float32)

    def _loop(self):
        while self._running:
            time.sleep(CADENCE_SEC)
            if not self._running:
                break
            try:
                win = self._drain_window()
                if win is None or len(win) < TARGET_SR * 0.6:     # need ~>0.6s of audio
                    continue
                if not self._wernicke.speech_present(win):         # cheap VAD: skip music/silence
                    continue
                text = self._wernicke.transcribe(win)              # shared, locked Whisper
                text = (text or "").strip()
                if not text or text == self._last_text:
                    continue
                self._last_text = text
                with self._lock:
                    self._lines.append((time.time(), text))
                    while len(self._lines) > MAX_LINES:
                        self._lines.popleft()
            except Exception as e:
                print(f"[game-audio] pass error: {e}")

    # ---------------- ambient read ----------------
    def summary(self) -> str:
        """One line of ambient context for the brain: what's recently been heard from the game/
        stream audio. Ages out stale lines so she isn't talking about something a minute gone."""
        now = time.time()
        with self._lock:
            recent = [t for (ts, t) in self._lines if now - ts <= KEEP_SEC]
        if not recent:
            return ""
        joined = " / ".join(recent[-MAX_LINES:])
        return (f'From the game/stream audio right now (NOT something the streamer said): "{joined}". '
                f"You can react to or talk about this.")
