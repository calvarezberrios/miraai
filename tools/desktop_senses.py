"""
desktop_senses.py — run this on the DESKTOP (the streaming machine) so Mira, running on the
LAPTOP, can SEE the desktop's screen and HEAR its game audio over the LAN.

Mira's own --vision / --game-audio capture the machine SHE runs on. When she's on the laptop
but the game + stream are on the desktop, that's the wrong machine. This little companion
closes the gap: it captures the desktop's primary monitor and its game-audio loopback, and
serves them over HTTP so the laptop can pull them. Nothing here touches the Discord VC — it's a
separate side channel just for her senses.

Division of labor (so neither machine is overloaded):
  * SCREEN  — captured here, downscaled to a JPEG, served raw at GET /frame. The laptop's VL
              model does the actual captioning; this only ships pixels.
  * GAME AUDIO — captured AND transcribed here (its own faster-whisper), so only short text
              dialogue lines cross the network. Served at GET /game-audio?since=<seq>.
  * TWITCH CHAT — not here; the laptop reads Twitch's network API directly.

Run (on the desktop, in the repo's venv):
    python tools/desktop_senses.py
It prints the URL to point the laptop at. Then in start_stream_servers.bat on the LAPTOP set:
    set DESKTOP_IP=<this desktop's LAN IP>
(the laptop's vision/game-audio then pull from http://%DESKTOP_IP%:8200/...).

Config (env / .env):
    MIRA_SENSES_PORT=8200                 HTTP port to serve on
    MIRA_SENSES_MAX_DIM=1024              longest edge of the served screen JPEG
    MIRA_VISION_EVERY=2                   seconds between screen grabs (kept fresh; cheap)
    MIRA_VISION_MONITOR=                  mss monitor index; default = primary (anchored at 0,0)
    MIRA_GAME_AUDIO_DEVICE=               loopback/output device name or index (default output)
    WHISPER_DEVICE=cuda  WHISPER_MODEL_SIZE=small.en  WHISPER_COMPUTE_TYPE=int8_float16
    MIRA_GAME_END_SILENCE=1.0  MIRA_GAME_MIN_SPOKEN=0.4  MIRA_GAME_MAX_SEC=30
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Load .env the same way the rest of the app does, if present.
try:
    import env_loader  # noqa: F401  (auto-loads .env into os.environ on import)
except Exception:
    pass

PORT = int(os.environ.get("MIRA_SENSES_PORT", "8200"))
MAX_DIM = int(os.environ.get("MIRA_SENSES_MAX_DIM", "1024"))
CAPTURE_SEC = float(os.environ.get("MIRA_VISION_EVERY", "2"))

TARGET_SR = 16000
END_SILENCE = float(os.environ.get("MIRA_GAME_END_SILENCE", "1.0"))
MIN_SPOKEN = float(os.environ.get("MIRA_GAME_MIN_SPOKEN", "0.4"))
MAX_SEC = float(os.environ.get("MIRA_GAME_MAX_SEC", "30"))
SILENCE_RMS = float(os.environ.get("MIRA_GAME_SILENCE_RMS", "0.006"))  # below this = "quiet"


# ===========================================================================
# SCREEN — grab the primary monitor, keep the latest JPEG
# ===========================================================================
class ScreenSense:
    def __init__(self) -> None:
        self._jpeg: bytes = b""
        self._ts = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._grab = None

    def start(self) -> bool:
        try:
            import mss
            from PIL import Image  # noqa: F401
        except Exception as e:
            print(f"[senses/screen] disabled — needs mss + pillow ({e})")
            return False
        try:
            sct = mss.mss()
            mon, idx = self._pick_monitor(sct)
            from PIL import Image

            def grab():
                shot = sct.grab(mon)
                return Image.frombytes("RGB", shot.size, shot.rgb)
            self._grab = grab
            print(f"[senses/screen] capturing monitor #{idx}: {mon['width']}x{mon['height']} "
                  f"at ({mon['left']},{mon['top']}), every {CAPTURE_SEC:.0f}s -> /frame")
        except Exception as e:
            print(f"[senses/screen] disabled — capture init failed ({e})")
            return False
        self._running = True
        threading.Thread(target=self._loop, name="senses-screen", daemon=True).start()
        return True

    def _pick_monitor(self, sct):
        mons = sct.monitors
        want = os.environ.get("MIRA_VISION_MONITOR", "").strip()
        if want.isdigit() and 0 <= int(want) < len(mons):
            return mons[int(want)], int(want)
        for i in range(1, len(mons)):
            m = mons[i]
            if m.get("left", 0) == 0 and m.get("top", 0) == 0:
                return m, i
        return (mons[1], 1) if len(mons) > 1 else (mons[0], 0)

    def _loop(self):
        from PIL import Image
        while self._running:
            try:
                img = self._grab()
                if max(img.size) > MAX_DIM:
                    img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                with self._lock:
                    self._jpeg = buf.getvalue()
                    self._ts = time.time()
            except Exception as e:
                print(f"[senses/screen] grab failed ({e})")
            time.sleep(CAPTURE_SEC)

    def latest(self):
        with self._lock:
            return self._jpeg, self._ts


# ===========================================================================
# GAME AUDIO — capture loopback, endpoint utterances, transcribe locally
# ===========================================================================
class AudioSense:
    def __init__(self) -> None:
        self._running = False
        self._sd = None
        self._np = None
        self._model = None
        self._stream = None
        self._dev_sr = 48000
        self._pending = []
        self._cbuf = None
        self._last_text = ""
        self._lock = threading.Lock()
        self._lines = deque(maxlen=200)   # (seq, ts, text)
        self._seq = 0

    def start(self) -> bool:
        try:
            import sounddevice as sd
            import numpy as np
            from faster_whisper import WhisperModel
        except Exception as e:
            print(f"[senses/audio] disabled — needs sounddevice + faster-whisper + numpy ({e})")
            return False
        self._sd, self._np = sd, np
        device = os.environ.get("WHISPER_DEVICE", "cuda").strip() or "cuda"
        size = os.environ.get("WHISPER_MODEL_SIZE", "small.en").strip() or "small.en"
        compute = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16").strip() or "int8_float16"
        try:
            self._model = WhisperModel(size, device=device, compute_type=compute)
            print(f"[senses/audio] whisper {size} on {device} ({compute})")
        except Exception as e:
            print(f"[senses/audio] whisper load failed on {device}/{compute} ({e}); trying CPU int8")
            try:
                self._model = WhisperModel(size, device="cpu", compute_type="int8")
            except Exception as e2:
                print(f"[senses/audio] disabled — whisper unavailable ({e2})")
                return False
        dev, channels, loopback = self._resolve_device()
        if dev is None:
            return False
        try:
            self._stream = self._open_stream(dev, channels, loopback)
            self._stream.start()
        except Exception as e:
            print(f"[senses/audio] couldn't open capture on device {dev!r}: {e}\n"
                  f"  List devices:  python tools/check_capture.py devices\n"
                  f"  Then set MIRA_GAME_AUDIO_DEVICE to a recordable output ('CABLE Output',\n"
                  f"  a Voicemeeter 'Output' bus, or 'Stereo Mix'), or update sounddevice (>=0.5.0).")
            return False
        self._running = True
        threading.Thread(target=self._loop, name="senses-audio", daemon=True).start()
        mode = "loopback" if loopback else "direct input"
        print(f"[senses/audio] hearing the game via {mode} (device {dev!r}, {channels}ch @ "
              f"{self._dev_sr} Hz) -> transcribed dialogue at /game-audio")
        return True

    # device resolution (mirrors peripheral_nervous_system/game_audio.py)
    def _resolve_device(self):
        sd = self._sd
        want = os.environ.get("MIRA_GAME_AUDIO_DEVICE", "").strip()
        try:
            if want:
                idx = int(want) if want.isdigit() else next(
                    i for i, d in enumerate(sd.query_devices()) if want.lower() in d["name"].lower())
                return (idx, *self._mode_for(idx))
            out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
            if out is None or out < 0:
                out = sd.query_hostapis(sd.default.hostapi)["default_output_device"]
            return (out, *self._mode_for(out))
        except Exception as e:
            print(f"[senses/audio] couldn't resolve a capture device ({e}); set MIRA_GAME_AUDIO_DEVICE.")
            return None, None, None

    def _mode_for(self, idx):
        info = self._sd.query_devices(idx)
        self._dev_sr = int(info.get("default_samplerate", 48000) or 48000)
        in_ch = int(info.get("max_input_channels", 0) or 0)
        out_ch = int(info.get("max_output_channels", 0) or 0)
        if in_ch > 0:
            return max(1, min(2, in_ch)), False
        return max(1, out_ch or 2), True

    def _open_stream(self, idx, channels, loopback):
        sd = self._sd
        extra = None
        if loopback:
            try:
                extra = sd.WasapiSettings(loopback=True)
            except TypeError:
                extra = None
        return sd.InputStream(device=idx, channels=channels, samplerate=self._dev_sr,
                              dtype="float32", blocksize=int(self._dev_sr * 0.1),
                              callback=self._cb, extra_settings=extra)

    def _cb(self, indata, frames, time_info, status):
        np = self._np
        a = indata if indata.ndim == 1 else indata.mean(axis=1)
        with self._lock:
            self._pending.append(np.asarray(a, dtype=np.float32).copy())

    def _resample_16k(self, win):
        np = self._np
        if self._dev_sr == TARGET_SR:
            return win.astype(np.float32)
        n = max(1, int(len(win) * TARGET_SR / self._dev_sr))
        x = np.linspace(0, 1, num=len(win), endpoint=False, dtype=np.float32)
        xi = np.linspace(0, 1, num=n, endpoint=False, dtype=np.float32)
        return np.interp(xi, x, win).astype(np.float32)

    def _drain_all(self):
        np = self._np
        with self._lock:
            if not self._pending:
                return None
            buf = np.concatenate(self._pending)
            self._pending = []
        return self._resample_16k(buf)

    def _rms(self, x):
        np = self._np
        if x is None or len(x) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x))))

    def _transcribe(self, audio) -> str:
        segs, _ = self._model.transcribe(audio, language="en", vad_filter=True,
                                         beam_size=1, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()

    def _loop(self):
        np = self._np
        SR = TARGET_SR
        win = int(SR * 0.2)
        while self._running:
            time.sleep(0.25)
            chunk = self._drain_all()
            if chunk is not None and len(chunk):
                self._cbuf = chunk if self._cbuf is None else np.concatenate([self._cbuf, chunk])
            if self._cbuf is None or len(self._cbuf) < int(SR * 0.3):
                continue
            # trailing-silence endpointing via RMS on the tail
            tail = self._cbuf[-win:] if len(self._cbuf) >= win else self._cbuf
            quiet = self._rms(tail) < SILENCE_RMS
            dur = len(self._cbuf) / SR
            # estimate trailing-silence length by scanning back from the end
            trailing = 0.0
            if quiet:
                step = int(SR * 0.1)
                k = len(self._cbuf)
                while k > 0:
                    seg = self._cbuf[max(0, k - step):k]
                    if self._rms(seg) >= SILENCE_RMS:
                        break
                    trailing += len(seg) / SR
                    k -= step
            maxed = dur >= MAX_SEC
            if not ((trailing >= END_SILENCE and dur - trailing >= MIN_SPOKEN) or maxed):
                # cap runaway silence-only buffers so we don't grow forever on pure quiet
                if quiet and dur > END_SILENCE * 2 and self._rms(self._cbuf) < SILENCE_RMS:
                    self._cbuf = None
                continue
            audio = self._cbuf
            self._cbuf = None
            try:
                text = self._transcribe(audio)
            except Exception as e:
                print(f"[senses/audio] transcribe error: {e}")
                continue
            if not text or text == self._last_text:
                continue
            self._last_text = text
            with self._lock:
                self._seq += 1
                self._lines.append((self._seq, time.time(), text))
            print(f"[senses/audio] game> {text}")

    def since(self, seq: int):
        with self._lock:
            new = [(s, ts, t) for (s, ts, t) in self._lines if s > seq]
            last = self._seq
        return last, [{"seq": s, "ts": ts, "text": t} for (s, ts, t) in new]


# ===========================================================================
# HTTP server
# ===========================================================================
SCREEN = ScreenSense()
AUDIO = AudioSense()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):       # quiet the default per-request logging
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/health":
            return self._send(200, "ok", "text/plain")
        if p.path == "/frame":
            jpeg, ts = SCREEN.latest()
            if not jpeg:
                return self._send(503, "no frame yet", "text/plain")
            return self._send(200, jpeg, "image/jpeg")
        if p.path == "/game-audio":
            since = int((parse_qs(p.query).get("since", ["0"])[0]) or 0)
            last, lines = AUDIO.since(since)
            return self._send(200, {"last": last, "lines": lines})
        return self._send(404, "not found", "text/plain")


def _lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)   # show the banner/IP + game> lines live in logs
    except Exception:
        pass
    have_screen = SCREEN.start()
    have_audio = AUDIO.start()
    if not (have_screen or have_audio):
        print("[senses] nothing to serve (both screen and audio failed to start). Exiting.")
        return
    ip = _lan_ip()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("\n" + "=" * 60)
    print(f"  Desktop senses serving on http://{ip}:{PORT}")
    print(f"    screen : {'ON  -> /frame' if have_screen else 'off'}")
    print(f"    audio  : {'ON  -> /game-audio' if have_audio else 'off'}")
    print(f"  On the LAPTOP set DESKTOP_IP={ip} in start_stream_servers.bat")
    print(f"  (open TCP {PORT} inbound in the desktop firewall if the laptop can't reach it)")
    print("=" * 60 + "\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[senses] shutting down.")
        srv.shutdown()


if __name__ == "__main__":
    main()
