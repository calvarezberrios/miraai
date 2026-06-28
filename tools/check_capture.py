"""
Pre-flight probe for stream mode — validate each subsystem on its own BEFORE going live.

Run on the DESKTOP (the machine that runs Mira). It reuses the real code paths, so a green
check here means the live run will work too.

    python tools/check_capture.py              # run every check
    python tools/check_capture.py twitch       # just the Twitch status (Helix) auth
    python tools/check_capture.py audio vision # any subset

Checks:
    twitch  - Helix app-token auth + your stream's live/offline + viewers + game
    chat    - anonymous IRC connect + JOIN your channel, confirm chat is readable
    audio   - open the WASAPI loopback and measure the level (PLAY SOME AUDIO during this)
    vision  - grab a screen frame and caption it via the laptop VL endpoint

Exit code is 0 only if every requested check passes.
"""

from __future__ import annotations

import sys
import time

# Make the project root importable when run as a loose script (python tools/check_capture.py).
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from env_loader import load_env

OK = "\033[92m[ OK ]\033[0m"
NO = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def _hdr(name: str) -> None:
    print(f"\n=== {name} " + "=" * (40 - len(name)))


# ---------------------------------------------------------------------------
def check_twitch() -> bool:
    _hdr("twitch status (Helix)")
    from peripheral_nervous_system import stream_status as S
    ss = S.StreamStatus()
    ss._client_id = S._cfg("TWITCH_CLIENT_ID")
    ss._secret = S._cfg("TWITCH_CLIENT_SECRET")
    ss._channel = S._cfg("TWITCH_CHANNEL").lstrip("#").lower()
    if not (ss._client_id and ss._secret and ss._channel):
        print(f"{NO} TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET / TWITCH_CHANNEL not all set in .env")
        return False
    if not ss._refresh_token():
        print(f"{NO} app-token auth failed — double-check the Client ID and Secret")
        return False
    print(f"{OK} authenticated (app access token acquired)")
    if not ss._resolve_user():
        print(f"{NO} channel '{ss._channel}' not found — check TWITCH_CHANNEL (name only, no '#')")
        return False
    print(f"{OK} resolved channel '{ss._channel}' (user id {ss._user_id})")
    ss._poll_stream()
    s = ss.snapshot()
    if s["live"]:
        print(f"{OK} LIVE now — {s['viewers']} viewer(s)"
              + (f", playing {s['game']}" if s["game"] else ""))
    else:
        print(f"{OK} reachable — stream is currently OFFLINE "
              f"(that's fine; she'll see it go live when you start)")
    print(f"     she will hear: \"{ss.summary()}\"")
    return True


# ---------------------------------------------------------------------------
def check_chat() -> bool:
    _hdr("twitch chat (IRC read)")
    import socket
    from peripheral_nervous_system import twitch_adapter as T
    channel = T._load_channel()
    if not channel:
        print(f"{NO} TWITCH_CHANNEL not set in .env")
        return False
    nick = f"justinfan{int(time.time()) % 100000}"
    try:
        sock = socket.create_connection((T.IRC_HOST, T.IRC_PORT), timeout=15)
        sock.settimeout(8)
        sock.sendall(f"NICK {nick}\r\n".encode())
        sock.sendall(f"JOIN #{channel}\r\n".encode())
        joined = False
        seen_msg = 0
        t0 = time.time()
        buf = b""
        while time.time() - t0 < 8:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            while b"\r\n" in buf:
                line, buf = buf.split(b"\r\n", 1)
                s = line.decode("utf-8", "replace")
                if s.startswith("PING"):
                    sock.sendall(b"PONG :tmi.twitch.tv\r\n")
                if " JOIN " in s or " 353 " in s or " 366 " in s:
                    joined = True
                if " PRIVMSG " in s:
                    seen_msg += 1
        sock.close()
    except Exception as e:
        print(f"{NO} couldn't reach Twitch IRC ({T.IRC_HOST}:{T.IRC_PORT}): {e}")
        return False
    if joined:
        extra = f"; saw {seen_msg} chat message(s) in 8s" if seen_msg else "; chat was quiet"
        print(f"{OK} connected and joined #{channel}{extra}")
        return True
    print(f"{WARN} connected but didn't confirm the JOIN in 8s — channel name may be wrong "
          f"('{channel}') or the server was slow. Try again.")
    return False


# ---------------------------------------------------------------------------
def check_audio() -> bool:
    _hdr("game audio (WASAPI loopback)")
    try:
        import sounddevice as sd
        import numpy as np
    except Exception as e:
        print(f"{NO} sounddevice/numpy missing: {e}")
        return False
    from peripheral_nervous_system.game_audio import GameAudio
    ga = GameAudio()
    ga._sd, ga._np = sd, np
    dev, channels = ga._resolve_loopback_device()
    if dev is None:
        print(f"{NO} no loopback device — set MIRA_GAME_AUDIO_DEVICE to a 'Stereo Mix' input")
        return False
    try:
        name = sd.query_devices(dev)["name"]
        sr = int(sd.query_devices(dev).get("default_samplerate", 48000) or 48000)
    except Exception:
        name, sr = str(dev), 48000
    print(f"     device: {name!r} @ {sr} Hz")
    print(f"     >>> PLAY SOME AUDIO now (a video / the game) — capturing 3s...")
    frames = []
    try:
        extra = None
        try:
            extra = sd.WasapiSettings(loopback=True)
        except TypeError:
            extra = None
        with sd.InputStream(device=dev, channels=channels, samplerate=sr, dtype="float32",
                            extra_settings=extra,
                            callback=lambda d, n, t, s: frames.append(d.copy())):
            time.sleep(3.0)
    except Exception as e:
        print(f"{NO} couldn't open the loopback stream: {e}\n"
              f"     Update python-sounddevice (>=0.5.0) or set MIRA_GAME_AUDIO_DEVICE.")
        return False
    if not frames:
        print(f"{NO} no audio frames captured at all (driver issue?)")
        return False
    a = np.concatenate([f.mean(axis=1) if f.ndim > 1 else f for f in frames]).astype(np.float32)
    peak = float(np.abs(a).max()) if a.size else 0.0
    rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
    print(f"     level: peak {peak:.3f}, rms {rms:.4f}")
    if peak < 0.001:
        print(f"{WARN} signal is basically silent — was audio actually playing on THIS output "
              f"device? If you route the game elsewhere, set MIRA_GAME_AUDIO_DEVICE to it.")
        return False
    print(f"{OK} hearing the output mix (she'll transcribe spoken parts, tagged 'Game')")
    return True


# ---------------------------------------------------------------------------
def check_vision() -> bool:
    _hdr("stream vision (screen -> laptop VL)")
    from peripheral_nervous_system.stream_vision import StreamVision
    sv = StreamVision()
    if not sv._init_capture():
        return False
    print(f"{OK} screen capture backend ready")
    if not sv._init_client():
        return False
    try:
        uri = sv._frame_data_uri()
    except Exception as e:
        print(f"{NO} couldn't grab/encode a frame: {e}")
        return False
    print(f"     sending a frame to the VL endpoint (this can take a few seconds)...")
    try:
        cap = sv._caption_frame(uri)
    except Exception as e:
        print(f"{NO} VL endpoint call failed: {e}\n"
              f"     Is the laptop server up? Does MIRA_VISION_MODEL match /v1/models?")
        return False
    if not cap:
        print(f"{NO} endpoint replied but with an empty caption — check the model/mmproj")
        return False
    print(f"{OK} caption: {cap!r}")
    return True


# ---------------------------------------------------------------------------
CHECKS = {"twitch": check_twitch, "chat": check_chat, "audio": check_audio, "vision": check_vision}


def main(argv) -> int:
    load_env()
    names = [a.lower() for a in argv if a.lower() in CHECKS] or list(CHECKS)
    results = {}
    for n in names:
        try:
            results[n] = CHECKS[n]()
        except Exception as e:
            print(f"{NO} {n} check crashed: {e}")
            results[n] = False
    _hdr("summary")
    for n in names:
        print(f"  {OK if results[n] else NO}  {n}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
