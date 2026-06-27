"""
brocas_area.py — speech production (TTS) for Mira.

Anatomy: Broca's area (frontal lobe) = assembling words into produced speech.

THREE engines, chosen at startup by the MIRA_TTS env var:
  - "kokoro"    (default): Kokoro-82M neural TTS (hexgrad/Kokoro-82M). Runs inline
                in this process (PyTorch), no subprocess, no RVC. Mira's voice is
                the "Jessica" voice (af_jessica) by default. Natural, light, fast.
  - "piper"              : Piper TTS -> optional RVC voice conversion. Fast and
                light; an older local-Windows path. RVC runs as a subprocess under
                a Python that has fairseq/faiss/torch (the GPT-SoVITS runtime).
  - "gptsovits"          : sends text to a locally-running GPT-SoVITS api_v2
                server and plays the returned audio. Heavier/higher quality; the
                RunPod path (set MIRA_TTS=gptsovits there).

Everything around the engine is shared: text cleaning, sentence splitting, the
two-stage synth/playback pipeline (sentence N plays while N+1 synthesizes), the
Phase-5 avatar lip-sync (amplitude envelope -> mouth), and the public API. Only
_synthesize() and warmup() branch on the engine.
"""

import atexit
import collections
import io
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import wave
from typing import Callable, Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

# ---------------------------------------------------------------------------
# Engine select
# ---------------------------------------------------------------------------
# "kokoro" (default, local), "piper" (local) or "gptsovits" (RunPod). Set MIRA_TTS to switch.
ENGINE = os.environ.get("MIRA_TTS", "kokoro").strip().lower()
if ENGINE not in ("kokoro", "piper", "gptsovits"):
    print(f"[brocas_area] unknown MIRA_TTS={ENGINE!r}; falling back to 'kokoro'")
    ENGINE = "kokoro"

# ---------------------------------------------------------------------------
# Config — Kokoro (used when ENGINE == "kokoro")
# ---------------------------------------------------------------------------
# Kokoro pins numpy==1.26.4 and pulls misaki[en]->spacy, none of which have
# Python 3.14 wheels, so it can't run in this (3.14) process. It runs in a
# dedicated Python 3.10 venv as a persistent subprocess — the same pattern as RVC.

# Kokoro-82M voice. "af_jessica" is the American-English female "Jessica" voice.
KOKORO_VOICE = os.environ.get("MIRA_KOKORO_VOICE", "af_jessica")
# Language pack for the g2p pipeline. 'a' = American English (matches af_* voices).
KOKORO_LANG = os.environ.get("MIRA_KOKORO_LANG", "a")
KOKORO_REPO = os.environ.get("MIRA_KOKORO_REPO", "hexgrad/Kokoro-82M")
# Speaking rate: 1.0 = Kokoro's native pace, <1.0 slower, >1.0 faster. Default 0.92
# (~8% slower) because her natural delivery runs a touch fast. Tune with MIRA_KOKORO_SPEED.
KOKORO_SPEED = os.environ.get("MIRA_KOKORO_SPEED", "0.92")

# Python runtime that has kokoro installed (the dedicated 3.10 venv).
KOKORO_PYTHON = os.environ.get(
    "MIRA_KOKORO_PYTHON",
    os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        r"..\..\..\..\.venv-kokoro\Scripts\python.exe",
    )),
)

# Path to our Kokoro inference script (repo-relative).
KOKORO_SCRIPT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), r"..\..\..\..\kokoro_tts\kokoro_infer.py"
))

# ---------------------------------------------------------------------------
# Config — Piper + RVC (used when ENGINE == "piper")
# ---------------------------------------------------------------------------

# Piper voice model (English female, converted to Mira's voice by RVC).
PIPER_MODEL = os.environ.get(
    "MIRA_PIPER_MODEL",
    r"D:\aiproject\piper_voices\en\en_US\hfc_female\medium\en_US-hfc_female-medium.onnx",
)

# Master switch for the RVC stage. When False, Mira speaks with the raw Piper
# female voice and no RVC subprocess is launched.
USE_RVC = os.environ.get("USE_RVC", "1").strip().lower() not in ("0", "false", "no", "")

# RVC model files — drop your .pth and .index here.
RVC_MODEL = os.environ.get("MIRA_RVC_MODEL", r"D:\aiproject\rvc_models\mira.pth")
RVC_INDEX = os.environ.get("MIRA_RVC_INDEX", r"D:\aiproject\rvc_models\mira.index")  # "" to skip

# Python runtime that has fairseq + faiss + torch (GPT-SoVITS bundled runtime).
RVC_PYTHON = os.environ.get(
    "MIRA_RVC_PYTHON", r"D:\GPT-SoVITS-v3lora-20250228\runtime\python.exe"
)

# Path to our RVC inference script (repo-relative).
RVC_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), r"..\..\..\..\rvc\rvc_infer.py")
)

PITCH_SHIFT = int(os.environ.get("MIRA_PITCH_SHIFT", "0"))      # semitones (0 = no change)
INDEX_RATE = float(os.environ.get("MIRA_INDEX_RATE", "0.5"))    # .index blend 0.0–1.0

# ---------------------------------------------------------------------------
# Config — GPT-SoVITS (used when ENGINE == "gptsovits")
# ---------------------------------------------------------------------------
API_URL = os.environ.get("GPTSOVITS_URL", "http://127.0.0.1:9880/tts")

# Reference clip the GPT-SoVITS server clones from (zero-shot). Read by the
# SERVER process, so the path must be valid where that server runs (same box).
_DEFAULT_REF = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice", "sample_clip.mp3")
)
VOICE = {
    "ref_audio_path": os.environ.get("MIRA_REF_AUDIO", _DEFAULT_REF),
    "prompt_text": os.environ.get(
        "MIRA_REF_TEXT",
        "You want to know all about me, huh? Well, let's just say I'm a bit of a handful.",
    ),
    "prompt_lang": os.environ.get("MIRA_REF_LANG", "en"),
    "text_lang": os.environ.get("MIRA_TEXT_LANG", "en"),
}
PARAMS = {
    "text_split_method": "cut5",
    "batch_size": 1,
    "media_type": "wav",
    "streaming_mode": False,
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 1.0,
    "parallel_infer": False,
    "split_bucket": False,
    # GPT-SoVITS speed: >1 faster, <1 slower. Default 1.12 (the original tuned pace).
    # Tune with MIRA_SOVITS_SPEED if you ever want her slower/faster.
    "speed_factor": float(os.environ.get("MIRA_SOVITS_SPEED", "1.12")),
}
SYNTH_TIMEOUT = float(os.environ.get("GPTSOVITS_TIMEOUT", "60"))

print(f"[brocas_area] TTS engine: {ENGINE}")

# ---------------------------------------------------------------------------
# Speech text cleaning (shared)
# ---------------------------------------------------------------------------

SPOKEN_ACTIONS = {
    "giggle": "hehe", "giggles": "hehe",
    "laugh": "haha",  "laughs": "haha",
    "chuckle": "heh", "chuckles": "heh",
    "giggling": "hehe", "laughing": "haha",
    "hums": "hmm", "hum": "hmm",
    "sighs": "haah", "sigh": "haah",
    "gasps": "gah",  "gasp": "gah",
    "snickers": "heh",
}

_ACTION_SPAN_RE = re.compile(r"\*([^*]+)\*|\(([^)]*)\)|\[([^\]]*)\]")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F0FF"
    "\U00002190-\U000021FF"
    "\U0000FE00-\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)
_SENTENCE_RE = re.compile(r".*?[.!?](?:\s|$)|.+$", re.S)


def _replace_action(match):
    inner = (match.group(1) or match.group(2) or match.group(3) or "").lower()
    for word, sound in SPOKEN_ACTIONS.items():
        if word in inner:
            return sound
    return ""


def _clean_for_speech(text):
    # type: (str) -> str
    text = _ACTION_SPAN_RE.sub(_replace_action, text)
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)
    text = re.sub(r"\s*[‒–—―-]\s*", ", ", text)
    text = text.replace("*", "").replace("_", "").replace("`", "")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return text.strip()


def _split_sentences(text):
    # type: (str) -> list
    parts = [m.group().strip() for m in _SENTENCE_RE.finditer(text)]
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Engine: Kokoro synthesis (a persistent subprocess keeps the model hot)
# ---------------------------------------------------------------------------

_kokoro_proc = None
_kokoro_lock = threading.Lock()


def _start_kokoro_server():
    # type: () -> Optional[subprocess.Popen]
    if not os.path.isfile(KOKORO_PYTHON):
        print(f"[brocas_area] kokoro python not found: {KOKORO_PYTHON}")
        return None
    cmd = [KOKORO_PYTHON, KOKORO_SCRIPT, "--serve",
           "--voice", KOKORO_VOICE, "--lang", KOKORO_LANG,
           "--repo", KOKORO_REPO, "--speed", str(KOKORO_SPEED)]
    try:
        # Capture the server's stderr instead of letting it inherit our console: torch/HF
        # warnings and its own "[kokoro] pipeline ready" diagnostics would otherwise stomp on
        # the startup progress bar. A daemon thread drains it (so the pipe can't fill and block
        # the server) into a small ring buffer we only surface if the server fails to come up.
        # encoding="utf-8": the text we send can contain non-Latin1 characters (e.g. 'ō' in
        # "Ohayō"); without this the pipe defaults to Windows cp1252 and the write raises
        # 'charmap' codec can't encode ... . The server side reconfigures its stdin to match.
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
    except Exception as e:
        print("[brocas_area] failed to launch Kokoro server:", e)
        return None

    stderr_tail = collections.deque(maxlen=50)

    def _drain_stderr():
        for ln in proc.stderr:
            stderr_tail.append(ln.rstrip("\n"))

    threading.Thread(target=_drain_stderr, daemon=True).start()

    for line in proc.stdout:
        if line.strip() == "READY":
            print("[brocas_area] Kokoro server ready")
            return proc
        if proc.poll() is not None:
            break
    # It died before signaling READY — surface the captured stderr so the real cause is visible.
    print("[brocas_area] Kokoro server exited before becoming ready")
    if stderr_tail:
        print("[brocas_area] Kokoro server stderr (tail):")
        for ln in stderr_tail:
            print("  " + ln)
    return None


def _ensure_kokoro_server():
    # type: () -> Optional[subprocess.Popen]
    global _kokoro_proc
    if _kokoro_proc is not None and _kokoro_proc.poll() is None:
        return _kokoro_proc
    with _kokoro_lock:
        if _kokoro_proc is not None and _kokoro_proc.poll() is None:
            return _kokoro_proc
        _kokoro_proc = _start_kokoro_server()
        return _kokoro_proc


@atexit.register
def _shutdown_kokoro_server():
    if _kokoro_proc is not None and _kokoro_proc.poll() is None:
        try:
            _kokoro_proc.stdin.write("QUIT\n")
            _kokoro_proc.stdin.flush()
            _kokoro_proc.wait(timeout=3)
        except Exception:
            _kokoro_proc.kill()


def _synth_kokoro(text):
    # type: (str) -> Optional[bytes]
    """Kokoro TTS (subprocess) -> WAV bytes (24 kHz mono). None on failure."""
    proc = _ensure_kokoro_server()
    if proc is None:
        return None
    with tempfile.NamedTemporaryFile(suffix="_kokoro.wav", delete=False) as tf:
        out_path = tf.name
    try:
        with _kokoro_lock:
            # text is a single cleaned sentence (no newlines); path has no '|'.
            proc.stdin.write("%s|%s\n" % (out_path, text.replace("\n", " ")))
            proc.stdin.flush()
            reply = proc.stdout.readline().strip()
        if reply != "OK":
            print("[brocas_area] kokoro:", reply)
            return None
        with open(out_path, "rb") as f:
            return f.read()
    except Exception as e:
        print("[brocas_area] kokoro server error:", e)
        return None
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Engine: Piper synthesis
# ---------------------------------------------------------------------------

_piper_voice = None
_piper_lock = threading.Lock()


def _ensure_piper_loaded():
    global _piper_voice
    if _piper_voice is not None:
        return
    with _piper_lock:
        if _piper_voice is not None:
            return
        from piper.voice import PiperVoice
        _piper_voice = PiperVoice.load(PIPER_MODEL)
        print("[brocas_area] Piper voice loaded:", PIPER_MODEL)


def _piper_synthesize(text, out_path):
    # type: (str, str) -> bool
    """Run Piper TTS and write WAV to out_path. Returns True on success."""
    try:
        _ensure_piper_loaded()
        with wave.open(out_path, "wb") as wav_file:
            _piper_voice.synthesize_wav(text, wav_file)
        return True
    except Exception as e:
        print("[brocas_area] piper error:", e)
        return False


# ---------------------------------------------------------------------------
# Engine: RVC voice conversion (a persistent subprocess keeps the model hot)
# ---------------------------------------------------------------------------

_rvc_proc = None
_rvc_lock = threading.Lock()


def _start_rvc_server():
    # type: () -> Optional[subprocess.Popen]
    if not os.path.isfile(RVC_MODEL):
        return None
    cmd = [RVC_PYTHON, RVC_SCRIPT, "--serve", "--model", RVC_MODEL,
           "--index_rate", str(INDEX_RATE)]
    if RVC_INDEX and os.path.isfile(RVC_INDEX):
        cmd += ["--index", RVC_INDEX]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=None, text=True, bufsize=1)
    except Exception as e:
        print("[brocas_area] failed to launch RVC server:", e)
        return None
    for line in proc.stdout:
        if line.strip() == "READY":
            print("[brocas_area] RVC server ready")
            return proc
        if proc.poll() is not None:
            break
    print("[brocas_area] RVC server exited before becoming ready")
    return None


def _ensure_rvc_server():
    # type: () -> Optional[subprocess.Popen]
    global _rvc_proc
    if _rvc_proc is not None and _rvc_proc.poll() is None:
        return _rvc_proc
    with _rvc_lock:
        if _rvc_proc is not None and _rvc_proc.poll() is None:
            return _rvc_proc
        _rvc_proc = _start_rvc_server()
        return _rvc_proc


@atexit.register
def _shutdown_rvc_server():
    if _rvc_proc is not None and _rvc_proc.poll() is None:
        try:
            _rvc_proc.stdin.write("QUIT\n")
            _rvc_proc.stdin.flush()
            _rvc_proc.wait(timeout=3)
        except Exception:
            _rvc_proc.kill()


def _rvc_convert(input_wav, output_wav):
    # type: (str, str) -> bool
    proc = _ensure_rvc_server()
    if proc is None:
        return False
    with _rvc_lock:
        try:
            proc.stdin.write("%s|%s|%d\n" % (input_wav, output_wav, PITCH_SHIFT))
            proc.stdin.flush()
            reply = proc.stdout.readline().strip()
            if reply == "OK":
                return True
            print("[brocas_area] rvc:", reply)
            return False
        except Exception as e:
            print("[brocas_area] rvc server error:", e)
            return False


def _synth_piper(text):
    # type: (str) -> Optional[bytes]
    """Piper -> (optional) RVC -> WAV bytes."""
    with tempfile.NamedTemporaryFile(suffix="_piper.wav", delete=False) as tf_in:
        piper_path = tf_in.name
    with tempfile.NamedTemporaryFile(suffix="_rvc.wav", delete=False) as tf_out:
        rvc_path = tf_out.name
    try:
        if not _piper_synthesize(text, piper_path):
            return None
        if not USE_RVC or not os.path.isfile(RVC_MODEL):
            with open(piper_path, "rb") as f:
                return f.read()
        if not _rvc_convert(piper_path, rvc_path):
            print("[brocas_area] RVC failed, falling back to Piper audio")
            with open(piper_path, "rb") as f:
                return f.read()
        with open(rvc_path, "rb") as f:
            return f.read()
    finally:
        for p in (piper_path, rvc_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Engine: GPT-SoVITS synthesis
# ---------------------------------------------------------------------------

def _synth_gptsovits(text):
    # type: (str) -> Optional[bytes]
    """Synthesize one chunk via the GPT-SoVITS api_v2 server. Returns WAV bytes."""
    payload = {
        "text": text,
        "text_lang": VOICE["text_lang"],
        "ref_audio_path": VOICE["ref_audio_path"],
        "prompt_text": VOICE["prompt_text"],
        "prompt_lang": VOICE["prompt_lang"],
    }
    payload.update(PARAMS)
    try:
        r = requests.post(API_URL, json=payload, timeout=SYNTH_TIMEOUT)
    except requests.RequestException as e:
        print("[brocas_area] request failed:", e)
        return None
    if r.status_code != 200 or not r.headers.get("Content-Type", "").startswith("audio"):
        print("[brocas_area] synth failed (%s): %s" % (r.status_code, r.text[:200]))
        return None
    return r.content


def _synthesize(text):
    # type: (str) -> Optional[bytes]
    """Dispatch to the active engine. Returns WAV bytes (optionally converted)."""
    if ENGINE == "kokoro":
        return _synth_kokoro(text)
    if ENGINE == "gptsovits":
        return _synth_gptsovits(text)
    return _synth_piper(text)


# ---------------------------------------------------------------------------
# Lip-sync — amplitude envelope streamed to the avatar's mouth during playback
# ---------------------------------------------------------------------------

_lip_cb = None                  # type: Optional[Callable[[float], None]]
_LIP_HOP = 0.03                 # seconds per envelope frame (~33 Hz)
# Output latency to compensate (mouth/caption vs. sound): used as a fallback when sounddevice
# can't report the live stream latency. Tune with MIRA_LIP_LATENCY (seconds).
_LIP_LATENCY = float(os.environ.get("MIRA_LIP_LATENCY", "0.05"))

_subtitle_cb = None             # type: Optional[Callable[[str, float], None]]


def set_lip_callback(cb):
    # type: (Optional[Callable[[float], None]]) -> None
    """Register a sink for lip-sync levels (0..1). None disables lip-sync."""
    global _lip_cb
    _lip_cb = cb


def set_subtitle_callback(cb):
    # type: (Optional[Callable[[str, float], None]]) -> None
    """Register a sink for subtitles: cb(text, duration_seconds), called when a line STARTS
    playing so the caption reveals word-by-word across the audio. None disables subtitles."""
    global _subtitle_cb
    _subtitle_cb = cb


def _emit_subtitle(text, duration):
    # type: (Optional[str], float) -> None
    if _subtitle_cb is not None and text:
        try:
            _subtitle_cb(text, float(duration))
        except Exception:
            pass


def _rms_envelope(data, sr):
    # type: (np.ndarray, int) -> list
    if data.ndim > 1:
        data = data.mean(axis=1)
    hop = max(1, int(sr * _LIP_HOP))
    levels = []
    for i in range(0, len(data), hop):
        chunk = data[i:i + hop]
        if chunk.size:
            levels.append(float(np.sqrt(np.mean(chunk * chunk))))
        else:
            levels.append(0.0)
    peak = max(levels) if levels else 0.0
    if peak <= 1e-6:
        return [0.0 for _ in levels]
    return [min(1.0, (l / peak) * 1.3) for l in levels]


def _start_lip_thread(data, sr, start_delay=0.0):
    # type: (np.ndarray, int, float) -> Optional[threading.Event]
    if _lip_cb is None:
        return None
    levels = _rms_envelope(data, sr)
    if not levels:
        return None
    stop = threading.Event()

    def _run():
        # Each envelope frame idx is emitted at t0 + idx*HOP, where t0 includes the output
        # latency (start_delay) so the mouth opens with the SOUND, not before it leaves the
        # device. Sleeping BEFORE the emit (vs. after) keeps it aligned to that schedule.
        t0 = time.time() + max(0.0, start_delay)
        for idx, level in enumerate(levels):
            if stop.is_set():
                break
            ahead = (t0 + idx * _LIP_HOP) - time.time()
            if ahead > 0:
                time.sleep(ahead)
            try:
                _lip_cb(level)
            except Exception:
                pass
        try:
            _lip_cb(0.0)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
    return stop


def lip_drive_bytes(wav_bytes, text=None):
    # type: (bytes, Optional[str]) -> Optional[threading.Event]
    """Start lip-sync (and emit the caption) from raw WAV bytes (Discord voice path plays via
    ffmpeg). `text` is the line being spoken, shown as a subtitle across the clip's duration."""
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception:
        return None
    _emit_subtitle(text, (len(data) / sr) if sr else 0.0)
    return _start_lip_thread(data, sr)


# ---------------------------------------------------------------------------
# Playback (local mic/speaker mode only; Discord plays via ffmpeg into the VC)
# ---------------------------------------------------------------------------

def _play(wav_bytes, text=None):
    # type: (bytes, Optional[str]) -> None
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    duration = (len(data) / sr) if sr else 0.0
    sd.play(data, sr)
    # Align the mouth (and caption) with the actual audio onset: sounddevice reports the
    # output stream's latency — the gap between play() and the first sample reaching the
    # speaker. Compensating it stops the lips running ahead of the sound.
    try:
        latency = float(sd.get_stream().latency)
    except Exception:
        latency = _LIP_LATENCY
    stop = _start_lip_thread(data, sr, start_delay=latency)
    _emit_subtitle(text, duration)
    sd.wait()
    if stop is not None:
        stop.set()


# ---------------------------------------------------------------------------
# Pipeline queues (motor sequencing)
# ---------------------------------------------------------------------------

_synth_q = queue.Queue()
_play_q  = queue.Queue()


def _synth_worker():
    while True:
        text, should_play = _synth_q.get()
        try:
            audio = _synthesize(text)
            if audio and should_play:
                _play_q.put((audio, text))      # carry the text so playback can caption it
        except Exception as e:
            print("[brocas_area] synth error:", e)
        finally:
            _synth_q.task_done()


def _play_worker():
    while True:
        audio, text = _play_q.get()
        try:
            _play(audio, text)
        except Exception as e:
            print("[brocas_area] playback error:", e)
        finally:
            _play_q.task_done()


threading.Thread(target=_synth_worker, daemon=True).start()
threading.Thread(target=_play_worker,  daemon=True).start()


# ---------------------------------------------------------------------------
# Public API (unchanged interface)
# ---------------------------------------------------------------------------

def say(text):
    # type: (str) -> None
    """Queue text for Mira to speak. Returns immediately; playback is serialized."""
    if not (text and text.strip()):
        return
    text = _clean_for_speech(text)
    if not text:
        return
    for sentence in _split_sentences(text):
        _synth_q.put((sentence, True))


_warmed = False   # set once the TTS has been fully heated, so repeat warmup() calls no-op


def warmup(progress=None, blocking=False):
    """FULLY heat the active TTS so her first spoken line is instant: start the engine AND
    run one throwaway synthesis — that first synth is what loads the voice and runs the
    first (slow) forward pass, and on a fresh machine it also downloads the weights. Doing
    it now means the first REAL line doesn't pay that cost.

    Idempotent: once warmed, later calls return immediately and print nothing (so a second
    caller — e.g. joining a Discord VC after the startup warmup — doesn't re-run the synth or
    re-print "loading voice model...").

    `progress(stage:str)` is called with short status strings (for a startup indicator);
    defaults to print. Runs in a background thread unless blocking=True; returns True/False
    (warmed ok) when blocking, else None."""
    def _log(stage):
        (progress or print)(stage)

    def _warm():
        global _warmed
        if _warmed:
            return True               # already hot — silent no-op
        try:
            if ENGINE == "kokoro":
                _log("starting Kokoro voice engine")
                if _ensure_kokoro_server() is None:
                    _log("failed to start Kokoro"); return False
                _log("loading voice model (first run downloads weights)")
                ok = _synth_kokoro("Voice check, one two.") is not None
            elif ENGINE == "gptsovits":
                _log("warming GPT-SoVITS server")
                ok = _synthesize("Voice check.") is not None
            else:  # piper (+ optional RVC)
                _log("loading Piper voice")
                _ensure_piper_loaded()
                if USE_RVC:
                    _log("starting RVC voice conversion")
                    _ensure_rvc_server()
                ok = _synth_piper("Voice check.") is not None
            _log("ready" if ok else "warmup synth failed")
            if ok:
                _warmed = True
            return ok
        except Exception as e:
            _log(f"warmup error: {e}")
            return False

    if blocking:
        return _warm()
    threading.Thread(target=_warm, daemon=True).start()
    return None


def wait_until_done():
    """Block until all queued speech has finished playing."""
    _synth_q.join()
    _play_q.join()


if __name__ == "__main__":
    say(f"Hey there. If you can hear this, the {ENGINE} voice is working correctly.")
    wait_until_done()
