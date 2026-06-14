"""
brocas_area.py — speech production (TTS) for Mira.

Pipeline: text → Piper TTS (fast neural TTS) → WAV → RVC voice conversion → playback.

Piper produces clean English-female speech quickly; RVC then converts the timbre
to Mira's trained voice using the .pth / .index model files.

RVC inference runs as a subprocess under the GPT-SoVITS bundled runtime Python,
which has fairseq + faiss + torch+cuda (unlike the main venv on Python 3.14).

Same two-stage synth/playback pipeline as before: sentence N plays while
sentence N+1 is still being synthesized.
"""

import atexit
import io
import os
import queue
import re
import subprocess
import tempfile
import threading
from typing import Optional

import sounddevice as sd
import soundfile as sf

# ---------------------------------------------------------------------------
# Config — edit these paths to match your setup
# ---------------------------------------------------------------------------

# Piper voice model (English female, converted to Mira's voice by RVC)
PIPER_MODEL = r"D:\aiproject\piper_voices\en\en_US\hfc_female\medium\en_US-hfc_female-medium.onnx"

# Master switch: when False, Mira speaks with the raw Piper female voice and RVC
# is skipped entirely (no subprocess launched). Flip to True to re-enable the
# RVC voice conversion once you have a model whose output sounds clean.
USE_RVC = True

# RVC model files — drop your .pth and .index here
RVC_MODEL  = r"D:\aiproject\rvc_models\mira.pth"
RVC_INDEX  = r"D:\aiproject\rvc_models\mira.index"   # set "" to skip

# Python runtime that has fairseq + faiss + torch (GPT-SoVITS bundled runtime)
RVC_PYTHON = r"D:\GPT-SoVITS-v3lora-20250228\runtime\python.exe"

# Path to our RVC inference script
RVC_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                           "..", "rvc", "rvc_infer.py")
RVC_SCRIPT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             r"..\..\..\..\rvc\rvc_infer.py"))

# Pitch shift in semitones (0 = no change; positive = higher)
PITCH_SHIFT = 0

# How much of the .index retrieval to blend in (0.0–1.0); 0 = pure model, no index.
# 0.5 is a clean default — high values (0.8–1.0) add timbre accuracy but can warble.
INDEX_RATE = 0.5

# ---------------------------------------------------------------------------
# Speech text cleaning (same as before)
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
    text = _ACTION_SPAN_RE.sub(_replace_action, text)
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)
    text = re.sub(r"\s*[‒–—―-]\s*", ", ", text)
    text = text.replace("*", "").replace("_", "").replace("`", "")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return text.strip()


def _split_sentences(text):
    parts = [m.group().strip() for m in _SENTENCE_RE.finditer(text)]
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Piper synthesis
# ---------------------------------------------------------------------------

def _piper_synthesize(text, out_path):
    # type: (str, str) -> bool
    """Run Piper TTS and write WAV to out_path. Returns True on success."""
    import wave
    try:
        _ensure_piper_loaded()
        with wave.open(out_path, "wb") as wav_file:
            _piper_voice.synthesize_wav(text, wav_file)
        return True
    except Exception as e:
        print("[brocas_area] piper error:", e)
        return False


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


# ---------------------------------------------------------------------------
# RVC voice conversion
# ---------------------------------------------------------------------------

# A single long-lived RVC subprocess keeps HuBERT + the model + faiss hot, so
# each conversion is just inference (~0.5-1s) instead of a ~10s cold reload.
_rvc_proc = None
_rvc_lock = threading.Lock()


def _start_rvc_server():
    # type: () -> Optional[subprocess.Popen]
    """Launch the persistent RVC server and wait until its models are loaded."""
    if not os.path.isfile(RVC_MODEL):
        return None
    cmd = [
        RVC_PYTHON, RVC_SCRIPT,
        "--serve",
        "--model",      RVC_MODEL,
        "--index_rate", str(INDEX_RATE),
    ]
    if RVC_INDEX and os.path.isfile(RVC_INDEX):
        cmd += ["--index", RVC_INDEX]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1,
        )
    except Exception as e:
        print("[brocas_area] failed to launch RVC server:", e)
        return None
    # block until the server prints READY (models hot)
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
    """Send one conversion request to the persistent RVC server. Serialized."""
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


# ---------------------------------------------------------------------------
# Full pipeline: text → Piper → RVC → bytes
# ---------------------------------------------------------------------------

def _synthesize(text):
    # type: (str) -> Optional[bytes]
    """Return synthesized (and optionally RVC-converted) audio as WAV bytes."""
    with tempfile.NamedTemporaryFile(suffix="_piper.wav", delete=False) as tf_in:
        piper_path = tf_in.name
    with tempfile.NamedTemporaryFile(suffix="_rvc.wav", delete=False) as tf_out:
        rvc_path = tf_out.name

    try:
        if not _piper_synthesize(text, piper_path):
            return None

        # RVC disabled, or model missing → use the Piper female voice directly.
        if not USE_RVC or not os.path.isfile(RVC_MODEL):
            with open(piper_path, "rb") as f:
                return f.read()

        if not _rvc_convert(piper_path, rvc_path):
            # Fallback: use Piper audio without RVC
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
# Playback
# ---------------------------------------------------------------------------

def _play(wav_bytes):
    # type: (bytes) -> None
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    sd.play(data, sr)
    sd.wait()


# ---------------------------------------------------------------------------
# Pipeline queues (motor sequencing — same design as before)
# ---------------------------------------------------------------------------

_synth_q = queue.Queue()
_play_q  = queue.Queue()


def _synth_worker():
    while True:
        text, should_play = _synth_q.get()
        try:
            audio = _synthesize(text)
            if audio and should_play:
                _play_q.put(audio)
        except Exception as e:
            print("[brocas_area] synth error:", e)
        finally:
            _synth_q.task_done()


def _play_worker():
    while True:
        audio = _play_q.get()
        try:
            _play(audio)
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


def warmup():
    """Pre-load Piper voice (and the RVC server if enabled) so reply 1 is fast."""
    threading.Thread(target=_ensure_piper_loaded, daemon=True).start()
    if USE_RVC:
        threading.Thread(target=_ensure_rvc_server, daemon=True).start()


def wait_until_done():
    """Block until all queued speech has finished playing."""
    _synth_q.join()
    _play_q.join()


if __name__ == "__main__":
    say("Hey there. If you can hear this, Piper and RVC are working correctly.")
    wait_until_done()
