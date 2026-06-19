"""Wernicke's area — speech comprehension (STT).

Continuous listening with live partial transcripts.
- Mic audio is captured in a background thread (sounddevice).
- Every REFRESH_SEC the current utterance is re-transcribed with
  faster-whisper, so words appear as you speak (on_partial).
- Silero VAD watches for END_SILENCE_SEC of quiet -> utterance is
  finalized and handed to on_final().
- If you keep talking past INTERRUPT_AFTER_SEC, on_interrupt() fires
  once with the partial transcript so Mira can butt in.
- pause()/resume() let main.py mute the mic while Mira speaks,
  so she doesn't hear herself.

Also exposes transcribe() so non-mic audio sources (e.g. the Discord
voice adapter) can reuse the same Whisper model + CUDA setup without
opening the microphone.
"""

import queue
import threading
import time

import numpy as np
import sounddevice as sd

import os
import sys
from pathlib import Path

# Register pip-installed CUDA DLLs (cublas64_12.dll, cudnn*.dll)
_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
for _sub in ("cublas", "cudnn"):
    _p = _nvidia / _sub / "bin"
    if _p.is_dir():
        os.add_dll_directory(str(_p))
        os.environ["PATH"] = str(_p) + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

# ---------------- config ----------------
SAMPLE_RATE = 16000
BLOCK_SEC = 0.05            # mic callback chunk size
REFRESH_SEC = 0.7           # how often live partials update
END_SILENCE_SEC = 3.0       # trailing silence that ends an utterance
INTERRUPT_AFTER_SEC = 25.0  # monologue length that lets Mira interrupt
# On a roomy GPU bump these via env: MODEL_SIZE=medium/large-v3, COMPUTE_TYPE=float16.
# (The old 1660 Super had a broken fp16 path; int8/small were a VRAM compromise.)
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")   # "small" fast; "medium"/"large-v3" more accurate
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")            # "cpu" if VRAM gets tight
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")  # "float16" on a working-fp16 GPU
INPUT_DEVICE = None         # None = default mic; or device index/name
LANGUAGE = "en"

_vad_options = VadOptions(min_silence_duration_ms=500)

# ---------------- state ----------------
_model = None
_audio_q = queue.Queue()
_listening = threading.Event()   # cleared = mic muted (Mira talking)
_flush = threading.Event()
_running = False
_thread = None
_stream = None


def _ensure_model():
    """Load the Whisper model once (shared by the mic loop and transcribe())."""
    global _model
    if _model is None:
        print("(wernicke) loading whisper model...")
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("(wernicke) ready.")
    return _model


def _mic_callback(indata, frames, time_info, status):
    if _listening.is_set():
        _audio_q.put(indata[:, 0].copy())


def _transcribe(buf):
    segments, _ = _model.transcribe(
        buf,
        language=LANGUAGE,
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=True,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe(audio):
    """Transcribe a finalized utterance: 16 kHz mono float32 numpy array -> text.

    For audio sources that do their own capture/endpointing (e.g. Discord voice).
    Loads the shared model on first use; does NOT touch the microphone.
    """
    _ensure_model()
    if audio is None or len(audio) == 0:
        return ""
    return _transcribe(np.asarray(audio, dtype=np.float32))


def _worker(on_final, on_partial, on_interrupt):
    buf = np.zeros(0, dtype=np.float32)
    last_refresh = 0.0
    interrupted = False

    while _running:

        if _flush.is_set():
            _flush.clear()
            buf = np.zeros(0, dtype = np.float32)
            interrupted = False

        try:
            buf = np.concatenate([buf, _audio_q.get(timeout=0.1)])
        except queue.Empty:
            pass

        now = time.monotonic()
        if now - last_refresh < REFRESH_SEC:
            continue
        last_refresh = now

        speech = get_speech_timestamps(buf, _vad_options) if len(buf) else []
        if not speech:
            # no speech yet -> keep only the last second so buffer doesn't grow
            if len(buf) > SAMPLE_RATE:
                buf = buf[-SAMPLE_RATE:]
            continue

        text = _transcribe(buf)
        if on_partial and text:
            on_partial(text)

        silence = (len(buf) - speech[-1]["end"]) / SAMPLE_RATE
        spoken = (speech[-1]["end"] - speech[0]["start"]) / SAMPLE_RATE

        if silence >= END_SILENCE_SEC:
            buf = np.zeros(0, dtype=np.float32)
            interrupted = False
            if text:
                on_final(text)
        elif spoken >= INTERRUPT_AFTER_SEC and not interrupted and on_interrupt:
            interrupted = True
            on_interrupt(text)


def start(on_final, on_partial=None, on_interrupt=None):
    """Load the model, open the mic, begin listening."""
    global _running, _thread, _stream
    _ensure_model()
    _running = True
    _listening.set()
    _stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * BLOCK_SEC),
        device=INPUT_DEVICE,
        callback=_mic_callback,
    )
    _stream.start()
    _thread = threading.Thread(
        target=_worker, args=(on_final, on_partial, on_interrupt), daemon=True
    )
    _thread.start()


def pause():
    """Mute the mic (call while Mira is speaking)."""
    _listening.clear()
    with _audio_q.mutex:
        _audio_q.queue.clear()

def flush():
    """ Discard any captured-but-unfinalized audio (e.g. after an interruption)."""
    with _audio_q.mutex:
        _audio_q.queue.clear()
    _flush.set()


def resume():
    """Unmute the mic."""
    _listening.set()


def stop():
    global _running
    _running = False
    if _stream is not None:
        _stream.stop()
        _stream.close()