"""
brocas_area.py - speech production (TTS) for Mira.

Anatomy: Broca's area (frontal lobe) = assembling words into produced speech.
Implementation: sends text to a locally-running GPT-SoVITS api_v2 server,
plays the returned audio, and serializes playback through a queue so replies
never overlap (motor sequencing).

Start the engine first, in the GPT-SoVITS folder (its OWN bundled runtime):
    .\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880
"""

import io
import queue
import threading
from typing import Optional

import requests
import sounddevice as sd
import soundfile as sf

# --- Voice config ---------------------------------------------------------
API_URL = "http://127.0.0.1:9880/tts"

# Mira's cloned voice. Point ref_audio_path at the same 3-10s clip you
# tested with in the WebUI, and paste its exact transcript below.
VOICE = {
    "ref_audio_path": r"D:\aiproject\brain\forebrain\cerebrum\frontal_lobe\voice\sample_clip.mp3",
    "prompt_text": "You want to know all about me, huh? Well, let's just say I'm a bit of a handful.",
    "prompt_lang": "en",   # language spoken in the reference clip
    "text_lang": "en",     # language Mira speaks
}

# Generation tunables (sane defaults; lower temperature = steadier output)
PARAMS = {
    "text_split_method": "cut5",   # split on punctuation; good for sentences
    "batch_size": 1,
    "media_type": "wav",
    "streaming_mode": False,
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 1.0,
    "parallel_infer": False,
    "split_bucket": False,
}

# --- Playback queue (motor sequencing) ------------------------------------
_speech_q = queue.Queue()  # holds text strings waiting to be spoken


def _synthesize(text):
    # type: (str) -> Optional[bytes]
    payload = {
        "text": text,
        "text_lang": VOICE["text_lang"],
        "ref_audio_path": VOICE["ref_audio_path"],
        "prompt_text": VOICE["prompt_text"],
        "prompt_lang": VOICE["prompt_lang"],
    }
    payload.update(PARAMS)
    try:
        r = requests.post(API_URL, json=payload, timeout=60)
    except requests.RequestException as e:
        print("[brocas_area] request failed:", e)
        return None
    if r.status_code != 200 or not r.headers.get("Content-Type", "").startswith("audio"):
        print("[brocas_area] synth failed (%s): %s" % (r.status_code, r.text[:200]))
        return None
    return r.content


def _play(wav_bytes):
    # type: (bytes) -> None
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    sd.play(data, sr)
    sd.wait()  # block until this clip finishes, so the next one plays after it


def _worker():
    while True:
        text = _speech_q.get()
        try:
            audio = _synthesize(text)
            if audio:
                _play(audio)
        except Exception as e:
            print("[brocas_area] playback error:", e)
        finally:
            _speech_q.task_done()


threading.Thread(target=_worker, daemon=True).start()


# --- Public API -----------------------------------------------------------
def say(text):
    # type: (str) -> None
    """Queue text for Mira to speak. Returns immediately; playback is serialized."""
    if text and text.strip():
        _speech_q.put(text.strip())


def wait_until_done():
    """Block until everything currently queued has finished speaking."""
    _speech_q.join()


if __name__ == "__main__":
    say("Hey there. If you can hear this, my voice is finally working.")
    wait_until_done()