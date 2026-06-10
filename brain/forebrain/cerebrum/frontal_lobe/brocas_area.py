"""
brocas_area.py - speech production (TTS) for Mira.

Anatomy: Broca's area (frontal lobe) = assembling words into produced speech.
Implementation: sends text to a locally-running GPT-SoVITS api_v2 server,
plays the returned audio, and serializes playback through queues so replies
never overlap (motor sequencing).

Latency design: a reply is split into sentences and pushed through a TWO-STAGE
pipeline -- a synth worker turns sentences into audio, a playback worker plays
them in order. Sentence N plays while sentence N+1 is still synthesizing, so
Mira starts talking after the FIRST sentence is ready, not the whole reply.

Start the engine first, in the GPT-SoVITS folder (its OWN bundled runtime):
    .\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880
"""

import io
import queue
import re
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
    "speed_factor": 1.12,
}

# --- Pipeline queues (motor sequencing) -----------------------------------
# _synth_q items: (text, should_play). _play_q items: audio bytes.
_synth_q = queue.Queue()
_play_q = queue.Queue()

_SENTENCE_RE = re.compile(r".*?[.!?](?:\s|$)|.+$", re.S)

# --- Speech cleaning ------------------------------------------------------
# Stage directions like *giggles* aren't spoken by TTS verbatim -- they'd be
# read out as the literal word. We convert laugh-type actions into vocal
# sounds and strip everything else that can't be voiced.

# Map the ACTION WORD (lowercased, no asterisks) to what Mira should vocalize.
# Match is substring-based, so "giggles softly" still hits "giggle".
SPOKEN_ACTIONS = {
    "giggle": "hehe",
    "giggles": "hehe",
    "laugh": "haha",
    "laughs": "haha",
    "chuckle": "heh",
    "chuckles": "heh",
    "giggling": "hehe",
    "laughing": "haha",
    "hums": "hmm",
    "hum": "hmm",
    "sighs": "haah",
    "sigh": "haah",
    "gasps": "gah",
    "gasp": "gah",
    "snickers": "heh",
}

# matches *...* and (...) and [...] stage-direction spans
_ACTION_SPAN_RE = re.compile(r"\*([^*]+)\*|\(([^)]*)\)|\[([^\]]*)\]")
# strips emoji / pictographs and most symbol blocks
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


def _replace_action(match):
    inner = (match.group(1) or match.group(2) or match.group(3) or "").lower()
    for word, sound in SPOKEN_ACTIONS.items():
        if word in inner:
            return sound
    return ""  # unspeakable stage direction -> drop entirely


def _clean_for_speech(text):
    # type: (str) -> str
    """Make text safe to vocalize: convert laugh-type actions to sounds,
    strip other stage directions, emoji, and leftover markdown."""
    text = _ACTION_SPAN_RE.sub(_replace_action, text)
    text = _EMOJI_RE.sub("", text)
    # hyphen INSIDE a word -> join the parts ("well-known" -> "wellknown")
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)
    # any remaining dash (spaced hyphen, en/em dash) -> pause, not "minus"
    text = re.sub(r"\s*[\u2012\u2013\u2014\u2015-]\s*", ", ", text)
    text = text.replace("*", "").replace("_", "").replace("`", "")
    text = re.sub(r"\s{2,}", " ", text)        # collapse doubled spaces
    text = re.sub(r"\s+([,.!?])", r"\1", text)  # tidy space-before-punct
    return text.strip()


def _split_sentences(text):
    # type: (str) -> list
    """Break a reply into sentence-ish chunks so the first can play ASAP."""
    parts = [m.group().strip() for m in _SENTENCE_RE.finditer(text)]
    return [p for p in parts if p]


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


def _synth_worker():
    while True:
        text, should_play = _synth_q.get()
        try:
            audio = _synthesize(text)
            if audio and should_play:
                _play_q.put(audio)      # hand off to playback stage
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
threading.Thread(target=_play_worker, daemon=True).start()


# --- Public API -----------------------------------------------------------
def say(text):
    # type: (str) -> None
    """Queue text for Mira to speak. Returns immediately; playback is serialized.
    The reply is split into sentences so the first one starts as soon as it's ready."""
    if not (text and text.strip()):
        return
    text = _clean_for_speech(text)
    if not text:
        return
    for sentence in _split_sentences(text):
        _synth_q.put((sentence, True))


def warmup():
    """Fire a throwaway synthesis so CUDA kernels/model are hot before the
    first real reply. Synthesizes but does not play. Returns immediately."""
    _synth_q.put(("Warming up.", False))


def wait_until_done():
    """Block until everything currently queued has finished speaking."""
    _synth_q.join()   # all sentences synthesized + handed to playback
    _play_q.join()    # all audio finished playing


if __name__ == "__main__":
    say("Hey there. If you can hear this, my voice is finally working.")
    wait_until_done()