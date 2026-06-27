"""
coordinator.py — the cerebellum: coordination, timing, and smoothing of movement.

Brain analogy: the cerebellum doesn't *decide* to move (that's the cortex), it
makes movement smooth and well-timed. It sits between the brain's movement intent
and the raw avatar output (motor_cortex) and does three small jobs:

  1. Smooths the lip-sync signal (fast attack / slower decay) before it hits the
     socket, so the mouth tracks speech energy without chattering.
  2. Maps Mira's SPOKEN words to a fitting one-shot body gesture (a greeting -> wave,
     a surprise -> surprised, ...). Gestures are driven by what she's actually
     saying — never fired at random.
  3. Tracks her persistent activity STATE — idle / talking / thinking — so the body always
     reflects what she's doing (talking while she speaks, thinking while the viewer talks and
     she drafts, idle otherwise). Deduped so only transitions are pushed.
  4. Forwards mood -> face.

The avatar's living idle and talking motion (breathing, weight-shift, head/arm motion) are
procedural in the renderer; thinking is a looping clip. One-shot gestures blend in over the
current state and auto-return. All calls forward to motor_cortex, a safe no-op if the avatar
isn't up.
"""

from __future__ import annotations

import re

from brain.forebrain.cerebrum.frontal_lobe import motor_cortex

# --- lip-sync smoothing ------------------------------------------------------
# Mouth opens fast on a syllable (attack) and closes a little slower (decay). Kept snappy so
# the mouth tracks the audio tightly — this is the only low-pass between the speech envelope
# and the viseme (the renderer now follows the mouth target almost immediately).
LIP_ATTACK = 0.9
LIP_DECAY = 0.45

# --- spoken content -> body gesture ------------------------------------------
# Scanned against her reply when she starts speaking; the FIRST group that
# matches plays its gesture (one per reply). If nothing matches she just stands
# and talks (idle motion + lips), facing the camera. Order = priority.
_SPEECH_GESTURES = [
    ("wave",      [r"\bhi\b", r"\bhey\b", r"hello", r"\bbye\b", r"goodbye", r"see ya", r"see you", r"\blater\b"]),
    ("clapping",  [r"congrat", r"well done", r"good job", r"\byay\b", r"\bwoo+\b", r"bravo", r"amazing"]),
    ("flirty",    [r"\bcute\b", r"\blove\b", r"darling", r"sweetheart", r"\bblush", r"\bkiss"]),
    ("surprised", [r"no way", r"\bwhat[?!]", r"really\?", r"\bwow\b", r"\bomg\b", r"whoa", r"oh my"]),
    ("jump",      [r"let'?s go", r"so excited", r"can'?t wait", r"\bhyped\b", r"woohoo"]),
    ("angry",     [r"\bugh\b", r"so annoying", r"knock it off", r"\brude\b", r"shut it"]),
    ("sad",       [r"\bsorry\b", r"so sad", r"miss you", r"\baww+\b"]),
    ("sleepy",    [r"\btired\b", r"sleepy", r"\byawn", r"so bored", r"\bbored\b"]),
]
_SPEECH_GESTURES = [(g, [re.compile(p, re.I) for p in pats]) for g, pats in _SPEECH_GESTURES]

_lip_value = 0.0


# --- persistent activity (idle / talking / thinking) -------------------------
# A small state so the body always reflects WHAT she's doing, blended in the renderer:
#   idle     = resting (living procedural motion)
#   talking  = she's speaking (livelier, voice-driven motion + lip-sync + content gestures)
#   thinking = the viewer is talking and she's listening / drafting her reply
# Deduped so we only push a change on transition (partials would otherwise re-send every tick).
_activity = "idle"


def _set_activity(state: str) -> None:
    global _activity
    if state != _activity:
        _activity = state
        motor_cortex.set_state(state)


def talking() -> None:
    """She's speaking now -> talking motion."""
    _set_activity("talking")


def thinking() -> None:
    """The viewer is talking and she's listening / drafting -> thinking pose."""
    _set_activity("thinking")


def go_idle() -> None:
    """Nothing happening -> return to the living idle stance."""
    _set_activity("idle")


# --- mood / face -------------------------------------------------------------
def set_mood(mood: str) -> None:
    """Mood drives the face."""
    motor_cortex.set_mood(mood or "neutral")


# --- gestures ----------------------------------------------------------------
def gesture(name: str) -> None:
    """Play a specific one-shot body gesture (escape hatch)."""
    if name:
        motor_cortex.play_gesture(name)


def gesture_for_speech(text: str) -> None:
    """Play a gesture that fits what she's about to say, if any. Called when she
    starts speaking; the gesture blends in over the idle stance and auto-returns."""
    if not text:
        return
    for name, pats in _SPEECH_GESTURES:
        if any(p.search(text) for p in pats):
            motor_cortex.play_gesture(name)
            return


def speaking_stopped() -> None:
    """Close the mouth when she finishes a line."""
    global _lip_value
    _lip_value = 0.0
    motor_cortex.lipsync(0.0)


# --- lip-sync ----------------------------------------------------------------
def lip(level: float) -> None:
    """Smooth a raw speech-amplitude level (0..1) and drive the mouth viseme.

    Called rapidly from the TTS playback thread. Fast attack / slower decay so
    the mouth tracks syllables without flickering on noise."""
    global _lip_value
    try:
        level = max(0.0, min(1.0, float(level)))
    except (TypeError, ValueError):
        return
    k = LIP_ATTACK if level > _lip_value else LIP_DECAY
    _lip_value += (level - _lip_value) * k
    motor_cortex.lipsync(_lip_value)
