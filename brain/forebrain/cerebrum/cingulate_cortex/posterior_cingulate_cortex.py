"""
posterior_cingulate_cortex.py — Mira's subconscious (the default mode network).

Anatomy: the posterior cingulate cortex is the core hub of the brain's default
mode network — the circuit active when you are NOT locked onto a task: the mind
wandering, daydreaming, turning things over quietly in the background. This module
is that background mind. It runs on its own threads, always on, doing three jobs:

  1. LISTENING & DRAFTING (while someone is talking).
     As a person speaks, their words arrive as live partial transcripts. The
     subconscious keeps re-drafting a candidate reply against the latest partial
     (and any related memories), refining it as more is said. The instant the
     person stops, the foreground asks for that finished draft via take_draft()
     and speaks it — so there's almost no "transcribe, THEN think, THEN talk" gap.
     If no usable draft is ready, the foreground falls back to thinking fresh.

  2. LISTENING & DECIDING (un-addressed conversation).
     Talk she merely overhears (not addressed to her) is handed here via heard().
     She mulls it for a beat, then decides for herself WHETHER to speak, WHAT to
     say, and WHEN. If she has nothing to add, she stays quiet.

  3. WANDERING & DAYDREAMING (idle).
     When it's been quiet for a while, her mind drifts — reminiscing about a real
     memory or wondering about something she's never experienced. Every such
     thought goes to her private subconscious_log (NEVER the chat log), where she
     can look back over it alongside her memories; the recent ones quietly color
     her later replies. Once in a while a thought escapes out loud — but never
     while someone is speaking.

It never touches the speaker/avatar directly — main.py hands it a `speak` callback
that performs the full, serialized "Mira speaks" sequence, so a chime-in or a
spoken daydream can never overlap a foreground turn.
"""

from __future__ import annotations

import datetime
import os
import random
import threading
import time
from typing import Callable, List, Optional

from brain.forebrain.subcortical_structures import thalamus
from brain.forebrain.subcortical_structures.limbic_system import amygdala
from brain.forebrain.subcortical_structures.limbic_system.hippocampus import recall
from brain.forebrain.cerebrum.frontal_lobe import prefrontal_cortex
from brain.forebrain.cerebrum.cingulate_cortex import subconscious_log

# ---------------------------------------------------------------------------
# Tunables (env-overridable, like the rest of the brain)
# ---------------------------------------------------------------------------
TICK_SEC = float(os.environ.get("MIRA_SUB_TICK", "0.5"))            # background loop cadence
CONSIDER_DELAY_SEC = float(os.environ.get("MIRA_SUB_MULL", "1.5"))  # mull overheard talk this long before deciding
WANDER_AFTER_SEC = float(os.environ.get("MIRA_WANDER_AFTER", "60")) # idle silence before her mind drifts
WANDER_EVERY_SEC = float(os.environ.get("MIRA_WANDER_EVERY", "120")) # min gap between wandering thoughts (background, logged)
WANDER_SPEAK_CHANCE = float(os.environ.get("MIRA_WANDER_SPEAK_CHANCE", "0.12"))  # chance a given thought is voiced (rare)
WANDER_SPEAK_COOLDOWN_SEC = float(os.environ.get("MIRA_WANDER_SPEAK_COOLDOWN", "300"))  # min gap between spoken musings
# She won't VOICE a wandering thought if no human has talked (text or voice) within this
# long — she still thinks in the background (logged), she just doesn't talk to an empty
# room. ("No one ELSE has talked": her own speech/daydreams don't count toward this.)
WANDER_SHARE_SILENCE_SEC = float(os.environ.get("MIRA_WANDER_SHARE_SILENCE", "300"))
THOUGHTS_SURFACED = int(os.environ.get("MIRA_THOUGHTS_SURFACED", "4"))   # recent thoughts offered to a reply
LISTEN_TIMEOUT_SEC = float(os.environ.get("MIRA_LISTEN_TIMEOUT", "6"))   # drop "someone speaking" if partials stop w/o a final
MIN_DRAFT_WORDS = int(os.environ.get("MIRA_MIN_DRAFT_WORDS", "4"))       # don't burn a slow generation on a 2-word fragment
DRAFT_SETTLE_SEC = float(os.environ.get("MIRA_DRAFT_SETTLE", "0.8"))     # wait for the partial to stop growing before drafting
DRAFT_MAX_WAIT_SEC = float(os.environ.get("MIRA_DRAFT_MAX_WAIT", "4"))   # ...but on a long monologue, redraft at least this often
# Draft on a faster model so the reply keeps pace with speech; the final spoken
# line is this draft. Default: same model as a normal reply. Set e.g. qwen2.5:3b.
DRAFT_MODEL = os.environ.get("MIRA_DRAFT_MODEL", "").strip() or None

DEBUG = os.environ.get("MIRA_SUB_DEBUG", "").strip().lower() in ("1", "true", "yes")
# Her wandering thoughts are private/internal — only echo them to the console when
# debugging. (When she chooses to SPEAK a thought, that still surfaces as a normal
# spoken line via the foreground; this only governs the silent ones.)
LOG_THOUGHTS = DEBUG


def _dbg(msg: str) -> None:
    if DEBUG:
        print(f"[subconscious:dbg] {msg}")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_speak: Optional[Callable] = None     # main.py's serialized "Mira speaks" callback
_thread: Optional[threading.Thread] = None        # the wander/decide loop
_drafter: Optional[threading.Thread] = None       # the listen-and-draft loop
_running = threading.Event()
_paused = threading.Event()           # set while the DMN is suppressed (e.g. note-taking)

_lock = threading.Lock()
_consider_at: Optional[float] = None  # deadline to deliberate an overheard message (None = nothing pending)
_last_channel = "local"                # channel of the most recent thing she heard
_last_activity = 0.0                   # last time anything was said (heard OR her own speech)
_last_input = 0.0                      # last time a HUMAN talked (gates voicing daydreams; 0 = no one yet)
_last_wander = 0.0                     # last time her mind wandered at all
_last_wander_spoke = 0.0               # last time a wandering thought was voiced

# Listening / drafting state
_someone_speaking = False              # True between the first partial and the final
_last_partial = ""                     # newest live transcript of what's being said
_draft_req = 0                         # bumped each time the partial changes (a new draft is wanted)
_draft_done = -1                       # _draft_req value the current _draft was built from
_draft: Optional[dict] = None          # {"text": ..., "partial": ...} — the reply waiting in the wings
_draft_event = threading.Event()       # wakes the drafter when a fresh partial arrives

# Context the foreground passes through so daydreams/drafts share the same recap.
_session_recap: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API (called by main.py)
# ---------------------------------------------------------------------------

def start(speak: Callable, session_recap: Optional[str] = None) -> None:
    """Begin the subconscious. `speak(text, *, user_text=None, channel='local',
    source='reply')` must perform the full serialized speak sequence."""
    global _speak, _thread, _drafter, _last_activity, _session_recap
    _speak = speak
    _session_recap = session_recap
    _last_activity = time.time()
    if _thread is not None and _thread.is_alive():
        return
    _running.set()
    _thread = threading.Thread(target=_loop, name="subconscious", daemon=True)
    _thread.start()
    _drafter = threading.Thread(target=_drafter_loop, name="subconscious-drafter", daemon=True)
    _drafter.start()


def stop() -> None:
    """Quiet her mind (on shutdown)."""
    _running.clear()
    _draft_event.set()                 # unblock the drafter so it can exit
    for t in (_thread, _drafter):
        if t is not None:
            t.join(timeout=2.0)


def touch(now: Optional[float] = None) -> None:
    """Mark that something was just said. Resets the wandering timer so her mind
    doesn't drift the instant after a message. Call on every inbound event."""
    global _last_activity
    _last_activity = time.time() if now is None else now


def note_input(now: Optional[float] = None) -> None:
    """Mark that a HUMAN just talked (text or voice). Distinct from touch(): her own
    speech/daydreams don't count here. Used to gate whether she'll VOICE a wandering
    thought — she stays quiet (but keeps thinking privately) when no one's been active."""
    global _last_input
    _last_input = time.time() if now is None else now


def pause() -> None:
    """Suppress the default mode network — no drafting, deliberating, or wandering.
    Used while she is taking notes (DLPFC top-down control quieting the DMN). Drops any
    in-flight draft/deliberation so nothing escapes after the pause."""
    global _consider_at
    _paused.set()
    _reset_listening()
    with _lock:
        _consider_at = None


def resume() -> None:
    """Bring the subconscious back online (note-taking ended)."""
    global _last_activity
    _paused.clear()
    _last_activity = time.time()      # don't let her mind wander the instant she resumes


def observe_partial(text: str, channel: str = "local") -> None:
    """Feed the subconscious a live partial transcript of what's being said right
    now. She marks that someone's speaking and (re)drafts a reply to it in the
    background, so she's ready to answer the moment they stop."""
    global _someone_speaking, _last_channel, _last_partial, _draft_req
    if _paused.is_set():
        return
    text = (text or "").strip()
    if not text:
        return
    wake = False
    with _lock:
        _someone_speaking = True
        _last_channel = channel or "local"
        if text != _last_partial:
            _last_partial = text
            # Only spend a (slow) generation once there's enough to react to — but
            # always keep _last_partial current so the final still matches.
            if len(text.split()) >= MIN_DRAFT_WORDS:
                _draft_req += 1
                wake = True
    touch()
    note_input()                       # a live partial means a human is talking right now
    if wake:
        _draft_event.set()             # wake the drafter to refine the reply


def take_draft(final_text: str) -> Optional[str]:
    """The person stopped — hand back the reply she drafted while listening, if it
    still fits what they actually said. Returns None if the foreground should think
    fresh instead. Always ends the listening/drafting state."""
    with _lock:
        d = _draft
    reply = d["text"] if (d and _draft_usable(d, final_text)) else None
    _dbg(f"take_draft -> {'HIT' if reply else 'miss'} (final={final_text!r})")
    _reset_listening()
    return reply


def end_listening() -> None:
    """Drop the listening/drafting state without using the draft (un-addressed talk,
    interrupts, or a lost utterance)."""
    _reset_listening()


def heard(now: Optional[float] = None, channel: str = "local") -> None:
    """Hand the subconscious an overheard (un-addressed) message to mull over.
    Arms a debounced deliberation; rapid bursts coalesce into one decision."""
    global _consider_at, _last_channel
    if _paused.is_set():
        return
    now = time.time() if now is None else now
    with _lock:
        _consider_at = now + CONSIDER_DELAY_SEC
        _last_channel = channel or "local"
    touch(now)


def recent_thoughts(focus: Optional[str] = None) -> List[str]:
    """Her private stream of consciousness, offered to a reply so her daydreams
    can subtly surface. With `focus`, prefer thoughts that tie into it."""
    if focus:
        related = subconscious_log.related(focus, 3)
        if related:
            return related
    return subconscious_log.recent(THOUGHTS_SURFACED)


# ---------------------------------------------------------------------------
# Listen-and-draft loop  (job 1)
# ---------------------------------------------------------------------------

def _drafter_loop() -> None:
    _dbg("drafter started")
    last_started = 0.0
    while _running.is_set():
        if not _draft_event.wait(timeout=TICK_SEC):
            continue
        if not _running.is_set():
            break
        if _paused.is_set():
            _draft_event.clear()
            continue                   # note-taking: don't draft replies
        # Let the partial settle. While the person is mid-phrase the transcript keeps
        # growing every ~0.7s; drafting each fragment would waste slow generations on
        # text that's about to change. So we wait for a brief pause (a phrase/sentence
        # boundary, or the end-silence) and draft the most complete text then — except
        # on a long monologue, where we redraft periodically so the draft keeps up.
        with _lock:
            before = _draft_req
        time.sleep(DRAFT_SETTLE_SEC)
        _draft_event.clear()
        with _lock:
            req = _draft_req
            partial = _last_partial
            already = (_draft_done == req)
        if not partial or already:
            continue
        still_growing = (req != before)
        if still_growing and (time.monotonic() - last_started) < DRAFT_MAX_WAIT_SEC:
            _draft_event.set()        # keep waiting for it to settle
            continue
        last_started = time.monotonic()
        _dbg(f"drafting req={req} partial={partial!r}")
        try:
            text = _produce_draft(partial)
        except Exception as e:
            _dbg(f"draft error: {e}")
            continue
        if text:
            with _lock:
                _draft_globals_store(text, partial, req)
            _dbg(f"draft stored req={req} ({len(text)} chars)")
    _dbg("drafter stopped")


def _draft_globals_store(text: str, partial: str, req: int) -> None:
    global _draft, _draft_done
    _draft = {"text": text, "partial": partial}
    _draft_done = req


def _produce_draft(partial: str) -> str:
    """Draft a reply to what's being said so far, grounded in related memories."""
    history, _seq = thalamus.snapshot()
    provisional = history + [{"role": "user", "content": partial}]
    memories = recall(partial)
    if _session_recap:
        memories = [f"From our last session: {_session_recap}"] + memories
    return prefrontal_cortex.think(
        provisional, amygdala.color(), memories, _situation(listening=True),
        inner_thoughts=recent_thoughts(partial), model=DRAFT_MODEL,
    )


def _draft_usable(d: dict, final_text: str) -> bool:
    """Is the draft close enough to what they actually ended up saying? Partials
    are usually a growing prefix of the final, and the final is literally the last
    partial, so an exact/prefix match is the common (best) case."""
    p = _norm(d.get("partial", ""))
    f = _norm(final_text)
    if not p or not f:
        return False
    if p == f:
        return True
    if f.startswith(p) and len(p) >= 0.6 * len(f):
        return True
    pt, ft = set(p.split()), set(f.split())
    if not ft:
        return False
    overlap = len(pt & ft) / len(ft)
    return overlap >= 0.7 and len(p) >= 0.5 * len(f)


def _reset_listening() -> None:
    global _someone_speaking, _last_partial, _draft, _draft_req, _draft_done
    with _lock:
        _someone_speaking = False
        _last_partial = ""
        _draft = None
        _draft_req += 1            # invalidate any in-flight draft
        _draft_done = -1


# ---------------------------------------------------------------------------
# Wander / decide loop  (jobs 2 & 3)
# ---------------------------------------------------------------------------

def _loop() -> None:
    _dbg("loop started")
    while _running.is_set():
        try:
            _tick()
        except Exception as e:
            print(f"[subconscious] tick error: {e}")
        time.sleep(TICK_SEC)
    _dbg("loop stopped")


def _tick() -> None:
    global _consider_at, _someone_speaking
    if _paused.is_set():
        return                         # note-taking: the DMN is suppressed
    now = time.time()

    # Safety: if partials stopped without a final ever landing, stop "listening".
    if _someone_speaking and now - _last_activity >= LISTEN_TIMEOUT_SEC:
        _reset_listening()

    # 1) Reactive: did we overhear something we should mull over now?
    with _lock:
        due = _consider_at is not None and now >= _consider_at
        consider_deadline = _consider_at if due else None
    if due:
        with _lock:
            go = _consider_at == consider_deadline
            if go:
                _consider_at = None
        if go:
            _deliberate_reply()
        return

    # 2) Idle: quiet long enough (and no one mid-sentence) for her mind to wander?
    if _someone_speaking:
        return
    idle = now - _last_activity
    if idle >= WANDER_AFTER_SEC and now - _last_wander >= WANDER_EVERY_SEC:
        _wander_tick()


def _deliberate_reply() -> None:
    if not thalamus.awaiting_reply():
        return   # the foreground already answered it (or she replied) in the meantime
    history, _seq = thalamus.snapshot()
    latest = _last_user_text(history)
    if not latest:
        return

    memories = recall(latest)
    if _session_recap:
        memories = [f"From our last session: {_session_recap}"] + memories

    line = prefrontal_cortex.consider_speaking(
        history, amygdala.color(), memories, _situation(),
        inner_thoughts=recent_thoughts(latest),
    )
    if line:
        _emit(line, user_text=latest, source="chime-in")


def _wander_tick() -> None:
    global _last_wander, _last_wander_spoke
    now = time.time()
    history, _seq = thalamus.snapshot()

    # Reminisce about a real memory, or wonder about the unexperienced. Lean toward
    # curiosity unless there's something concrete to reminisce on.
    seed = _last_user_text(history) or "GameRaiderX and the things we've talked about"
    memories = recall(seed)
    if _session_recap:
        memories = [f"From our last session: {_session_recap}"] + memories
    mode = "memory" if (memories and random.random() < 0.5) else "curiosity"

    thought = prefrontal_cortex.wander(
        mode=mode,
        mood_flavor=amygdala.color(),
        situation=_situation(),
        memories=memories if mode == "memory" else None,
        recent_thoughts=subconscious_log.recent(THOUGHTS_SURFACED),
        history=history,                      # current session chat log (drift off it)
    )
    _last_wander = now
    if not thought:
        return

    # Thoughts live in her private log, never the chat log.
    subconscious_log.record(thought, mode=mode)
    if LOG_THOUGHTS:
        print(f"[mira's mind wanders] {thought}")

    # Once in a while a thought escapes out loud — but rarely, never while someone's
    # talking, never on top of an unanswered message, and NEVER to a room where no human
    # has spoken for a while (she keeps the thought private then). The thought above is
    # already saved to her subconscious_log regardless of whether she voices it.
    human_active = (_last_input > 0) and (now - _last_input <= WANDER_SHARE_SILENCE_SEC)
    if (human_active
            and now - _last_wander_spoke >= WANDER_SPEAK_COOLDOWN_SEC
            and random.random() < WANDER_SPEAK_CHANCE
            and not _someone_speaking
            and not thalamus.awaiting_reply()):
        _last_wander_spoke = now
        _emit(thought, user_text=None, source="daydream")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emit(text: str, *, user_text: Optional[str], source: str) -> None:
    """Speak a line through the foreground's serialized speak sequence."""
    if _speak is None or not text:
        return
    try:
        _speak(text, user_text=user_text, channel=_last_channel, source=source)
    except Exception as e:
        print(f"[subconscious] speak failed: {e}")
    touch()


def _last_user_text(history: List[dict]) -> str:
    for m in reversed(history):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split()).strip(" .!?,")


def _situation(listening: bool = False) -> str:
    """A light 'when' note for autonomous moments (the foreground builds a richer
    one for addressed turns from the actual event)."""
    dt = datetime.datetime.now()
    when = "Right now it is " + dt.strftime("%A, %B %d, %Y, at %I:%M %p").lstrip("0") + "."
    if listening:
        return when + " Someone is speaking to you right now; this is what they've said so far."
    return when + " No one is speaking to you directly at this moment."
