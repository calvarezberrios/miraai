"""
dorsolateral_prefrontal_cortex.py — the scribe (deliberate note-taking).

Anatomy: the dorsolateral prefrontal cortex (DLPFC) is the seat of WORKING MEMORY,
SUSTAINED ATTENTION, and top-down EXECUTIVE CONTROL. When you sit down to take notes
you are doing exactly this: holding incoming information in mind, deliberately and
selectively attending to it, and — crucially — SUPPRESSING your own urge to talk and
your mind's tendency to wander. It is a different job from the persona/reasoning
prefrontal_cortex ("the CEO") next door, and different from the hippocampus, which
forms memories automatically and internally; note-taking is deliberate, externalized,
and on-command.

So this module owns a NOTE-TAKING MODE. While a session is active:
  * Mira does not speak and does not chime in (main.py routes every heard utterance
    here instead of into a reply), and
  * her default mode network (the subconscious / posterior cingulate cortex) is PAUSED,
    so she doesn't daydream aloud — DLPFC top-down control quieting the DMN.

Every utterance is written to a .txt file LIVE (crash-safe). The LLM is only called at
RECAP and at FINALIZE (never per utterance) to respect the one-model-at-a-time / small
GPU constraint — so it organizes the raw transcript into topic- (or, for a TTRPG,
player+character-) structured notes plus a summary only when asked.

Comms are TEXT/CONSOLE ONLY: start/stop confirmations and recaps go through a `notify`
callback the caller supplies (console locally, the Discord text channel on Discord) —
never the voice, so the "she just listens" rule holds.

Public API (used by main.py):
    is_active() -> bool
    intercept(event, *, notify) -> bool      # the single gate at the top of handle_message
    finalize_if_active(*, notify) -> None     # flush an open session on shutdown
"""

from __future__ import annotations

import datetime
import os
import re
import threading
from typing import Callable, Dict, List, Optional, Tuple

from brain.forebrain.cerebrum.frontal_lobe import prefrontal_cortex
from brain.forebrain.cerebrum.cingulate_cortex import posterior_cingulate_cortex as _subconscious

# Where note files land. One .txt per session; gitignored (contains conversation content).
NOTES_DIR = os.environ.get("MIRA_NOTES_DIR", "notes")

# ---------------------------------------------------------------------------
# Session state (one session at a time). Guarded by _lock because intercept()
# can run on the Discord worker thread as well as the local main thread.
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_active = False
_profile = "default"                 # "default" | "ttrpg"
_file = None                         # open text handle (append, utf-8)
_path: Optional[str] = None
_topic_hint = ""
_mode_label = ""
_started_at: Optional[datetime.datetime] = None
_transcript: List[Tuple[datetime.datetime, str, str]] = []   # (ts, speaker, text)
_speakers: "set[str]" = set()
_cast: Dict[str, str] = {}           # player display name -> character


# ---------------------------------------------------------------------------
# Command patterns (matched on normalized text, leading "mira" address stripped)
# ---------------------------------------------------------------------------
# "take notes", "start taking notes", and qualified forms like "take dnd notes" /
# "take game notes" / "take meeting notes" (up to a few words between the verb and "notes").
_START_RE = re.compile(r"\b(?:take|taking|start|begin)\b(?:\s+\w+){0,3}\s+notes?\b|\bnote[- ]?taking\b", re.I)
_TTRPG_RE = re.compile(r"\b(ttrpg|rpg|dnd|d ?& ?d|d and d|dungeons|campaign|one[- ]?shot|the game|our game|game session|the session)\b", re.I)
_TOPIC_RE = re.compile(r"\bnotes?\s+(?:about|on|of|for|regarding|re)\s+(.+)$", re.I)
_STOP_RE = re.compile(r"\b(stop|end|finish|done|wrap up|wrap)\b.{0,12}\bnote", re.I)
_RECAP_RE = re.compile(r"\b(recap|summari[sz]e|summary|read back|what (?:do you have|have you got|have we got)(?: so far)?|catch me up)\b", re.I)
# Cast registration (TTRPG): third-person and first-person. Only honored on a message
# addressed to Mira, so ordinary in-play dialogue can't accidentally register a character.
_CAST3_RE = re.compile(r"\b(?P<player>[\w'’.-]+)\s+(?:is\s+|will\s+be\s+|gonna\s+be\s+)?(?:playing|plays|playing as|plays as)\s+(?P<character>.+)$", re.I)
_CAST1_RE = re.compile(r"\b(?:i'?m|i am|my character is|my character'?s|i'?ll play|i will play|i play)\s+(?:playing\s+|as\s+|named\s+|called\s+|the\s+)?(?P<character>.+)$", re.I)

_ADDRESS_RE = re.compile(r"^\s*(?:hey\s+|ok(?:ay)?\s+|yo\s+)?mira[\s,:-]+", re.I)


def _norm(text: str) -> str:
    return " ".join((text or "").split())


def _strip_address(text: str) -> str:
    return _ADDRESS_RE.sub("", text or "", count=1).strip()


def _slugify(text: str, fallback: str = "session") -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return text[:40].strip("-") or fallback


def _clean_character(name: str) -> str:
    name = (name or "").strip().strip(".,!?\"'")
    # drop a trailing aside like "playing Lyra, the elf ranger now" -> keep up to first clause
    name = re.split(r"\s+(?:and|but|so|because|today|tonight|now)\b", name, maxsplit=1, flags=re.I)[0]
    return name.strip().strip(".,!?\"'")[:60]


# ---------------------------------------------------------------------------
# Public: the gate
# ---------------------------------------------------------------------------

def is_active() -> bool:
    return _active


def intercept(event, *, notify: Callable[[str], None]) -> bool:
    """Single entry point at the top of handle_message. Returns True if this event
    was consumed by note-taking (caller must then return and do nothing else).

    - START phrase: begin a session (or note one's already running).
    - While active: STOP finalizes, RECAP delivers a recap, a TTRPG CAST line (only when
      addressed to Mira) registers a player->character, and anything else is recorded.
    - While inactive: STOP/RECAP/CAST are NOT commands — return False so normal chat runs.
    """
    text = getattr(event, "text", "") or ""
    speaker = getattr(event, "speaker", None) or "Someone"
    low = text.lower()
    addressed = bool(getattr(event, "mentioned", False)) or ("mira" in low)
    stripped = _strip_address(text)
    body = _norm(stripped)

    # NOTE on ordering: STOP/RECAP are checked before START, because "stop taking notes"
    # also contains the START phrase "taking notes" — STOP must win.
    if _active:
        if _STOP_RE.search(body):
            summary, path = _finalize()
            if path:
                notify(f"[Notes saved to {path}.]")
            if summary:
                notify(summary)
            notify("[Mira has stopped taking notes.]")
            return True

        if _RECAP_RE.search(body):
            notify("[Mira is pulling together a recap...]")
            try:
                recap = _recap()
            except Exception as e:
                recap = f"[recap failed: {e}]"
            notify(recap or "[Nothing noted yet.]")
            return True

        if _START_RE.search(body):
            notify("[Mira is already taking notes. Say 'stop taking notes' to finish.]")
            return True

        if _profile == "ttrpg" and addressed:
            cast_line = _try_register_cast(body, speaker)
            if cast_line:
                notify(cast_line)
                return True

        _record(speaker, stripped or text)
        return True

    # --- no active session: only a START phrase is a command ---
    if _START_RE.search(body) and not _STOP_RE.search(body):
        ttrpg = bool(_TTRPG_RE.search(body))
        m = _TOPIC_RE.search(body)
        topic = (m.group(1).strip() if m else "")
        # "for our game/campaign" is the profile signal, not a literal topic
        if ttrpg and topic and _TTRPG_RE.fullmatch(topic.strip()):
            topic = ""
        _start(topic_hint=topic, profile=("ttrpg" if ttrpg else "default"),
               mode_label=_mode_label_for(event), first_speaker=speaker)
        kind = "TTRPG session notes" if ttrpg else "notes"
        extra = f" on {topic}" if topic else ""
        notify(f"[Mira is now taking {kind}{extra}. She'll stay silent and just listen. "
               f"Say 'recap' for a summary, 'stop taking notes' to finish.]")
        return True

    return False   # not a START and no session -> let normal chat handle it


def finalize_if_active(*, notify: Callable[[str], None]) -> None:
    """Flush an open session on shutdown so notes are never lost."""
    if not _active:
        return
    summary, path = _finalize()
    if path:
        notify(f"[Mira saved the open note session to {path}.]")
    if summary:
        notify(summary)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def _mode_label_for(event) -> str:
    chan = str(getattr(event, "channel", "") or "")
    if chan == "discord_voice":
        return "Discord voice"
    if chan == "discord_text":
        return "Discord text"
    return "local mic"


def _start(*, topic_hint: str, profile: str, mode_label: str, first_speaker: str) -> None:
    global _active, _profile, _file, _path, _topic_hint, _mode_label, _started_at
    global _transcript, _speakers, _cast
    with _lock:
        _profile = profile
        _topic_hint = topic_hint
        _mode_label = mode_label
        _started_at = datetime.datetime.now()
        _transcript = []
        _speakers = set()
        _cast = {}

        os.makedirs(NOTES_DIR, exist_ok=True)
        provisional = _slugify(topic_hint, "ttrpg-session" if profile == "ttrpg" else "session")
        stamp = _started_at.strftime("%Y%m%d_%H%M%S")
        _path = os.path.join(NOTES_DIR, f"{provisional}_{stamp}.txt")
        _file = open(_path, "a", encoding="utf-8")
        header = (
            "=" * 60 + "\n"
            "Mira — Session Notes\n"
            f"Topic: {topic_hint or '(to be determined)'}\n"
            f"Profile: {'TTRPG' if profile == 'ttrpg' else 'general'}\n"
            f"Mode: {mode_label}\n"
            f"Started: {_started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            + "=" * 60 + "\n\n"
        )
        _file.write(header)
        _file.flush()
        _active = True

    # Quiet her mind: pause the default mode network so it doesn't draft/chime/daydream.
    try:
        _subconscious.pause()
    except Exception:
        pass


def _record(speaker: str, text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    ts = datetime.datetime.now()
    with _lock:
        if not _active or _file is None:
            return
        _transcript.append((ts, speaker, text))
        _speakers.add(speaker)
        try:
            _file.write(f"[{ts.strftime('%H:%M:%S')}] {speaker}: {text}\n")
            _file.flush()
        except Exception as e:
            print(f"[notes] write failed: {e}")


def _try_register_cast(body: str, speaker: str) -> Optional[str]:
    """Parse 'Alice plays Lyra' / 'I'm playing Lyra' (addressed to Mira). Returns a
    short confirmation string if a mapping was registered, else None."""
    m = _CAST3_RE.search(body)
    if m:
        player = m.group("player").strip()
        if player.lower() in ("i", "i'm", "im", "we", "you", "she", "he", "they"):
            player = speaker
        character = _clean_character(m.group("character"))
        if character:
            with _lock:
                _cast[player] = character
            return f"[Noted: {player} plays {character}.]"
    m = _CAST1_RE.search(body)
    if m:
        character = _clean_character(m.group("character"))
        if character:
            with _lock:
                _cast[speaker] = character
            return f"[Noted: {speaker} plays {character}.]"
    return None


def _finalize() -> Tuple[Optional[str], Optional[str]]:
    """Close out the session: write the organized body + summary + footer, close the
    file, rename it to include the derived main topic, resume the subconscious.
    Returns (summary_text_or_None, final_path_or_None)."""
    global _active, _file, _path
    # snapshot under lock, then do the (slow) LLM work unlocked
    with _lock:
        if not _active:
            return (None, None)
        transcript = list(_transcript)
        profile = _profile
        cast = dict(_cast)
        topic_hint = _topic_hint
        started = _started_at
        speakers = sorted(_speakers)
        path = _path
        f = _file

    summary = None
    organized = None
    slug = None
    if transcript:
        transcript_text = _transcript_text(transcript)
        cast_text = _cast_block(cast)
        try:
            organized = _organize(transcript_text, profile, cast_text)
        except Exception as e:
            organized = f"(could not organize notes automatically: {e})"
        try:
            summary = _summarize(transcript_text, profile, cast_text)
        except Exception as e:
            summary = None
        try:
            slug = _topic_slug(summary or transcript_text, profile)
        except Exception:
            slug = None

    ended = datetime.datetime.now()
    with _lock:
        try:
            if f is not None:
                if not transcript:
                    f.write("(no audio was captured during this session)\n")
                else:
                    title = ("SESSION NOTES (by player & character)"
                             if profile == "ttrpg" else "NOTES BY TOPIC")
                    f.write("\n" + "=" * 60 + "\n" + title + "\n" + "=" * 60 + "\n")
                    f.write((organized or "").strip() + "\n")
                    f.write("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60 + "\n")
                    f.write((summary or "(no summary)").strip() + "\n")
                dur = _human_duration(started, ended) if started else "?"
                f.write("\n" + "-" * 60 + "\n")
                if speakers:
                    f.write(f"Participants: {', '.join(speakers)}\n")
                if cast:
                    f.write("Cast: " + "; ".join(f"{p} = {c}" for p, c in cast.items()) + "\n")
                f.write(f"Started {started.strftime('%Y-%m-%d %H:%M:%S') if started else '?'} "
                        f"— Ended {ended.strftime('%Y-%m-%d %H:%M:%S')}  "
                        f"({dur}, {len(transcript)} lines)\n")
                f.flush()
                f.close()
        except Exception as e:
            print(f"[notes] finalize write failed: {e}")

        final_path = path
        if path and slug:
            stamp = (started or ended).strftime("%Y%m%d_%H%M%S")
            target = os.path.join(NOTES_DIR, f"{slug}_{stamp}.txt")
            target = _unique_path(target)
            if target != path:
                try:
                    os.replace(path, target)
                    final_path = target
                except Exception as e:
                    print(f"[notes] rename failed: {e}")

        _active = False
        _file = None
        _path = None

    try:
        _subconscious.resume()
    except Exception:
        pass
    return (summary, final_path)


def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{base}-{i}{ext}"):
        i += 1
    return f"{base}-{i}{ext}"


# ---------------------------------------------------------------------------
# LLM helpers — neutral note-taker prompts, faithful to the transcript. Reuse the
# same Ollama client + model the rest of the brain uses (keeps that model hot).
# ---------------------------------------------------------------------------

def _llm(system: str, user: str, max_tokens: int, temperature: float = 0.25) -> str:
    resp = prefrontal_cortex.client.chat.completions.create(
        model=prefrontal_cortex.MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        # Honor MIRA_NO_THINK so a reasoning model (Qwen3 turbo) doesn't bury the notes
        # under a <think> block. No-op on servers/models that ignore it (qwen2.5 on Ollama).
        extra_body=prefrontal_cortex._EXTRA,
    )
    return (resp.choices[0].message.content or "").strip()


# --- context-aware chunking -------------------------------------------------
# The local model has a fixed context window (llama.cpp n_ctx, default 8192). A long
# session's transcript can be far bigger than that, so organize/summarize/recap can't
# send it in one shot — they map over context-sized chunks and reduce the results.
# Override the window to match your server's -c / n_ctx with MIRA_NOTES_CTX.
_CTX_TOKENS = int(os.environ.get("MIRA_NOTES_CTX", "8192"))


def _est_tokens(text: str) -> int:
    """Rough token estimate. Transcripts carry timestamps, punctuation, and multi-part
    speaker names ("Lucien | Torvik || Ryan") that tokenize much denser than prose —
    measured ~2.7 chars/token here — so we assume 2.6 chars/token and round up. Deliberately
    pessimistic: better to under-fill the window than overflow it and get a 400."""
    return max(1, int(len(text) / 2.6) + 1)


def _input_budget(system: str, cast_text: str, max_tokens: int) -> int:
    """How many transcript tokens we can afford in one call: the context window minus the
    system prompt, the cast block, the reserved output, and a safety margin."""
    overhead = _est_tokens(system) + _est_tokens(cast_text) + max_tokens + 512
    return max(512, _CTX_TOKENS - overhead)


def _chunk_lines(text: str, budget_tokens: int) -> List[str]:
    """Split text into chunks each within budget_tokens, never splitting a line. A single
    line longer than the budget is hard-split as a last resort."""
    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for ln in text.split("\n"):
        t = _est_tokens(ln)
        if t > budget_tokens:                      # pathologically long single line
            if cur:
                chunks.append("\n".join(cur))
                cur, cur_tok = [], 0
            step = max(1, int(budget_tokens * 3.5))
            for i in range(0, len(ln), step):
                chunks.append(ln[i:i + step])
            continue
        if cur and cur_tok + t > budget_tokens:
            chunks.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(ln)
        cur_tok += t
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _transcript_text(transcript: List[Tuple[datetime.datetime, str, str]]) -> str:
    return "\n".join(f"[{ts.strftime('%H:%M:%S')}] {sp}: {tx}" for ts, sp, tx in transcript)


def _cast_block(cast: Dict[str, str]) -> str:
    if not cast:
        return ""
    return "Known cast (player -> character):\n" + "\n".join(
        f"- {p} plays {c}" for p, c in cast.items()
    ) + "\n\n"


_ORG_DEFAULT = (
    "You are a careful note-taker. You are given a timestamped transcript of audio that "
    "was heard, labeled by speaker. Produce clean, well-organized notes in markdown.\n"
    "- Group related points under short topic headings (## Heading).\n"
    "- Under each heading, list concise bullet points of the key information.\n"
    "- Pull out decisions made, questions raised, and action items into their own bullets.\n"
    "- Attribute a point to the speaker when it matters.\n"
    "- Be faithful: include ONLY what was actually said; never invent anything.\n"
    "Output only the notes."
)

_ORG_TTRPG = (
    "You are the table scribe for a tabletop RPG (TTRPG) session. You are given a "
    "timestamped transcript labeled by the real-world SPEAKER names.\n"
    "First, identify each player's CHARACTER. Use the cast list provided if present; for "
    "anyone not listed, infer their character from how names are used in play — a Game "
    "Master / DM narrates the world and voices NPCs, while players speak and act as their "
    "own characters. If you genuinely cannot tell, say so rather than guessing wildly.\n"
    "Then organize the notes in markdown with these sections:\n"
    "## Cast & Party — list 'Player — Character (role/class if known)'.\n"
    "## Story So Far — the key events, in the order they happened.\n"
    "## NPCs — notable non-player characters and what is known about them.\n"
    "## Decisions & Rolls — important choices and notable dice outcomes.\n"
    "## Quests & Objectives — goals, leads, and open threads.\n"
    "## Loot & Rewards — items, gold, or boons gained.\n"
    "Attribute actions to the character (and player) responsible. Be faithful: include "
    "ONLY what actually happened in the transcript; do not invent lore. Output only the notes."
)


_REDUCE_TTRPG = (
    "You are the table scribe for a tabletop RPG (TTRPG) session. Below are notes written "
    "from CONSECUTIVE parts of the SAME session, in order. Merge them into one consolidated "
    "set of notes, de-duplicating repeated points and keeping events in order. Use exactly "
    "these markdown sections:\n"
    "## Cast & Party — 'Player — Character (role/class if known)'.\n"
    "## Story So Far — the key events, in order.\n"
    "## NPCs — notable non-player characters and what is known.\n"
    "## Decisions & Rolls — important choices and notable dice outcomes.\n"
    "## Quests & Objectives — goals, leads, open threads.\n"
    "## Loot & Rewards — items, gold, or boons gained.\n"
    "Be faithful to the notes; do not invent anything. Output only the merged notes."
)

_REDUCE_DEFAULT = (
    "Below are notes written from CONSECUTIVE parts of the SAME session, in order. Merge "
    "them into one clean, de-duplicated set of markdown notes: group related points under "
    "short ## headings, keep concise bullets, and pull out decisions, questions, and action "
    "items. Be faithful to the notes; do not invent anything. Output only the merged notes."
)


def _reduce_notes(notes_text: str, profile: str, cast_text: str) -> str:
    """Merge partial notes (from consecutive transcript chunks) into one document. If the
    partials themselves overflow the window, merge them in groups and repeat."""
    system = _REDUCE_TTRPG if profile == "ttrpg" else _REDUCE_DEFAULT
    budget = _input_budget(system, cast_text, max_tokens=900)
    guard = 0
    while _est_tokens(notes_text) > budget and guard < 5:
        groups = _chunk_lines(notes_text, budget)
        notes_text = "\n\n".join(
            _llm(system, cast_text + "Partial notes:\n" + g, max_tokens=600) for g in groups
        )
        guard += 1
    return _llm(system, cast_text + "Partial notes:\n" + notes_text, max_tokens=900)


def _organize(transcript_text: str, profile: str, cast_text: str) -> str:
    system = _ORG_TTRPG if profile == "ttrpg" else _ORG_DEFAULT
    budget = _input_budget(system, cast_text, max_tokens=900)
    if _est_tokens(transcript_text) <= budget:
        return _llm(system, cast_text + "Transcript:\n" + transcript_text, max_tokens=900)
    # Too long for one pass: organize each context-sized chunk, then merge the partials.
    chunks = _chunk_lines(transcript_text, budget)
    partials = []
    for i, ch in enumerate(chunks, 1):
        part = _llm(system, cast_text + f"Transcript (part {i} of {len(chunks)}):\n" + ch,
                    max_tokens=600)
        if part:
            partials.append(part)
    if not partials:
        return ""
    if len(partials) == 1:
        return partials[0]
    return _reduce_notes("\n\n".join(partials), profile, cast_text)


_PARTIAL_SUMMARY_SYS = (
    "You are a note-taker. Faithfully list the key things covered in this part of a session "
    "as a few concise bullet points — events, decisions, names, and open threads only. Do not "
    "invent anything; output only the bullets."
)


def _condense_for_summary(transcript_text: str, cast_text: str, budget: int,
                          temperature: float) -> str:
    """Collapse an over-long transcript into a small set of faithful bullets that fits the
    window, so a final summary/recap pass can run over it. Returns the transcript unchanged
    when it already fits."""
    if _est_tokens(transcript_text) <= budget:
        return transcript_text
    chunks = _chunk_lines(transcript_text, budget)
    notes = []
    for i, ch in enumerate(chunks, 1):
        n = _llm(_PARTIAL_SUMMARY_SYS,
                 cast_text + f"Transcript (part {i} of {len(chunks)}):\n" + ch,
                 max_tokens=220, temperature=temperature)
        if n:
            notes.append(n)
    combined = "\n".join(notes)
    guard = 0
    while _est_tokens(combined) > budget and guard < 5:
        groups = _chunk_lines(combined, budget)
        combined = "\n".join(
            _llm(_PARTIAL_SUMMARY_SYS, "Notes:\n" + g, max_tokens=220, temperature=temperature)
            for g in groups
        )
        guard += 1
    return combined


def _summarize(transcript_text: str, profile: str, cast_text: str) -> str:
    if profile == "ttrpg":
        system = (
            "Summarize this TTRPG session in 2-5 sentences as a 'previously, on...' recap "
            "for the players next time: where the party is, what just happened, and any "
            "cliffhanger or open objective. Refer to characters by name. Be faithful and concise."
        )
    else:
        system = (
            "Summarize this session in 2-5 sentences: the main subject, the key points, and "
            "any decisions or action items. Be faithful and concise; third person."
        )
    budget = _input_budget(system, cast_text, max_tokens=220)
    body = _condense_for_summary(transcript_text, cast_text, budget, temperature=0.3)
    out = _llm(system, cast_text + "Transcript:\n" + body, max_tokens=220, temperature=0.3)
    return "" if out.upper().startswith("NOTHING") else out


def _recap() -> str:
    with _lock:
        transcript = list(_transcript)
        profile = _profile
        cast = dict(_cast)
    if not transcript:
        return "[Nothing noted yet.]"
    transcript_text = _transcript_text(transcript)
    cast_text = _cast_block(cast)
    who = "Refer to characters by name. " if profile == "ttrpg" else ""
    system = (
        "You are a note-taker giving a quick interim recap of what has been heard so far. "
        "From the transcript, output a short markdown recap: 3-6 bullet points of the key "
        f"things covered, then a final line 'Summary: <one sentence>'. {who}"
        "Be faithful — only what was actually said. Keep it brief."
    )
    budget = _input_budget(system, cast_text, max_tokens=320)
    body_in = _condense_for_summary(transcript_text, cast_text, budget, temperature=0.3)
    body = _llm(system, cast_text + "Transcript:\n" + body_in, max_tokens=320, temperature=0.3)
    return "[Recap so far]\n" + body


def _topic_slug(text: str, profile: str) -> Optional[str]:
    hint = ("a short title for this RPG session (a location, quest, or what happened)"
            if profile == "ttrpg" else "a short title for the main topic of these notes")
    system = (f"Give {hint}. Reply with ONLY 2 to 4 words, no quotes and no punctuation.")
    raw = _llm(system, text[:2000], max_tokens=16, temperature=0.2)
    raw = raw.splitlines()[0] if raw else ""
    slug = _slugify(raw, "")
    return slug or None


def _human_duration(start: datetime.datetime, end: datetime.datetime) -> str:
    secs = max(0, int((end - start).total_seconds()))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
