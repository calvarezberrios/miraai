"""
Output hygiene — turn the model's text into clean displayable/speakable lines.

Hermes-3 (like most roleplay tunes) sprinkles in stage directions, markdown, and emoji.
Policy (matches the persona + the main brain's prefrontal filters):
  * *wrapped actions* (*giggles*) are ALLOWED — kept in the text as body language; a TTS
    layer must skip them at speak time (they're stage directions, not speech).
  * BARE action beats ("fidgets with hair") are dropped — with no asterisks they'd be
    displayed and spoken as if they were words she said.
  * Markdown emphasis, emoji, and trailing questions go. Trailing *actions* are silent,
    so they don't shield a closing question from the no-questions rule.
"""

from __future__ import annotations

import re

# A properly *wrapped* stage direction — the sanctioned form, kept in the text.
_WRAPPED_ACTION_RE = re.compile(r"^\*[^*\n]+\*[.,!?~\s]*$")
# Emoji + pictographs + dingbats + symbol ranges + variation selector.
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200B-\U0000200D"  # zero-width space/non-joiner/joiner (emoji-strip leftovers)
    "\U00002060\U0000FEFF"   # word-joiner / BOM
    "\U00002022"             # bullet
    "]"
)


def clean(text: str) -> str:
    """Displayable text: *wrapped actions* kept, bare actions dropped, no markdown
    emphasis, no emoji, tidy whitespace, never ends on a spoken question."""
    if not text:
        return ""
    # Underscore emphasis -> plain; lone/unpaired asterisks (markdown bold leftovers etc.)
    # are cleaned per-sentence below so paired *action* wrapping survives.
    t = _EMOJI.sub("", text).replace("_", " ")
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return _drop_trailing_questions(_drop_bare_actions(t.strip()))


def strip_actions_for_tts(text: str) -> str:
    """What she actually SAYS: remove the *wrapped actions* (stage directions) entirely.
    Use this on the way into any TTS engine — never speak a gesture."""
    if not text:
        return ""
    t = re.sub(r"\*[^*\n]+\*", "", text)
    t = t.replace("*", "")
    return re.sub(r"\s{2,}", " ", t).strip()


# Bare (asterisk-less) stage directions — "fidgets with hair", "giggles nervously".
# A sentence that STARTS with a third-person action verb and stays short IS an action
# beat (she's the implied subject); "Looks like ..." is real speech and excluded.
_ACTION_VERBS = (
    "giggles", "blushes", "fidgets", "smiles", "laughs", "sighs", "nods", "shrugs",
    "looks", "glances", "twirls", "wags", "waves", "winks", "bites", "tucks", "brushes",
    "hides", "covers", "leans", "tilts", "shifts", "stares", "pouts", "mumbles",
    "whispers", "murmurs", "hums", "taps", "wraps", "hugs", "clutches", "blinks",
    "gazes", "fiddles", "squirms", "tugs", "pulls", "scratches", "rubs", "adjusts",
    "straightens", "shuffles", "curls", "buries", "peeks", "averts", "flushes",
    "stammers", "swallows", "exhales", "inhales", "shivers", "trembles", "grins",
    "smirks", "chuckles", "yawns", "stretches", "crosses", "clasps",
    "gasps", "frowns", "glares", "huffs", "sniffs", "sniffles", "gulps", "hesitates",
    "freezes", "perks", "beams", "slumps", "bounces", "cuddles", "snuggles", "fumbles",
    "paces", "sways", "tenses", "turns",
    "trails", "toys", "plays", "twiddles", "shakes", "rests", "folds", "wrings",
    "traces", "twists", "chews", "nibbles", "picks", "drums", "flips", "scoots",
    "settles", "plops",
    "types", "peers", "ponders", "snickers", "claps", "wiggles", "clears",
)
# Up to ~12 trailing words so full narrations match ("looks up from the book she is
# reading"); "looks/sounds like ..." and "turns out ..." are speech and excluded.
_BARE_ACTION_RE = re.compile(
    r"^(?:" + "|".join(_ACTION_VERBS) + r")(?!\s+(?:like|out)\b)(?:\s+[\w'’-]+){0,12}$", re.I)
_ACTION_START_RE = re.compile(
    r"^(?:" + "|".join(_ACTION_VERBS) + r")\b(?!\s+(?:like|out)\b)", re.I)


def _split_bare_action(fragment: str):
    """If the fragment BEGINS with a bare beat — commas allowed, possibly fused straight
    into speech ("blinks in confusion Hmm?") — split (beat, rest). Beats are lowercase;
    speech resumes at a Capitalized token. None if the fragment doesn't open on a beat."""
    f = (fragment or "").strip()
    if not f or f.startswith("*") or not _ACTION_START_RE.match(f):
        return None
    tokens = f.split()
    i = 1
    while i < len(tokens):
        core = tokens[i].strip(",;()—-")
        if core and (core[0].isupper() or core[0] in "\"'“”‘’*"):
            break
        i += 1
    beat = " ".join(tokens[:i]).rstrip(",;.!?~ ").strip()
    rest = " ".join(tokens[i:]).strip()
    return (beat, rest)


def _is_wrapped_action(fragment: str) -> bool:
    return bool(_WRAPPED_ACTION_RE.match((fragment or "").strip()))


def _is_bare_action(fragment: str) -> bool:
    f = (fragment or "").strip()
    if f.startswith("*"):
        return False        # wrapped form is allowed — keep it
    return bool(_BARE_ACTION_RE.match(f.strip(".,!?~*()[]— ").strip()))


def _drop_bare_actions(text: str) -> str:
    """(Name kept for compatibility) — bare beats are WRAPPED into *action* form rather
    than deleted, including beats fused into the front of a sentence."""
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        if not p.strip():
            continue
        split = _split_bare_action(p)
        if split:
            beat, rest = split
            out.append("*" + beat + "*" + ((" " + rest) if rest else ""))
        else:
            out.append(p)
    return " ".join(out).strip()


def _drop_trailing_questions(text: str) -> str:
    """Persona hard rule: never END on a spoken question. Trailing *wrapped actions* are
    silent, so they're transparent: "How about you? *giggles*" still ends on a question
    and the question goes. A reply that is entirely one question is kept (never mute)."""
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    while len(parts) > 1:
        i = len(parts) - 1
        while i >= 0 and _is_wrapped_action(parts[i]):
            i -= 1
        if i < 0:
            break
        if not parts[i].rstrip("\"'“”‘’)]").rstrip().endswith("?"):
            break
        parts.pop(i)
    return " ".join(parts)
