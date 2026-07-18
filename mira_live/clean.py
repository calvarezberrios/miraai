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
)
_BARE_ACTION_RE = re.compile(
    r"^(?:" + "|".join(_ACTION_VERBS) + r")(?!\s+like\b)(?:\s+[\w'’-]+){0,6}$", re.I)


def _is_wrapped_action(fragment: str) -> bool:
    return bool(_WRAPPED_ACTION_RE.match((fragment or "").strip()))


def _is_bare_action(fragment: str) -> bool:
    f = (fragment or "").strip()
    if f.startswith("*"):
        return False        # wrapped form is allowed — keep it
    return bool(_BARE_ACTION_RE.match(f.strip(".,!?~*()[]— ").strip()))


def _drop_bare_actions(text: str) -> str:
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = [p for p in parts if p.strip() and not _is_bare_action(p)]
    return " ".join(kept).strip()


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
