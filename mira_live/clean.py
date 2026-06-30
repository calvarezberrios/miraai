"""
Output hygiene — turn the model's text into clean SPOKEN lines.

The persona says "plain text only", but Hermes-3 (like most roleplay tunes) still sprinkles in
*stage directions*, markdown, and emoji. Those look wrong in the chat and, worse, get read aloud
by TTS later. This strips them to what she actually *says*.
"""

from __future__ import annotations

import re

# Action/stage-direction beats: *winks*, *sighs*, *wags tail* -> dropped (not spoken).
_ASTERISK_ACTION = re.compile(r"\*[^*\n]+\*")
# Emoji + pictographs + dingbats + symbol ranges + variation selector.
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"
    "\U00002022"           # bullet
    "]"
)


def clean(text: str) -> str:
    """Plain spoken text: no *actions*, no markdown emphasis, no emoji, tidy whitespace."""
    if not text:
        return ""
    t = _ASTERISK_ACTION.sub("", text)   # drop whole *action* beats first
    t = t.replace("*", "").replace("_", " ")  # stray emphasis markers
    t = _EMOJI.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
