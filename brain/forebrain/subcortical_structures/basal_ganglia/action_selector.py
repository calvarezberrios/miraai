"""
Basal ganglia — action selection.

Decides what Mira does with a message she heard: reply, or stay quiet. She still
HEARS everything (short-term context updates upstream) — this only gates SPEAKING.

should_respond() returns one of four tiers via Decision.reason:
  - "addressed"    -> reply now (she was @-mentioned / DM'd / replied-to / named)
  - "consider"     -> un-addressed, but a conversation is active in this channel;
                      the caller should run a relevance check (a quick LLM yes/no)
                      and reply only if the message is actually related/continuing
  - "interjection" -> reply now (rare, rate-limited, unprompted chime-in)
  - "ignore"       -> stay quiet

The caller calls mark_engaged(channel) whenever she ACTUALLY speaks. That keeps
the active-conversation window alive, so follow-up messages keep coming back as
"consider" and she can sustain a topic without anyone re-addressing her. The
window lapses on its own once the conversation goes quiet.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass

# --- who she answers to by name ----------------------------------------------
NAME = "Mira"
ALIASES = ()                  # extra names she answers to, e.g. ("mir", "foxy")

# --- conversation continuity --------------------------------------------------
ACTIVE_WINDOW_SEC = 180.0     # after she speaks, she stays "in the conversation"
                              # this long, relevance-checking un-addressed messages

# --- unprompted interjections -------------------------------------------------
RANDOM_CHANCE = 0.05          # chance to chime in on an un-addressed, out-of-window msg
RANDOM_COOLDOWN_SEC = 240.0   # ...at most once per this long, per channel

# per-channel state
_last_spoke = {}              # channel -> time she last replied (drives the window)
_last_random = {}             # channel -> time of her last random interjection

_name_pattern = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in (NAME, *ALIASES) if n) + r")\b",
    re.IGNORECASE,
)


@dataclass
class Decision:
    respond: bool
    reason: str


def name_mentioned(text: str) -> bool:
    return bool(text and _name_pattern.search(text))


def mark_engaged(channel, now=None) -> None:
    """Call whenever Mira actually speaks — keeps the active window alive."""
    _last_spoke[channel] = time.time() if now is None else now


def should_respond(text, *, mentioned=False, reply_to_her=False,
                   channel="", now=None) -> Decision:
    now = time.time() if now is None else now

    # 1) directly addressed -> reply
    if mentioned or reply_to_her or name_mentioned(text):
        return Decision(True, "addressed")

    # 2) conversation active -> hand it to the caller for a relevance check
    if now - _last_spoke.get(channel, 0.0) <= ACTIVE_WINDOW_SEC:
        return Decision(False, "consider")

    # 3) otherwise -> rare, rate-limited unprompted chime-in
    if now - _last_random.get(channel, 0.0) >= RANDOM_COOLDOWN_SEC:
        if random.random() < RANDOM_CHANCE:
            _last_random[channel] = now
            return Decision(True, "interjection")

    return Decision(False, "ignore")


def disengage(channel) -> None:
    """Drop a conversation early (e.g. on an explicit 'stop')."""
    _last_spoke.pop(channel, None)