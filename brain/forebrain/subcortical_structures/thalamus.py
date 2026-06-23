from collections import deque
from threading import Lock

MAX_TURNS = 100 # how many recenet messages Mira remembers

working_memory = deque(maxlen = MAX_TURNS) # Mira's working memory, stores recent messages for context

# A monotically increasing counter, bumped on every append (inbound or outbound).
# The subconscious watches this to tell when something NEW has arrived since it
# last looked, without having to diff the whole buffer. A lock keeps the deque +
# counter consistent across the main thread and the background subconscious thread.
_seq = 0
_lock = Lock()

def receive(user_message: str, speaker: str = None) -> list[dict]:
    """Route incoming message into working memory, return current context.

    When `speaker` is given (the Discord display name, or "You" locally) the turn is stored
    as "Speaker: text" so the brain can tell WHO said what across multiple people in a voice
    channel. The dict stays a plain {role, content} (no extra keys), so it's still a valid
    OpenAI message. Her own replies (remember_reply) stay unprefixed."""
    global _seq
    content = f"{speaker}: {user_message}" if speaker else user_message
    with _lock:
        working_memory.append({"role": "user", "content": content})
        _seq += 1
        return list(working_memory)

def remember_reply(reply: str) -> None:
    global _seq
    with _lock:
        working_memory.append({"role": "assistant", "content": reply})
        _seq += 1

def snapshot() -> tuple[list[dict], int]:
    """Return a consistent (context copy, sequence number) pair.

    The sequence number lets a caller (the subconscious) detect 'has anything been
    said since I last checked?' cheaply: if seq is unchanged, nothing is new."""
    with _lock:
        return list(working_memory), _seq

def awaiting_reply() -> bool:
    """True if the most recent thing said was an inbound message Mira hasn't
    responded to yet (i.e. the tail is a 'user' turn, not one of her own)."""
    with _lock:
        return bool(working_memory) and working_memory[-1]["role"] == "user"
