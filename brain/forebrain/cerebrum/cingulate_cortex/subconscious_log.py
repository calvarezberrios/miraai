"""
subconscious_log.py — Mira's private stream of consciousness.

Her wandering thoughts and daydreams are kept HERE, never in the conversation /
chat log (the thalamus working memory or the hippocampus session log). This
persists across runs, so she can look back over what's been on her mind the same
way she looks back over her memories.

It is deliberately kept SEPARATE from episodic memory (hippocampus): a daydream
is not a fact. Keeping the two apart means an idle "I wonder what snow tastes
like" can never leak into grounded recall and be mistaken for something that
actually happened. The conscious mind may glance at recent/related thoughts (they
are offered as clearly-labelled "private daydreams"), but they are never facts.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import List

_FILE = os.path.join("memory_store", "subconscious_log.jsonl")
_KEEP = 300                       # how many recent thoughts to hold in memory
_lock = threading.Lock()
_recent: deque = deque(maxlen=_KEEP)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "so", "to", "of", "in", "on",
    "for", "with", "is", "are", "was", "were", "be", "been", "it", "its", "i",
    "im", "you", "your", "me", "my", "we", "he", "she", "they", "them", "this",
    "that", "what", "just", "like", "about", "maybe", "would", "could", "hmm",
}


def _load() -> None:
    try:
        with open(_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    _recent.append(json.loads(line))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[subconscious_log] load skipped: {e}")


_load()


def record(text: str, mode: str = "idle") -> None:
    """Persist one wandering thought. `mode` is e.g. 'memory' or 'curiosity'."""
    text = (text or "").strip()
    if not text:
        return
    entry = {"ts": time.time(), "mode": mode, "text": text}
    with _lock:
        _recent.append(entry)
        try:
            os.makedirs(os.path.dirname(_FILE), exist_ok=True)
            with open(_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[subconscious_log] write skipped: {e}")


def recent(n: int = 6) -> List[str]:
    """The last `n` thoughts she had, newest last."""
    with _lock:
        return [e["text"] for e in list(_recent)[-n:]]


def _keywords(text: str) -> set:
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def related(query: str, n: int = 3) -> List[str]:
    """Past thoughts whose wording overlaps `query` — what's drifted through her
    mind that ties into what's being talked about now. Keyword overlap (no model
    call) so it's instant and safe to call mid-conversation."""
    q = _keywords(query or "")
    if not q:
        return []
    with _lock:
        scored = []
        for e in _recent:
            overlap = len(q & _keywords(e["text"]))
            if overlap:
                scored.append((overlap, e["ts"], e["text"]))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [t for _o, _ts, t in scored[:n]]


def review(n: int = 20) -> List[dict]:
    """Recent thoughts with their timestamps and modes — for her to look back over
    (or for tooling/inspection)."""
    with _lock:
        return list(_recent)[-n:]
