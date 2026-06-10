"""Hippocampus -- long-term episodic memory + consolidation.

Forms and recalls memories that persist across sessions, beyond the
working memory held in the thalamus. Backed by a local Chroma vector
store; embeddings are served locally by Ollama (nomic-embed-text) through
the same OpenAI-compatible client the rest of the brain uses.

Two flavors of memory formation:
  * consolidate()        -- ongoing: pulls atomic durable facts from the
                            recent buffer (the "sleep" step), Mira's call.
  * summarize_session()  -- at session end: a short narrative recap of the
                            whole run, so Mira can recall "what happened
                            last time".
"""

import time
import uuid
from datetime import datetime

import chromadb
from openai import OpenAI

# Same local Ollama endpoint the chat model uses.
_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"  # used for consolidation / summary reasoning

# On-disk store -- survives restarts. Folder is created on first run.
_db = chromadb.PersistentClient(path="./memory_store")
_collection = _db.get_or_create_collection(name="mira_memory")


def _embed(text):
    """Turn text into a vector using the local embedding model."""
    resp = _client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def remember(text, kind="fact"):
    """Store a long-term memory. `kind` tags it (fact, session_summary, ...)."""
    _collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[_embed(text)],
        documents=[text],
        metadatas=[{"kind": kind, "ts": time.time()}],
    )


def recall(query, n=3):
    """Return up to `n` stored memories most relevant to `query`."""
    count = _collection.count()
    if count == 0:
        return []
    results = _collection.query(
        query_embeddings=[_embed(query)],
        n_results=min(n, count),
        include=["documents"],
    )
    return results["documents"][0]


def forget_all():
    """Wipe all long-term memories."""
    global _collection
    _db.delete_collection("mira_memory")
    _collection = _db.get_or_create_collection("mira_memory")


# --- consolidation: Mira decides what atomic facts are worth keeping -----

_buffer = []       # exchanges since last consolidation (cleared by consolidate)
_session_log = []  # full transcript for this run (cleared by summarize_session)

_CONSOLIDATION_PROMPT = (
    "You are the memory-formation process for a VTuber named Mira. "
    "Read the conversation excerpt and extract only durable facts worth "
    "remembering long-term: people's names, preferences, recurring viewers, "
    "promises made, and notable events. Ignore small talk, jokes, flirting, "
    "and anything trivial or fleeting. Write each fact as one short, standalone "
    "sentence in the third person (e.g. \"The user's name is Sam.\" or "
    "\"The user is learning to play guitar.\"). One fact per line, no numbering "
    "or bullets. If nothing is worth remembering, reply with exactly: NOTHING"
)

_SESSION_SUMMARY_PROMPT = (
    "You are the memory-formation process for a VTuber named Mira. "
    "Summarize this whole streaming session in 2-4 short sentences for Mira's "
    "own future reference: the main topics, who showed up, the general mood, "
    "and anything notable or promised. Write in the third person, past tense "
    "(e.g. \"Talked with Sam about his new cat; chat was upbeat.\"). "
    "If the session was empty or trivial, reply with exactly: NOTHING"
)


def observe(user_message, reply):
    """Record one exchange into both the consolidation buffer and session log."""
    _buffer.append({"role": "user", "content": user_message})
    _buffer.append({"role": "assistant", "content": reply})
    _session_log.append({"role": "user", "content": user_message})
    _session_log.append({"role": "assistant", "content": reply})


def _digest(prompt, log, temperature):
    """Run the LLM over a transcript with the given system prompt."""
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in log)
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
        max_tokens=200,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def consolidate():
    """Digest the buffer into durable atomic memories. Returns stored facts."""
    if not _buffer:
        return []
    raw = _digest(_CONSOLIDATION_PROMPT, _buffer, 0.3)
    _buffer.clear()
    if not raw or raw.upper().startswith("NOTHING"):
        return []
    facts = [line.strip("-*\u2022\u00b7 \t") for line in raw.splitlines() if line.strip()]
    facts = [f for f in facts if f and f.upper() != "NOTHING"]
    for fact in facts:
        remember(fact, kind="consolidated")
    return facts


def summarize_session():
    """Write a narrative recap of this whole session. Returns it (or None)."""
    if not _session_log:
        return None
    summary = _digest(_SESSION_SUMMARY_PROMPT, _session_log, 0.4)
    _session_log.clear()
    if not summary or summary.upper().startswith("NOTHING"):
        return None
    date = datetime.now().strftime("%Y-%m-%d")
    remember(f"Session on {date}: {summary}", kind="session_summary")
    return summary


def last_session():
    """Return the text of the most recent session summary, or None."""
    got = _collection.get(
        where={"kind": "session_summary"},
        include=["documents", "metadatas"],
    )
    docs = got.get("documents") or []
    if not docs:
        return None
    metas = got.get("metadatas") or [{}] * len(docs)
    newest = max(zip(docs, metas), key=lambda dm: (dm[1] or {}).get("ts", 0))
    return newest[0]