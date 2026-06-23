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

import os
import time
import uuid
from datetime import datetime

import chromadb
from openai import OpenAI

from brain.forebrain.cerebrum.frontal_lobe import prefrontal_cortex as _pfc

# Embeddings ALWAYS run on an Ollama with nomic-embed-text: the Chroma store below is
# built with this model, so the embedding provider can't change without rebuilding the
# store. (This is why an Ollama is still needed for long-term memory even when the chat
# brain is elsewhere — Gemini, Groq, or a remote llama.cpp.)
#
# MIRA_EMBED_BASE_URL lets embeddings point somewhere OTHER than the chat endpoint. This
# matters in a split setup: when the chat brain is a remote llama.cpp (OLLAMA_BASE_URL ->
# another PC's :8080, which has no nomic-embed-text), set MIRA_EMBED_BASE_URL to an Ollama
# that does (a local one, or that PC's :11434). Defaults to OLLAMA_BASE_URL, so single-box
# setups are unchanged.
_embed_client = OpenAI(
    base_url=os.environ.get(
        "MIRA_EMBED_BASE_URL",
        os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")),
    api_key="ollama")
EMBED_MODEL = "nomic-embed-text"

# Consolidation / session-summary reasoning reuses the main chat client + model, so it
# follows MIRA_LLM_PROVIDER (Ollama or Gemini) — one provider for all of Mira's thinking.

# On-disk store -- survives restarts. Folder is created on first run.
_db = chromadb.PersistentClient(path="./memory_store")
_collection = _db.get_or_create_collection(name="mira_memory")


def _embed(text):
    """Turn text into a vector using the local embedding model."""
    resp = _embed_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def remember(text, kind="fact"):
    """Store a long-term memory. `kind` tags it (fact, session_summary, ...). If the
    local embedder is unreachable (e.g. Ollama not running on a Gemini-only run), the
    memory is skipped rather than crashing the caller."""
    try:
        embedding = _embed(text)
    except Exception as e:
        print(f"[hippocampus] embed unavailable — memory not stored: {e}")
        return
    _collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{"kind": kind, "ts": time.time()}],
    )


# How close a memory must be (L2 distance from the query embedding) to count as relevant
# enough to surface. Calibrated on nomic-embed-text: genuinely on-topic queries land
# ~0.5-0.8; generic chit-chat ("how are you", "lol") and things she has no memory of land
# ~1.0-1.2. So ~0.85 injects real matches and otherwise stays quiet. Tune with
# MIRA_RECALL_MAX_DISTANCE (raise = recall more eagerly, lower = stricter).
_RECALL_MAX_DISTANCE = float(os.environ.get("MIRA_RECALL_MAX_DISTANCE", "0.85"))


def recall(query, n=3, max_distance=None):
    """Quick semantic search for memories actually RELEVANT to `query`. Returns only the
    memories close enough to be about what's being discussed (distance <= the threshold),
    so a name/fact/detail in the conversation pulls what she knows about it, while generic
    chatter pulls nothing — no random facts bleeding in, and a smaller prompt to prefill.
    Empty list if nothing clears the bar or the embedder is unreachable."""
    count = _collection.count()
    if count == 0:
        return []
    try:
        query_embedding = _embed(query)
    except Exception as e:
        print(f"[hippocampus] embed unavailable — no recall this turn: {e}")
        return []
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n, count),
        include=["documents", "distances"],
    )
    docs = results["documents"][0]
    dists = results["distances"][0]
    bar = _RECALL_MAX_DISTANCE if max_distance is None else max_distance
    return [d for d, dist in zip(docs, dists) if dist <= bar]


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
    "remembering long-term ABOUT OTHER PEOPLE AND EVENTS: the people she talks to "
    "(their real names, preferences, what they're up to), recurring viewers, "
    "promises made, and notable events. "
    "Do NOT record anything about Mira herself - her appearance, personality, "
    "backstory, feelings, or that she is an AI/kitsune/the creator's creation. She "
    "already knows who she is; storing self-descriptions makes her talk about herself "
    "in the third person, so never write a fact whose subject is Mira. "
    "Ignore small talk, jokes, flirting, role/mention tokens like \"@everyone\" or "
    "\"@Mira\", and anything trivial or fleeting. "
    "IMPORTANT — identities: turns are prefixed with the speaker's display name (\"Name: ...\"). "
    "If a speaker says who they are, or that their display name belongs to a known person (e.g. a "
    "user named CoolName123 says \"I'm GameRaiderX\"), record that mapping as a durable fact so "
    "she recognizes that name next time: \"The Discord user CoolName123 is GameRaiderX, the "
    "creator.\" Capture the same for anyone identifying themselves. "
    "Write each fact as one short, standalone sentence in the third person (e.g. "
    "\"Sam is learning to play guitar.\"). Use real names, never \"@everyone\". "
    "One fact per line, no numbering or bullets. If nothing is worth remembering, "
    "reply with exactly: NOTHING"
)

_SESSION_SUMMARY_PROMPT = (
    "You are the memory-formation process for a VTuber named Mira. "
    "Summarize this whole streaming session in 2-4 short sentences for Mira's "
    "own future reference: the main topics, who showed up, the general mood, "
    "and anything notable or promised. Write in the third person, past tense "
    "(e.g. \"Talked with Sam about his new cat; chat was upbeat.\"). "
    "If the session was empty or trivial, reply with exactly: NOTHING"
)


def observe(user_message, reply, speaker=None):
    """Record one exchange into both the consolidation buffer and session log.

    `speaker` (the Discord display name) is prefixed onto the user turn so the consolidation /
    session-summary LLM can see WHO said what — that's what lets it learn identity mappings
    ("CoolName123 is GameRaiderX") and attribute facts to the right person."""
    user_content = f"{speaker}: {user_message}" if speaker else user_message
    _buffer.append({"role": "user", "content": user_content})
    _buffer.append({"role": "assistant", "content": reply})
    _session_log.append({"role": "user", "content": user_content})
    _session_log.append({"role": "assistant", "content": reply})


def _digest(prompt, log, temperature):
    """Run the LLM over a transcript with the given system prompt. Uses the main chat
    client/model, so consolidation follows the active provider (Ollama or Gemini)."""
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in log)
    resp = _pfc.client.chat.completions.create(
        model=_pfc.MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
        max_tokens=200,
        temperature=temperature,
        # Disable reasoning when MIRA_NO_THINK=1 (no-op otherwise). Without this, a reasoning
        # model (e.g. the Qwen3 "turbo" build) spends the whole 200-token budget on a <think>
        # block and returns EMPTY content — so consolidation/session-summary would silently
        # store nothing. Mirrors think()/think_stream().
        extra_body=_pfc._EXTRA,
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