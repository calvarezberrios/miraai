# Mira AI VTuber — Project Handoff

## Purpose of this doc
Primer for continuing the build in a new chat session. Paste this in (along with the
human brain reference + build plan docs) to get back up to speed.

---

## Project overview
Building an AI VTuber named **Mira**, architected around the human brain. Each brain
region maps to a software module. Working through a phased build plan.

## Key decisions / constraints
- **Fully local, no cloud LLMs.** Running models via **Ollama**.
- The code uses the `openai` Python package against Ollama's OpenAI-compatible
  endpoint, so swapping to Groq or another provider later is just a base_url/model
  change. Same client is reused for chat and embeddings.
- **Chat model:** `llama3.2:3b` (chosen for speed on local hardware).
- **Embedding model:** `nomic-embed-text` (local, for long-term memory).
- **OS:** Windows (PowerShell). Ollama installed to `D:\Ollama\`, project at `D:\aiproject\`.
- Ollama server must be running (tray app or `ollama serve`) before the app works.

## Environment
- Project root: `D:\aiproject\`
- Git repo initialized, `.gitignore` present, `.venv` present.
- `.gitignore` should cover: `.venv/`, `__pycache__/`, `memory_store/`.
- Installed packages: `openai`, `chromadb`.

---

## Folder structure (current)
```
D:\aiproject\
├── main.py                 # entry point — chat loop, wires modules, consolidation triggers
├── memory_store\           # Chroma vector store (auto-created, gitignored)
└── brain\
    └── forebrain\
        ├── cerebrum\
        │   └── frontal_lobe\
        │       └── prefrontal_cortex.py     # persona + reasoning (the LLM call)
        └── subcortical_structures\
            ├── thalamus.py                  # input routing + working memory
            └── limbic_system\
                ├── amygdala.py              # emotion state that colors responses
                └── hippocampus.py           # long-term memory + consolidation
```
Every package dir has an `__init__.py`. Organization follows the brain reference:
`brain > <division> > ... > <region>.py`. Lowercase folders, snake_case files.

---

## Module summary

### prefrontal_cortex.py — "the CEO"
- Holds `PERSONA` system prompt (Mira: flirty, playful, sarcastic kitsune VTuber;
  short 1-3 sentence spoken replies; no emoji/stage directions).
- `think(history, mood_flavor="", memories=None)` — Ollama chat call. Injects current
  mood AND any recalled long-term memories into the system prompt, returns reply text.

### thalamus.py — input routing + working memory
- `working_memory` = `deque(maxlen=20)` — short-term context.
- `receive(user_message)` / `remember_reply(reply)`.

### amygdala.py — emotion
- `mood` global; `feel(user_message)` keyword-matches to set mood; `color()` returns a
  mood-flavor instruction injected into the persona. (Keyword-based; LLM-based mood
  classification deferred.)

### hippocampus.py — long-term memory + consolidation
- Local **Chroma** store at `./memory_store`; embeddings via `nomic-embed-text`.
- `remember(text, kind)` / `recall(query, n=3)` — store + semantic retrieval.
- `observe(user, reply)` — feeds two logs: a consolidation buffer + a full session log.
- `consolidate()` — ongoing "sleep" step: hands the recent buffer to the LLM, which
  **autonomously decides** what atomic facts are worth keeping (returns NOTHING if not).
- `summarize_session()` — at session end, writes a narrative recap (kind=session_summary).
- `last_session()` — returns the most recent recap (by stored `ts`), for next-session continuity.
- `forget_all()` — wipes all memories (rebuilds the collection).

### main.py — chat loop + triggers
- On startup: pulls `last_session()` and injects it so Mira can reference last time.
- Each turn: feel → receive → recall → think → remember_reply → observe.
- Consolidation fires every **95 messages OR 30 minutes**, whichever first; also on quit.
- On quit: runs consolidation + `summarize_session()`.

---

## Status: PHASE 2 COMPLETE ✅
- Long-term memory across sessions (Chroma + local embeddings) ✅
- Recall wired into the think loop (semantic, meaning-based) ✅
- Autonomous consolidation — Mira decides what's worth saving ✅
- Session summaries + next-session recall of "what happened last time" ✅

### Known refinements deferred (small follow-ups, not blockers)
- **No dedup:** consolidation can re-store facts Mira already knows. Add by checking
  `recall` similarity before writing.
- **No relevance cutoff on recall:** as memory grows, add a distance threshold so only
  genuinely-related memories get injected into the prompt.
- **Session summary is exit-only:** a crash skips it (atomic facts still survive, since
  `consolidate()` writes incrementally). A crash-safe version would summarize periodically.
- **30-min timer is checked on next input** (console `input()` blocks). True background
  timing → a thread, naturally hypothalamus/scheduler territory in Phase 7.

---

## NEXT: Phase 3 — Voice
- Speech output: TTS (Piper local, or ElevenLabs paid) → *Broca's area*.
- Speech input (optional): Whisper STT → *Wernicke's area*.
- Audio queue so playback doesn't overlap → *motor sequencing*.
- Anatomically: Broca's/Wernicke's live in the cerebrum (frontal/temporal lobes), so
  expect `brain\forebrain\cerebrum\frontal_lobe\brocas_area.py` and a temporal_lobe dir.

---

## Working style for next session
- Step by step, one step at a time, not verbose.
- Stay local / free unless a fully-free fast cloud option is explicitly wanted.
