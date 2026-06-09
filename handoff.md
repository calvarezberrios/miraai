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
- Groq was evaluated as a fast free cloud fallback (OpenAI-compatible, ~30 req/min,
  ~6k tokens/min free tier) but we chose to stay local. The code uses the `openai`
  Python package against Ollama's OpenAI-compatible endpoint, so swapping to Groq or
  another provider later is just a base_url/model change.
- **Model in use:** `llama3.2:3b` (chosen for speed on local hardware).
- **OS:** Windows (PowerShell). Ollama installed to `D:\Ollama\` (secondary drive),
  project at `D:\aiproject\`.
- Ollama server must be running (tray app or `ollama serve`) before the app works.

## Environment
- Project root: `D:\aiproject\`
- Git repo initialized, `.gitignore` present, `.venv` present (should ignore
  `.venv/` and `__pycache__/`).
- Installed packages: `openai`.

---

## Folder structure (current)
```
D:\aiproject\
├── main.py                         # entry point — wires modules together, chat loop
└── brain\
    ├── __init__.py
    └── forebrain\
        ├── __init__.py
        ├── prefrontal_cortex.py    # persona + reasoning (the LLM call)
        ├── thalamus.py             # input routing + working memory (last 20 turns)
        └── amygdala.py             # emotion state that colors responses
```
File organization follows the brain reference: `brain > <division> > <region>.py`.
Naming rules: lowercase folders, snake_case files, no spaces.

---

## Module summary

### prefrontal_cortex.py — "the CEO"
- Holds `PERSONA` system prompt (Mira: playful, a little sarcastic, curious; short
  1-3 sentence spoken-style replies; no emoji/stage directions).
- `think(history, mood_flavor="")` — makes the Ollama chat call, injects current mood
  into the system prompt, returns reply text.

### thalamus.py — input routing + working memory
- `working_memory` = `deque(maxlen=20)` — short-term context.
- `receive(user_message)` — appends user msg, returns full context list.
- `remember_reply(reply)` — appends assistant reply.

### amygdala.py — emotion
- `mood` global: happy | annoyed | excited | neutral.
- `feel(user_message)` — keyword matching against TRIGGERS dict to set mood
  (settles to neutral if nothing matches).
- `color()` — returns a mood-flavor instruction line injected into the persona.
- NOTE: currently keyword-based. Can upgrade to LLM-based mood classification later
  (smarter, catches sarcasm/slang, but doubles LLM calls per message → slower).
  Decided to defer until the full pipeline exists and we can judge if it's worth the latency.

---

## Status: PHASE 1 COMPLETE ✅
Working text brain confirmed running:
- Persona consistency ✅
- Working memory (remembers name within last 20 turns) ✅
- Emotion state shifts tone (e.g. "I subscribed!" → excited, "this is boring" → annoyed) ✅

### Immediate next action
- Commit the checkpoint:
  `git add -A && git commit -m "Phase 1: text brain complete"`
- Confirm `.gitignore` covers `.venv/` and `__pycache__/`.

---

## NEXT: Phase 2 — Memory (hippocampus)
Goal: long-term memory across sessions (facts, viewers, past streams) — beyond the
20-turn working memory.
- Plan: local **vector DB (Chroma)** to stay fully local.
- Requires an embedding model. Ollama can serve one locally — will need to
  `ollama pull nomic-embed-text` (one more local pull, no cloud).
- Will add a `hippocampus.py` module and wire retrieval into the think loop.

---

## Working style for next session
- Step by step, one step at a time, not verbose.
- Stay local / free unless a fully-free fast cloud option is explicitly wanted.
