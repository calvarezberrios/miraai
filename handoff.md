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
- **TTS engine:** **GPT-SoVITS** (local voice cloning) — NOT Piper. Mira's voice is a
  young-adult-female anime timbre cloned zero-shot from a short reference clip.
- **STT engine (planned):** **Whisper** via faster-whisper (accent-robust). Not built yet.
- **OS:** Windows (PowerShell). Ollama installed to `D:\Ollama\`, project at `D:\aiproject\`.
- **GPU: NVIDIA, under 8GB VRAM, FULL-PRECISION ONLY.** This machine produces silent/NaN
  output under fp16 — see the GPT-SoVITS gotcha below. Watch total VRAM once Ollama +
  GPT-SoVITS + Whisper are all loaded.
- Ollama server must be running (tray app or `ollama serve`) before the app works.
- The GPT-SoVITS API server must ALSO be running before voice works (see Running the app).

## Environment
- Project root: `D:\aiproject\`
- Git repo initialized, `.gitignore` present, `.venv` present.
- `.gitignore` should cover: `.venv/`, `__pycache__/`, `memory_store/`, `test_out.wav`.
- Installed packages: `openai`, `chromadb`, **`requests`, `sounddevice`, `soundfile`**
  (the last three added in Phase 3 for TTS playback).
- **GPT-SoVITS** installed SEPARATELY at `D:\GPT-SoVITS\` (its own bundled Python runtime;
  do NOT mix with the aiproject venv). Not part of the git repo (too large).

---

## GPT-SoVITS setup (read before touching voice)
- Windows integrated package at `D:\GPT-SoVITS\`. Started in API mode, not the WebUI:
  ```powershell
  cd D:\GPT-SoVITS
  .\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880
  ```
  Listens on port 9880; `/tts` endpoint takes text + ref_audio_path + prompt_text/lang
  and returns a WAV.
- **CRITICAL GOTCHA — half precision:** `api_v2` reads its precision from
  `GPT_SoVITS\configs\tts_infer.yaml`. With `is_half: true` on this GPU, synthesis
  returns a correct-LENGTH but ZERO-AMPLITUDE wav (silent). FIX: set `is_half: false`
  in the `custom:` section (and any other `is_half: true` lines), keep `device: cuda`,
  restart the server. Symptom to recognize next time: audio file is full-size but
  `peak amplitude == 0.0`.
- **Models (v1 pretrained):** `s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt` (GPT) and
  `s2G488k.pth` (SoVITS). The WebUI selected these; after a server restart `api_v2` loads
  whatever `tts_infer.yaml` points to. Current voice sounds correct, so leave it; if the
  timbre ever drifts, re-pin via `/set_gpt_weights` and `/set_sovits_weights`.
- **Reference voice:** clip + transcript live at
  `brain\forebrain\cerebrum\frontal_lobe\voice\reference.wav`; transcript is in the
  `VOICE` dict in `brocas_area.py`. Reference must be 3–10s, clean, single speaker.

---

## Folder structure (current)
```
D:\aiproject\
├── main.py                 # entry point — chat loop, wires modules, consolidation + speech
├── memory_store\           # Chroma vector store (auto-created, gitignored)
└── brain\
    └── forebrain\
        ├── cerebrum\
        │   └── frontal_lobe\
        │       ├── prefrontal_cortex.py     # persona + reasoning (the LLM call)
        │       ├── brocas_area.py           # TTS: GPT-SoVITS API + playback queue  [NEW]
        │       └── voice\
        │           └── reference.wav        # cloned-voice reference clip           [NEW]
        └── subcortical_structures\
            ├── thalamus.py                  # input routing + working memory
            └── limbic_system\
                ├── amygdala.py              # emotion state that colors responses
                └── hippocampus.py           # long-term memory + consolidation
```
Every package dir has an `__init__.py`. Organization follows the brain reference:
`brain > <division> > ... > <region>.py`. Lowercase folders, snake_case files.

(Scratch diagnostic scripts from debugging — `voice_test.py`, `devices.py`, `test2.py` —
sit in the root; keep or delete, not load-bearing.)

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

### brocas_area.py — speech production (TTS)  [NEW in Phase 3]
- `VOICE` dict: ref_audio_path, prompt_text (the clip's transcript), prompt_lang, text_lang.
- `PARAMS`: generation knobs. Note `parallel_infer: False` + `split_bucket: False` —
  disables api_v2's parallel/batched path, which is the source of its intermittent
  blank-wav bug. `temperature` ~1.0 (lower = steadier).
- `OUTPUT_DEVICE` (None = system default; set to an index/name to force a device).
- `_synthesize(text)` — POSTs to `http://127.0.0.1:9880/tts`, returns WAV bytes.
- `_play(bytes)` — decodes with soundfile, plays via sounddevice, blocks till done.
- Background worker thread + `queue.Queue` = **motor sequencing** (replies never overlap).
- `say(text)` — enqueue (returns instantly). `wait_until_done()` — block till queue drains.

### main.py — chat loop + triggers
- On startup: pulls `last_session()` and injects it so Mira can reference last time.
- Each turn: feel → receive → recall → think → remember_reply → observe → **brocas_area.say(reply)**.
- Consolidation fires every **95 messages OR 30 minutes**, whichever first; also on quit.
- On quit: runs consolidation + `summarize_session()` + **brocas_area.wait_until_done()**
  (so her last line finishes before exit).

---

## Running the app (startup order matters now)
1. Start **Ollama** (tray app or `ollama serve`).
2. Start **GPT-SoVITS** API server in its own terminal:
   `cd D:\GPT-SoVITS; .\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880` — wait for the
   "TTS config" / models-loaded message.
3. From `D:\aiproject\` (venv active): `python main.py`.

---

## Status: PHASE 3 — VOICE (partially complete)
- **Broca's area (TTS)** — GPT-SoVITS, cloned voice, plays automatically in the loop ✅
- **Motor sequencing** — playback queue inside `brocas_area.py` ✅
- **Wernicke's area (STT)** — NOT STARTED (next).

### Phase 2 refinements still deferred (small follow-ups, not blockers)
- **No dedup:** consolidation can re-store facts Mira already knows. Add by checking
  `recall` similarity before writing.
- **No relevance cutoff on recall:** add a distance threshold so only genuinely-related
  memories get injected into the prompt.
- **Session summary is exit-only:** a crash skips it (atomic facts still survive). A
  crash-safe version would summarize periodically.
- **30-min timer is checked on next input** (console `input()` blocks). True background
  timing → a thread, naturally hypothalamus/scheduler territory in Phase 7.

---

## NEXT: finish Phase 3 — Wernicke's area (STT)
- Speech input via **Whisper**, run through **faster-whisper** (same accuracy, ~4x faster
  on GPU). For accent robustness use **large-v3 with INT8 quantization**
  (`compute_type="int8"` or `"int8_float16"`) so it fits under 8GB alongside Ollama + GPT-SoVITS.
- Anatomically temporal lobe → expect
  `brain\forebrain\cerebrum\temporal_lobe\wernickes_area.py` (+ `__init__.py`).
- No server needed (unlike GPT-SoVITS); it's a Python package in the aiproject venv.
- **Watch VRAM:** llama3.2:3b + GPT-SoVITS (fp32) + Whisper all on one <8GB card is tight.
  If it won't fit, drop Whisper to `small`/`medium` INT8, or run STT on CPU (it's intermittent).
- Decide the trigger: push-to-talk vs continuous listening. Console build currently blocks
  on `input()`, so mic capture will need its own thread (preview of Phase 4 attention work).

## Working style for next session
- Step by step, one step at a time, not verbose.
- Stay local / free unless a fully-free fast cloud option is explicitly wanted.