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
- **STT engine:** **Whisper** via **faster-whisper** (accent-robust). BUILT (see Wernicke's area).
- **OS:** Windows (PowerShell). Ollama installed to `D:\Ollama\`, project at `D:\aiproject\`.
- **GPU: NVIDIA GTX 1660 Super, 6GB VRAM, FULL-PRECISION ONLY.** The 16-series (Turing
  TU116, no tensor cores) has a known broken/half-rate fp16 path — half precision yields
  NaN/silent output. This is HARDWARE, not config; do not chase fp16 fixes. Consequence:
  GPT-SoVITS must run fp32 (`is_half: false`), which is the main cap on voice-gen speed.
  A future GPU with a working fp16 path + 12GB+ VRAM is the only real speedup (unlocks
  half precision AND lets all three models coexist). VRAM is also the day-to-day squeeze:
  llama3.2:3b + GPT-SoVITS (fp32) + Whisper on 6GB is tight — watch it.
- Ollama server must be running (tray app or `ollama serve`) before the app works.
- The GPT-SoVITS API server must ALSO be running before voice works (see Running the app).

## Environment
- Project root: `D:\aiproject\`
- Git repo initialized, `.gitignore` present, `.venv` present.
- `.gitignore` should cover: `.venv/`, `__pycache__/`, `memory_store/`, `test_out.wav`.
- Installed packages: `openai`, `chromadb`, `requests`, `sounddevice`, `soundfile`,
  **`faster-whisper`, `numpy`** (last two added in Phase 3 for STT).
- **CUDA DLL fix (Windows + faster-whisper):** CTranslate2 needs cuBLAS/cuDNN DLLs not on
  PATH → `RuntimeError: cublas64_12.dll not found`. Fixed by `pip install nvidia-cublas-cu12
  "nvidia-cudnn-cu12==9.*"` AND a DLL-registration shim at the top of `wernickes_area.py`
  (adds the pip `nvidia\*\bin` dirs via `os.add_dll_directory` + prepends them to PATH).
  Both are required. Note the machine has stacked conda `(base)` + `.venv`; install with
  `python -m pip` so wheels land in the venv, not base.
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
        │       ├── brocas_area.py           # TTS: GPT-SoVITS API + pipeline + speech-clean
        │       └── voice\
        │           └── reference.wav        # cloned-voice reference clip
        │   └── temporal_lobe\
        │       └── wernickes_area.py        # STT: faster-whisper continuous listening  [NEW]
        └── subcortical_structures\
            ├── thalamus.py                  # input routing + working memory
            └── limbic_system\
                ├── amygdala.py              # emotion state that colors responses
                └── hippocampus.py           # long-term memory + consolidation
```
Every package dir has an `__init__.py`. Organization follows the brain reference:
`brain > <division> > ... > <region>.py`. Lowercase folders, snake_case files.

(Scratch diagnostic scripts from debugging — `voice_test.py`, `devices.py`, `test2.py`,
plus `stt_test.py` (standalone Wernicke's-area mic test) — sit in the root; keep or
delete, not load-bearing.)

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

### brocas_area.py — speech production (TTS)
- `VOICE` dict: ref_audio_path, prompt_text (the clip's transcript), prompt_lang, text_lang.
- `PARAMS`: generation knobs. `parallel_infer: False` + `split_bucket: False` disable
  api_v2's parallel/batched path (source of its intermittent blank-wav bug). `temperature`
  ~1.0 (lower = steadier). **`speed_factor: 1.12`** — speaks slightly faster without
  re-pitching (1.0 = normal; ~1.25 is the natural-sounding ceiling).
- **Two-stage pipeline (latency):** `say()` splits the reply into sentences; a synth worker
  turns each into audio, a separate playback worker plays them in order. Sentence N plays
  while N+1 synthesizes, so she starts talking after the FIRST sentence is ready.
  NOTE: with the persona's mostly-single-sentence replies, this helps less than it would
  for long replies. On a 1660 Super, per-sentence synth time is the floor (fp32-bound).
- **`warmup()`** — fires a throwaway (synthesized-not-played) generation at startup to heat
  CUDA kernels so the first REAL reply isn't slow. Called once in main.py.
- **`_clean_for_speech(text)`** — runs inside `say()` before splitting, on SPOKEN text only
  (printed transcript is untouched). Converts laugh-type stage directions to vocal sounds
  via the `SPOKEN_ACTIONS` dict (`*giggles*`→"hehe", `*laughs*`→"haha", etc., substring
  match); DROPS any other `*...*`/`(...)`/`[...]` direction it can't voice (`*winks*`,
  `*shrugs*`); strips emoji + stray markdown; removes intra-word hyphens so parts join
  (`well-known`→`wellknown`, was being read as "minus"); turns standalone dashes into a
  comma pause. Caveat: also strips dashes in numbers/ranges (`3-4`→`34`) — fine for chat.
- `_synthesize(text)` POSTs to `http://127.0.0.1:9880/tts`; `_play(bytes)` decodes with
  soundfile, plays via sounddevice, blocks till done.
- `say(text)` — enqueue (returns instantly). `wait_until_done()` — blocks until BOTH the
  synth and playback queues drain (motor sequencing; replies never overlap).

### wernickes_area.py — speech comprehension (STT)  [NEW in Phase 3]
- **faster-whisper**, model `small` / `int8` / `cuda` by default (config consts at top:
  `MODEL_SIZE`, `DEVICE`, `COMPUTE_TYPE`, `INPUT_DEVICE`). Bump to `medium`/`large-v3` if
  accent needs it; switch `DEVICE="cpu"` if VRAM gets tight.
- **Continuous listening:** mic captured in a background thread via sounddevice; current
  utterance re-transcribed every `REFRESH_SEC` (0.7s) so words appear live (`on_partial`).
- **VAD endpointing:** faster-whisper's Silero VAD (`get_speech_timestamps`) watches for
  `END_SILENCE_SEC` (3.0s) of trailing silence → utterance finalized → `on_final(text)`.
- **Interrupt:** if you talk past `INTERRUPT_AFTER_SEC` (25s) without stopping, `on_interrupt`
  fires ONCE with the partial so Mira can butt in mid-ramble.
- `pause()` / `resume()` mute/unmute the mic (main.py mutes while Mira speaks so she doesn't
  hear herself). `flush()` discards captured-but-unfinalized audio (used after an interrupt
  so she doesn't answer the same ramble twice).
- `start(on_final, on_partial, on_interrupt)` loads the model + opens the mic; `stop()` closes.
- DLL shim at the very top registers pip CUDA DLLs (see CUDA DLL fix above) — must stay first.

### main.py — voice-driven loop + triggers  [UPDATED in Phase 3]
- **No longer an `input()` chat loop** — driven by the mic. `ears.start()` wires
  `on_final`→`handle_turn`, `on_interrupt`→`handle_turn(..., interrupting=True)`,
  `on_partial`→prints live `[mic]` text. `brocas_area.warmup()` fires once at startup.
- `handle_turn()` (guarded by a `turn_lock` so voice + typed turns never overlap):
  feel → receive → recall → think → remember_reply → observe → **pause mic → say →
  wait_until_done → (flush if interrupting) → resume mic**.
- Interrupt turns inject an `INTERRUPT_NOTE` into the mood flavor so Mira reacts as a
  playful mid-thought interjection.
- A typed line still works (the main thread reads stdin); typing `quit`/`exit` stops the
  mic, runs consolidation + `summarize_session()` + `wait_until_done()`, then exits.
- On startup: pulls `last_session()` and injects it so Mira can reference last time.
- Consolidation fires every **95 messages OR 30 minutes**, whichever first; also on quit.

---

## Running the app (startup order matters now)
1. Start **Ollama** (tray app or `ollama serve`).
2. Start **GPT-SoVITS** API server in its own terminal:
   `cd D:\GPT-SoVITS; .\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880` — wait for the
   "TTS config" / models-loaded message.
3. From `D:\aiproject\` (venv active): `python main.py`. First run downloads the Whisper
   model (one-time). Then just **talk** — pause ~3s and she replies. Typing still works.
   `quit`/`exit` to close. **Use headphones** — on speakers her audio can bleed into the
   mic at the resume() edge (echo cancellation is a later refinement).

---

## Status: PHASE 3 — VOICE (COMPLETE ✅)
- **Broca's area (TTS)** — GPT-SoVITS, cloned voice, plays automatically ✅
- **Motor sequencing** — two-stage synth→playback pipeline in `brocas_area.py` ✅
- **Speech cleaning** — laugh actions→sounds, unspeakable directions dropped, hyphen fix,
  `speed_factor` 1.12 ✅
- **Wernicke's area (STT)** — faster-whisper continuous listening, live partials, 3s-silence
  endpointing, 25s interrupt, mic mute while speaking ✅
- **Voice-driven main loop** — mic replaces `input()`, typed input still works ✅

### Known limitations / deferred (not blockers)
- **Voice-gen speed is fp32-bound on the 1660 Super.** Already did the free wins (warmup,
  sentence pipeline, VRAM mgmt). A couple seconds faster needs a GPU with a working fp16
  path + 12GB+ VRAM. Nothing more to squeeze in software for now.
- **Speakers→mic bleed:** mic is muted while Mira talks, but on speakers the tail can leak
  at resume(). Use headphones; real echo cancellation is Phase 4+ territory.
- **Barge-in:** words spoken WHILE she's interrupting are lost (mic muted then flushed).
  True overlap handling is Phase 4 attention work.

### Phase 2 refinements still deferred (small follow-ups, not blockers)
- **No dedup:** consolidation can re-store facts Mira already knows. Add by checking
  `recall` similarity before writing.
- **No relevance cutoff on recall:** add a distance threshold so only genuinely-related
  memories get injected into the prompt.
- **Session summary is exit-only:** a crash skips it (atomic facts still survive). A
  crash-safe version would summarize periodically.
- **30-min consolidation timer** is now checked inside `handle_turn` rather than at a
  blocking `input()`, but still only fires on activity. True background timing → a thread,
  naturally hypothalamus/scheduler territory in Phase 7.

---

## NEXT: Phase 4 — Attention & Behavior
Per the build plan:
- **Message prioritization** — filter spam, rank by relevance/subs/keywords →
  *reticular activating system*. Anatomically brainstem-wide; expect a new module outside
  the forebrain tree (e.g. `brain\brainstem\reticular_formation\` or similar).
- **Action selector** — respond / ignore / react / start topic → *basal ganglia*
  (`brain\forebrain\subcortical_structures\basal_ganglia\`).
- **Idle behaviors** when chat is quiet (ramble, hum, comment) → *default mode network*.
- Threading groundwork from the mic (background capture, the interrupt path, turn_lock)
  is the foothold for real-time attention here.

## Working style for next session
- Step by step, one step at a time, not verbose.
- Stay local / free unless a fully-free fast cloud option is explicitly wanted.