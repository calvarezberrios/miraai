# Mira AI VTuber — Project Handoff

## Purpose of this doc
Primer for continuing the build in a new chat session. Paste this in (along with the
human brain reference + build plan docs) to get back up to speed.

---

## Recent changes — 2026-06-23 (READ FIRST; supersedes stale hardware/voice bits below)

**Hardware moved.** Mira now runs on a **laptop with an RTX 5050 (8 GB, Blackwell, sm_120)**, and
the **LLM runs on a separate desktop PC** over the LAN (llama.cpp model id `turbo` at
`192.168.12.151:8080`, embeddings `:11434`) — see `LAPTOP_SETUP.md`. So the old handoff notes about
a **GTX 1660 Super / 6 GB / "broken fp16, full-precision only" / `D:\aiproject`** are **STALE**: the
laptop GPU has a working fp16 path and only holds **STT + TTS** (the LLM is off-box). Launch Discord
mode with **`start_discord.bat`** (sets the LAN endpoints + `--discord --draft`).

- **🔴 Discord voice ~5s "stutter/cutoff" — ROOT-CAUSED & FIXED.** This was a real bug in the pinned
  experimental py-cord voice build: received audio packets are processed as tasks on the asyncio
  event loop, but the loop was only woken by its ~5s voice heartbeat, so audio arrived in 5-second
  bursts and the endpointer chopped sentences. Fixed with a tiny `_loop_pump` thread in
  `discord_adapter.py` that keeps the loop awake during voice (`call_soon_threadsafe` every 10 ms).
  **Full write-up + upstream-PR assessment: `DISCORD_VOICE_FIX.md`.** This makes Discord voice
  finally smooth — you can talk freely without being cut off.
- **Discord endpointing rewritten.** Ends a turn on a *transmission* gap (`now - last_voice`), NOT
  on silence inside the (bursty) buffer; **never finalizes while Mira is speaking** (`_brain_busy`)
  so her reply can't chunk your continued speech; re-transcribes the whole utterance fresh at
  finalize. `VOICE_END_SILENCE` default `0.7s` (was 2.0). Diagnostics added (all default-off):
  `MIRA_VOICE_DEBUG`, `MIRA_VOICE_DUMP` (dump utterance WAV), `MIRA_VOICE_LOG` (DAVE drop log),
  `MIRA_VOICE_PARTIAL_WINDOW_SEC` (partials transcribe only the last N sec).
- **STT:** `wernickes_area.py` defaults now `MODEL_SIZE=distil-large-v3`, `COMPUTE_TYPE=float16`
  (great English accuracy, ~340 ms/utterance, ~2 GB). NOTE `start_discord.bat` sets
  `WHISPER_MODEL_SIZE` explicitly, so change the model THERE for real runs.
- **TTS (Kokoro) now runs on the GPU.** `pip install kokoro` pulls **CPU-only** torch; fixed by
  installing `torch==2.12.1+cu130` into `.venv-kokoro` (Blackwell needs cu128+). `kokoro_infer.py`
  has a `_isolate_cuda_dlls()` shim so it loads its own CUDA-13 cuDNN, not the main venv's CUDA-12
  one (that clash caused `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`). Logs `device=cuda:0`.
- **Speaker identity (NEW).** Mira now knows who's talking by **Discord display name** and treats an
  unfamiliar name as a stranger until told who they are; saying "I'm GameRaiderX" is consolidated to
  memory so that name is recognized next session. Touches: `intents.members=True`
  (`discord_adapter`), speaker-prefixed turns (`thalamus.receive`, `hippocampus.observe`), a per-turn
  identity block + conditional "Senpai" in `prefrontal_cortex._build_system`, identity-capture in the
  consolidation prompt, and speaker wiring in `main.py` (`_recall_for`). Needs **Server Members
  Intent** enabled in the Developer Portal or names show as `<Object id=…>`.
- **Consolidation bug fixed.** `hippocampus._digest()` didn't pass the no-think flag, so on the
  `turbo` reasoning model it returned EMPTY → long-term memory silently stored nothing. Now passes
  `extra_body=_pfc._EXTRA`.
- **Drafting (job 1 of the subconscious) can run standalone.** New `--draft` flag (enabled in
  `start_discord.bat`) and `subconscious.start(draft_only=True)`: she pre-drafts a reply *while you
  talk* (identity-aware) and answers the instant you stop, WITHOUT the full subconscious's chime-ins
  or daydreams. `--subconscious` still enables the whole background mind.
- **Text-chat conversation continuity (NEW).** Discord text follow-ups WITHOUT her name now continue
  an active conversation: `main.py` acts on `should_respond`'s `reason="consider"` tier via
  `judge_relevance` (non-voice channels). Also fixed `judge_relevance()` to pass the no-think flag
  (was empty→always-NO on turbo, like the `_digest` bug).
- **PDF documents / RAG (NEW).** Attach a PDF in Discord (e.g. a game rulebook) → `discord_adapter`
  (`_ingest_pdf`, `pypdf`) extracts text → `hippocampus.remember_document(name, text)` chunks +
  embeds it into a SEPARATE `mira_documents` Chroma collection (persists). On each turn
  `main.py._recall_for` (and the drafter) also `recall_document(text)`; relevant excerpts go into a
  dedicated "reference material — quote precisely" block in `_build_system` (new `documents=` param
  on `think`/`think_stream`/`consider_speaking`). Commands: `mira what games do you know`,
  `mira forget <name>`. Local model is ~8k ctx (Qwen3.6-35B-A3B, text-only) so it RAGs, can't hold a
  whole book; needs the desktop embedder. Text PDFs only (no OCR).
- **Console de-dup.** A voice utterance printed 3× (`[mic]` live, `[voice]`, then again before her
  reply). Removed the `[voice]` line and now `_clear_mic_line()` wipes the rolling `[mic]` caption on
  finalize so it settles into one committed "Speaker: text" line.

---

## Note-taking mode (DLPFC scribe) — NEW (READ FIRST)

Mira can now **take notes** of what she hears — local mic or Discord voice — but **only
when asked**. While a session is active she **does not speak or chime in**: she just
listens and records. New brain region (per `human_brain_reference.md`, this is working
memory + sustained attention + top-down control = the **dorsolateral prefrontal cortex**):
`brain/forebrain/cerebrum/frontal_lobe/dorsolateral_prefrontal_cortex.py` (aliased
`scribe` in `main.py`). Sits beside `prefrontal_cortex.py`; does NOT overload it.

- **How it works:** `main.py` `handle_message()` first calls `scribe.intercept(event,
  notify=adapter.notify)`. If it consumes the event (a command, or an utterance to
  record while active) main returns immediately — no reply, no subconscious. `on_event`
  also skips `subconscious.observe_partial` while active.
- **Subconscious is paused** during a session via new `posterior_cingulate_cortex.pause()
  /resume()` (a `_paused` Event gating `_tick`, the drafter, `observe_partial`, `heard`),
  so the default-mode network can't daydream aloud and break the silence.
- **Commands** (matched on the heard/typed text; leading "mira" optional):
  - start: `take notes` / `start taking notes` / `take notes about <topic>`;
    **TTRPG**: `take dnd/ttrpg/rpg/game/campaign notes` → ttrpg profile.
  - `recap` / `summarize` / `what do you have so far` → delivers an organized recap.
  - `stop taking notes` / `end notes` / `done taking notes` → finalize + save.
  - TTRPG cast (only when addressed): `Alice plays Lyra` / `I'm playing Lyra`.
- **Comms are TEXT/CONSOLE ONLY** (user decision): confirmations + recaps go through
  `adapter.notify()` — new method on `IOAdapter` (default `print`); `DiscordAdapter`
  overrides it to also post to the last text channel (`self._last_text_channel`). Voice
  stays silent.
- **Files:** one `.txt` per session in `notes/` (gitignored). Raw transcript written
  **live** (crash-safe, attributed `[HH:MM:SS] Speaker: text`). At stop/quit the LLM
  (reusing `prefrontal_cortex.client`/`MODEL`, kept hot) appends a topic-organized body
  (TTRPG: organized **by player & character** with Cast/Story/NPCs/Quests/Loot sections)
  + a SUMMARY + footer, then the file is **renamed** to `<derived-topic>_<timestamp>.txt`.
  `recap` is delivery-only (not written). LLM runs only at recap/finalize, never
  per-utterance (6GB VRAM / one-model-at-a-time).
- **Quit mid-session auto-finalizes** (`scribe.finalize_if_active()` in the quit branch)
  so notes aren't lost.
- Verified with an isolated functional test (stubbed LLM + subconscious): start/record/
  recap/stop, TTRPG cast + grouping, inactive passthrough, empty-session, already-active.
  No new dependencies.

---

## LLM provider — Gemini option (NEW)

Mira's chat brain can now run on **Google Gemini** instead of local Ollama, to compare
response quality. Both speak the OpenAI API, so the switch is just base_url/key/model on
the shared client (`prefrontal_cortex.client` / `MODEL`). Set in `.env`:
`MIRA_LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...`, optional `MIRA_GEMINI_MODEL`
(default `gemini-2.0-flash`). Default provider is `ollama` (unchanged).

- Routes through Gemini: `think` / `consider_speaking` / `wander`, the **subconscious**
  (reuses the client), the **note-taking scribe** (reuses it), and **consolidation/summary**
  (`hippocampus._digest` now reuses `prefrontal_cortex.client`/`MODEL` — this also retired
  the stale `CHAT_MODEL = "llama3.2:3b"`).
- **Embeddings stay on Ollama** (`nomic-embed-text`, dedicated `hippocampus._embed_client`):
  the Chroma store is built with them and can't switch providers without a rebuild. So
  **Ollama is still required for long-term memory** even on Gemini. `recall()`/`remember()`
  now **degrade gracefully** (skip, no crash) if the embedder is unreachable, so a
  Gemini-only run (Ollama off) still chats — just without memory that run.
- Startup logs the active provider: `[prefrontal_cortex] LLM provider: gemini (model: ...)`.
- Implementation: provider branch at the top of `prefrontal_cortex.py`. Verified the switch
  selects the right client/base_url/model under both providers (no live Gemini key tested —
  user supplies `GEMINI_API_KEY`).

---

## Recent changes — 2026-06-18 (READ FIRST; supersedes stale bits below)

- **Fine-tuned chat model.** Mira's conversational model was fine-tuned on the
  `Cynaptics/persona-chat` dataset (QLoRA on **Qwen2.5-3B-Instruct**, trained free on
  Kaggle/Colab — the 6 GB GTX 1660 can't train it). Result is registered in Ollama as
  **`mira`**. Pipeline + notebooks live in **`finetune/`** (`mira_finetune_colab.ipynb`,
  `mira_finetune_kaggle.ipynb`, `prepare_data.py`, `README.md`). **GGUF gotcha:** Ollama's
  *safetensors* importer outputs gibberish (`@@@@`) for this model — always import the
  `.gguf` (Unsloth writes it to a `*_gguf/` sibling folder), or convert the safetensors
  with llama.cpp's `convert_hf_to_gguf.py`. Never let Ollama convert safetensors directly.
- **Config now comes from `.env`.** New **`env_loader.py`** is called at the top of
  `main.py` (before the brain imports, since `prefrontal_cortex` reads `MIRA_MODEL` at
  import time) and loads `.env` into `os.environ`. Any `MIRA_*`/config var can live in
  `.env`; a real system env var still overrides it. `.env` now holds `MIRA_MODEL=mira`
  and `DISCORD_BOT_TOKEN=...` (renamed from `DISCORD_TOKEN` so the Discord adapter reads it).
- **Prompt simplified — GROUNDING and STYLE blocks REMOVED.** `prefrontal_cortex.py` no
  longer stacks persona + grounding + style. It's now ONE rewritten `PERSONA` string in a
  deadpan, chaotic **"Neuro-sama"-style** voice, with the key constraints folded in as
  character (don't fabricate real events/history; no tacked-on engagement questions; no
  asterisks / `Mira:` name labels). `_build_system()` = `PERSONA` + live context
  (situation/mood/daydreams/memories) only. ⚠ This removed the old hard anti-confabulation
  guardrail — watch for invention and tighten the persona's "don't make things up" line if
  it creeps back. **Any text below describing separate GROUNDING/STYLE blocks is STALE.**
- **Subconscious wander fix.** `wander()` was emitting greetings/dialogue ("Hey, how are
  you?") because it reused the reply-oriented prompt with no user turn; it now uses a
  dedicated inner-monologue prompt (persona only + anti-greeting framing + examples).
  Private wandering thoughts no longer print to console unless `MIRA_SUB_DEBUG=1`
  (`LOG_THOUGHTS = DEBUG` in `posterior_cingulate_cortex.py`).
- **Python 3.14 / protobuf.** The venv runs Python 3.14; protobuf must be **≥5** (4.x
  crashes chromadb/opentelemetry import with "Metaclasses with custom tp_new are not
  supported"). Currently 6.x. Don't let a stray `pip install` pull protobuf below 5.
- **TTS (verify):** runtime now logs `TTS engine: kokoro` — the `MIRA_TTS` default in
  `brocas_area.py` is **`kokoro`** (separate `.venv-kokoro`), NOT the Piper/RVC described in
  the TTS section below. That section is likely stale; confirm before relying on it.

---

## Project overview
Building an AI VTuber named **Mira**, architected around the human brain. Each brain
region maps to a software module. Working through a phased build plan.

## Key decisions / constraints
- **Fully local, no cloud LLMs.** Running models via **Ollama**.
- The code uses the `openai` Python package against Ollama's OpenAI-compatible
  endpoint, so swapping providers later is just a base_url/model change. Same client is
  reused for chat and embeddings.
- **Chat model:** `qwen2.5:3b` (changed from `llama3.2:3b`). On the 6GB GPU running
  STT + LLM + TTS together, a 3B is the sweet spot; qwen2.5:3b follows instructions /
  grounding rules and invents far less than llama3.2:3b while still fitting alongside
  Whisper and GPT-SoVITS. Set in ONE place: the `MODEL` constant in `prefrontal_cortex.py`.
  Bigger (`qwen2.5:7b` / `llama3.1:8b`) grounds even better but crowds the GPU.
- **Embedding model:** `nomic-embed-text` (local, for long-term memory).
- **TTS engine:** **Piper** (replaced GPT-SoVITS), with optional **RVC** voice conversion.
  Piper generates fast English-female speech. RVC (timbre conversion to a trained `.pth`/`.index`
  voice) is wired up and **currently ENABLED** (`USE_RVC = True` in `brocas_area.py`); current
  voice = **Yui (K-On!)**. Set `USE_RVC = False` to fall back to the raw Piper female voice
  (no subprocess launched) if a model's output isn't clean enough.
- **STT engine:** **Whisper** via **faster-whisper** (accent-robust).
- **OS:** Windows (PowerShell). Ollama at `D:\Ollama\`, project at `D:\aiproject\`.
- **GPU: NVIDIA GTX 1660 Super, 6GB VRAM, FULL-PRECISION ONLY.** Turing TU116 has a broken
  fp16 path (half precision → NaN/silent). HARDWARE, not config; do not chase fp16 fixes.
  GPT-SoVITS must run fp32 (`is_half: false`). VRAM is the day-to-day squeeze. A future GPU
  with working fp16 + 12GB+ is the only real speedup.
- Ollama server must be running before the app works.
- No separate TTS server needed — Piper runs inline, RVC via subprocess.

## NEW: Mira now runs in two modes
- `python main.py`            → **local** mode (mic + speakers; unchanged, rock-solid).
- `python main.py --discord`  → **Discord** mode (bot in text channels + voice channels).
- Modes are mutually exclusive, chosen at launch. Same brain/persona/memory in both.
- Local voice is the **reliable** real-time path. Discord voice works but rides an
  experimental library (see Discord section) — use local when you want smooth conversation.

## NEW: Phase 5 avatar (the body) — COMPLETE (pending visual tuning of the rest-pose angles)
- Mira now has a **VRM avatar** rendered in a local browser window (our own
  `@pixiv/three-vrm` renderer, no third-party app). `python main.py` brings it up
  automatically and opens the browser; capture it in OBS for streaming.
- Driven by **`motor_cortex.py`** (the *motor cortex*): a tiny local web server (aiohttp +
  WebSocket) that serves the renderer and streams face + gesture commands. The brain talks to the
  **cerebellum** (`brain/hindbrain/cerebellum/coordinator.py`), which smooths/maps movement and
  forwards to `motor_cortex`. See the "Avatar" section below.
- **What she does:** face follows `amygdala.mood`; **lip-sync** drives the mouth from the TTS
  amplitude envelope (local + Discord voice); body holds a **living procedural rest pose** (arms
  down, breathing, slow head/arm drift, blink) and **blends one-shot VRMA gestures in/out of it**
  with no snap. **Gestures are triggered by her spoken words** (greeting→wave, surprise→surprised,
  …) — not on a random timer. See the Avatar section + Status.

---

## Environment
- Project root: `D:\aiproject\`. Git repo, `.gitignore`, `.venv` present.
- `.gitignore` covers: `.venv/`, `__pycache__/`, `memory_store/`, `test_out.wav`, `.env`,
  `clock.json`.
- Installed: `openai`, `chromadb`, `requests`, `sounddevice`, `soundfile`,
  `faster-whisper`, `numpy`, **`aiohttp`** (avatar web/WS server), and for Discord:
  **`py-cord[voice]` (PINNED — see below)** + **`ffmpeg` on PATH** (for voice playback).
- Avatar front-end deps are **npm**, vendored under `avatar/node_modules` (run `npm install`
  in `avatar/`); needs **Node** on PATH. See the Avatar section.
- **CUDA DLL fix (Windows + faster-whisper):** `pip install nvidia-cublas-cu12
  "nvidia-cudnn-cu12==9.*"` AND the DLL-registration shim at the top of `wernickes_area.py`.
  Both required. Install with `python -m pip` so wheels land in the venv, not conda base.
- **GPT-SoVITS** installed SEPARATELY at `D:\GPT-SoVITS\` (own bundled Python). Not in git.

### ⚠️ py-cord is PINNED to an experimental commit — DO NOT casually upgrade
Discord voice **receive** requires DAVE (Discord's mandatory E2EE, enforced since Mar 2026).
Stable py-cord (2.8.x) does NOT decrypt received voice; an in-progress PR build does. We are
pinned to that build:
```
py-cord[voice] @ git+https://github.com/Pycord-Development/pycord.git@820460aa4
```
A plain `pip install -U py-cord` or a fresh venv will pull stable and **silently break voice
receive** — and the symptom looks identical to the bugs we already chased (noise / no audio).
Keep that exact line in requirements. Re-check the PR every few weeks; when py-cord ships
steady DAVE receive (or it merges to stable), unpin and lower `VOICE_END_SILENCE` back to ~2s.

---

## TTS setup — Piper + RVC (replaced GPT-SoVITS)
No server to start. Piper is a pip package; RVC runs as a subprocess.

### File layout
```
D:\aiproject\
├── piper_voices\en\en_US\hfc_female\medium\
│   ├── en_US-hfc_female-medium.onnx        ← already downloaded
│   └── en_US-hfc_female-medium.onnx.json
├── rvc_models\
│   ├── hubert_base.pt                       ← already downloaded (from lj1995/VoiceConversionWebUI)
│   ├── mira.pth                             ← DROP YOUR RVC MODEL HERE
│   └── mira.index                           ← DROP YOUR RVC INDEX HERE (optional)
└── rvc\
    └── rvc_infer.py                         ← standalone inference script
```

### Enabling / placing an RVC model
`USE_RVC` at the top of `brocas_area.py` toggles it (currently ON). Drop a **v2** `.pth` as
`D:\aiproject\rvc_models\mira.pth` and its `.index` as `mira.index`. Current voice = **Yui
(K-On!)**, a v2 40k model from HF `SmlCoke/rvc-yui`. Tunables at top of `brocas_area.py`:
`PITCH_SHIFT`, `INDEX_RATE` (0.5 default; ↑ for more character, ↓ if it warbles).

### How RVC runs — uses the proven `rvc_python` library (NOT a hand-rolled model)
An earlier version reimplemented the RVC network from scratch; it produced intelligible but
noisy/garbled audio. **Replaced with the `rvc_python` package + RMVPE F0** — far crisper.
`rvc_python` (and `torchcrepe`, `torchfcpe`) are installed into the **GPT-SoVITS bundled Python**
(`D:\GPT-SoVITS-v3lora-20250228\runtime\python.exe`, Python 3.9 with torch+cuda+faiss) — the main
venv (Python 3.14) can't install them. Installed there with `pip install <pkg> --no-deps` so the
existing torch/faiss/fairseq weren't disturbed.

`brocas_area.py` launches `rvc_infer.py --serve` ONCE as a **persistent subprocess**. It loads the
HuBERT + RMVPE + RVC model once (rvc_python auto-downloads HuBERT/RMVPE base models to its own
`base_models/`), prints `READY`, then takes `input|output|pitch` requests on stdin. Hot conversions
are **~0.7s**; cold load is ~10s, so `warmup()` pre-starts it. rvc_python auto-detects the GTX 1660
and forces fp32 (matches this GPU's broken-fp16 constraint).

### F0 method (crispness knob)
`rvc_infer.py` defaults `--f0method rmvpe` — the modern, cleanest pitch estimator. harvest (the old
hand-rolled default) sounds warbly by comparison. Other options: `pm` (fast/rough), `crepe`.

### Fallback behavior
- If `USE_RVC=False` or `mira.pth` is missing → uses raw Piper output (clean female voice).
- If RVC subprocess fails → same Piper fallback, no crash.
- Pitch shift: `PITCH_SHIFT` constant in `brocas_area.py` (0 = no shift, positive = higher).

### Running
1. Start **Ollama**.
2. From `D:\aiproject\` (venv active):
   - Local: `python main.py`
   - Discord: `python main.py --discord`

---

## Avatar setup — VRM in the browser (Phase 5)
Our own renderer; no VTube Studio / VSeeFace. Everything local and free.

### How it runs
- `motor_cortex.start()` (called from `main.py`) launches an **aiohttp server on
  `127.0.0.1:8234`** in its own thread + asyncio loop, serves `avatar/index.html`,
  and opens the browser. A `/ws` WebSocket streams blendshape + gesture messages.
- `avatar/index.html` loads `mira.vrm` with `@pixiv/three-vrm`, runs the render loop,
  applies expressions (mood + lip sync) to the face, holds a **living procedural rest pose**
  on the body, and blends one-shot VRMA gesture clips in/out of it.
- **Brain → avatar API.** The brain talks to the **cerebellum** (`coordinator.py`) for movement;
  it smooths/maps and forwards to `motor_cortex`, the raw output device. Both layers are safe
  no-ops if the server is down.
  - `cerebellum.set_mood(mood)` → `motor_cortex.set_mood(mood)` — amygdala mood → face (`MOOD_MAP`).
  - `cerebellum.lip(level)` → `motor_cortex.lipsync(level)` — smoothed mouth viseme (driven by TTS).
  - `cerebellum.gesture_for_speech(text)` / `cerebellum.gesture(name)` → `motor_cortex.play_gesture(name)`.
  - `motor_cortex.set_expressions({...})` — raw blendshape targets (escape hatch).

### File layout
```
avatar\
├── index.html          # three-vrm renderer + WS client + procedural rest pose (breathing/sway) + blink
├── serve.py            # standalone launcher (preview without the brain): python avatar/serve.py
├── mira.vrm            # the avatar model (gitignored — heavy binary, dropped in by user)
├── animations\*.vrma   # 11 gesture clips (gitignored; see licensing note)
├── package.json        # pins three + three-vrm + three-vrm-animation
└── node_modules\       # vendored deps (gitignored; reproduce with `npm install` in avatar/)
```

### Dependencies (vendored locally, offline-friendly)
- `three@0.169`, `@pixiv/three-vrm@3.5.3`, `@pixiv/three-vrm-animation@3.x` — installed
  via `npm install` in `avatar/` (Node present). Loaded in `index.html` through an
  importmap pointing at `/node_modules/...`. Nothing fetched from a CDN at runtime.

### Gestures (VRMA)
- 11 free `.vrma` clips pulled from the MIT repo `tk256ailab/vrm-viewer` into
  `avatar/animations/`. Friendly names → files in `GESTURES` (motor_cortex) / `GESTURES`
  (index.html): wave→Goodbye, thinking→Thinking, clapping→Clapping, flirty→Blush,
  plus surprised/angry/sad/jump/sleepy/look (the `Relax`/`LookAround` "idle"/"talking" entries
  are no longer auto-played — see below).
- **Idle is NOT a clip.** It's the procedural rest pose in the render loop (bind pose + arms down
  + breathing/head-drift/arm-sway). One-shot gestures are triggered by her **spoken words**
  (`cerebellum.gesture_for_speech`), blend in fast and ease back to the rest pose (`gestureWeight`,
  `GESTURE_IN`/`GESTURE_OUT`) — no clip-to-clip snap, no looping-idle snap, no random firing.
- **⚠ Licensing:** those clips' original provenance is undocumented (demo assets). Kept
  gitignored / local-only. For streaming/monetizing, swap in the unambiguously-free
  **VRoid official 7-pack** (https://vroid.booth.pm/items/5512385, manual BOOTH download).

### Streaming / rendering note
- The render loop uses `requestAnimationFrame` (smooth when the window is visible).
  Browsers **throttle background tabs** — a low-fps `setInterval` fallback keeps her from
  freezing if minimised, but for real capture use an **OBS Browser Source** (no throttling)
  or keep the window visible. Background is CSS chroma green (`#00b140`); canvas is
  transparent over it. Press `h` in the window to hide the debug HUD.
- Debug keys in the window: `1`–`6` expressions, `0` reset, `q/w/e/r/t/y/u` gestures,
  `p` toggle the procedural rest pose. Console: `__mira.state()` (shows `gestureWeight`/`restEnabled`),
  `__mira.play(name)`, `__mira.setRest('leftUpperArm', x, y, z)` to live-tune arm angles.

---

## Folder structure (current)
```
D:\aiproject\
├── main.py                 # entry point — routes ALL I/O through the active adapter
├── memory_store\           # Chroma vector store (auto-created, gitignored)
├── notes\                  # [NEW] note-taking sessions, one .txt each (auto-created, gitignored)
├── clock.json              # persistent last-active timestamp (gitignored)
├── PRIVACY_POLICY.md       # template (placeholders to fill: operator, contact, date, jurisdiction)
├── TERMS_OF_SERVICE.md     # template (same placeholders)
├── avatar\                             # [NEW] the body — browser VRM renderer (NOT a brain region)
│   ├── index.html / serve.py / mira.vrm / animations\*.vrma / node_modules\
├── peripheral_nervous_system\          # [NEW] swappable I/O substrate (NOT a brain region)
│   ├── io_adapter.py        # IOAdapter base + InputEvent dataclass + FINAL/PARTIAL/INTERRUPT
│   ├── local_adapter.py     # mic + speakers (wraps wernickes + brocas); local mode
│   └── discord_adapter.py   # Discord bot: text + voice (py-cord); the big one
└── brain\
    ├── forebrain\
    │   ├── cerebrum\
    │   │   ├── frontal_lobe\
    │   │   │   ├── prefrontal_cortex.py   # persona + GROUNDING + think() + consider_speaking()
    │   │   │   ├── dorsolateral_prefrontal_cortex.py  # [NEW] scribe: note-taking mode (record/organize/save)
    │   │   │   ├── brocas_area.py         # TTS: Piper (+ optional RVC) pipeline + lip-sync envelope
    │   │   │   ├── motor_cortex.py        # [NEW] avatar server: face (mood) + body (VRMA gestures)
    │   │   │   └── voice\reference.wav
    │   │   └── temporal_lobe\
    │   │       └── wernickes_area.py      # STT: faster-whisper (+ public transcribe())
    │   └── subcortical_structures\
    │       ├── basal_ganglia\
    │       │   └── action_selector.py     # should_respond() addressed-detection + engagement window
    │       ├── hypothalamus.py            # [NEW] persistent clock (last_active across restarts)
    │       └── limbic_system\
    │           ├── amygdala.py            # emotion state that colors responses
    │           └── hippocampus.py         # long-term memory + consolidation
    └── hindbrain\                         # [NEW]
        └── cerebellum\
            └── coordinator.py             # [NEW] lip smoothing + spoken-word→gesture mapping + mood→face
```
Every package dir has `__init__.py`. Brain core is I/O-agnostic; the PNS adapters feed it.

---

## Module summary (changes since Phase 3 handoff)

### prefrontal_cortex.py — "the CEO"
- `MODEL` constant (currently `qwen2.5:3b`) — used by `think()`, `consider_speaking()`, and
  the legacy `judge_relevance()`.
- **`GROUNDING` block** — hard anti-confabulation rules kept separate from persona: she may
  only reference people / events / past conversations that are in the current chat or in her
  recalled memories; otherwise she says she doesn't know rather than invent. There's also an
  explicit "no memories surfacing right now" guard. `temperature` lowered 0.9 → 0.7.
- `think(history, mood_flavor, memories, situation)` — generates a reply; used when she's
  **directly addressed** (name / @ / reply / DM) or interrupting. Always speaks.
- **`consider_speaking(...)`** — the autonomous floor decision for chatter she was NOT
  addressed in. ONE model call both decides AND writes: returns her line, or `[QUIET]` →
  stay silent. Biased toward joining in. This replaced the old rigid two-gate (should_respond
  "consider" + judge_relevance), which made her ignore too much. **Now driven by the
  subconscious** (cingulate cortex), not inline in main.py.
- **`wander(mode, ...)`** — [NEW] generates one brief inner thought (a daydream): `"memory"`
  (reminisce on a real memory) or `"curiosity"` (wonder about the unexperienced). Used by the
  subconscious when her mind is idle.
- `think(...)` / `consider_speaking(...)` now accept `inner_thoughts=` so her recent daydreams
  quietly color what she says.
- `_build_system(...)` — shared prompt assembly for both paths.

### cingulate_cortex/posterior_cingulate_cortex.py — [NEW] subconscious (default mode network)
- Always-on; two background threads. Three jobs:
  1. **Listening & drafting** — `observe_partial(text, channel)` is fed every live partial
     transcript from `on_event`. A dedicated drafter thread waits for the partial to *settle*
     (a phrase/sentence pause or the end-silence — so it doesn't burn a slow generation on every
     growing fragment) then drafts a full reply against the latest text + related memories. When
     the person stops, main calls **`take_draft(final)`**: if a draft fits what they actually
     said (partials are a growing prefix; the final == the last partial, so exact/prefix match is
     the common case) it's spoken immediately — no transcribe-then-think gap. Otherwise main
     falls back to `think()`. `end_listening()` drops the draft (interrupts / un-addressed).
  2. **Listening & deciding** — overheard (un-addressed) talk is handed via `heard(channel=...)`;
     it mulls for a beat then runs `consider_speaking` and decides whether/what/when to chime in.
  3. **Wandering** — when it's quiet (`WANDER_AFTER`) and no one is speaking, her mind drifts via
     `wander()`. Every thought is written to her private **subconscious_log** (NEVER the chat
     log); recent/related ones color later replies via `recent_thoughts(focus)`. A thought is
     voiced aloud only occasionally and **never while someone is speaking**.
- API: `start(speak, session_recap)`, `stop()`, `touch()`, `observe_partial()`, `take_draft()`,
  `end_listening()`, `heard()`, `recent_thoughts()`. Speaks only through main's serialized
  `speak_reply` (never overlaps a foreground turn).
- Latency note: drafting only beats `think()` if generation is faster than the speech+silence
  window. On a small GPU, model **swaps** dominate — keep one model warm. Best results with a
  single fast model everywhere (`MIRA_MODEL=qwen2.5:3b`); `MIRA_DRAFT_MODEL` can set a separate
  draft model but mixing models causes VRAM swaps that can make the first draft slow.
- Tunables: `MIRA_WANDER_*`, `MIRA_SUB_*`, `MIRA_DRAFT_MODEL`, `MIRA_MIN_DRAFT_WORDS`,
  `MIRA_DRAFT_SETTLE`, `MIRA_DRAFT_MAX_WAIT`, `MIRA_LISTEN_TIMEOUT`; `MIRA_SUB_DEBUG=1` traces it.

### cingulate_cortex/subconscious_log.py — [NEW] private stream of consciousness
- Persistent JSONL at `memory_store/subconscious_log.jsonl`. Her wandering thoughts live here,
  kept **separate from episodic memory** (a daydream is not a fact, so it can't leak into grounded
  recall). `record(text, mode)`, `recent(n)`, `related(query, n)` (instant keyword overlap, no
  model call), `review(n)`.

### thalamus.py — working memory + sequencing
- Added a `_seq` counter, `snapshot()` (consistent context+seq copy) and `awaiting_reply()`
  so the subconscious can cheaply tell what's new and whether the tail is an unanswered message.

### action_selector.py (basal ganglia)
- `should_respond(text, *, mentioned, reply_to_her, channel, now)` → Decision. Now used by
  main.py mainly to detect **addressed** (name/@/reply/DM). Un-addressed chatter goes to
  `consider_speaking`. `mark_engaged()` / window logic still present but no longer gates.

### hypothalamus.py — [NEW] persistent clock
- Persists `last_active` to `memory_store/clock.json`. `touch()`, `last_active()`,
  `time_since_last_active()`. Read at startup so Mira knows the downtime gap (used in the
  situation note). (The subconscious in the cingulate cortex now does run a background
  thread; the hypothalamus itself is still just the persistent clock.)

### wernickes_area.py — STT
- Local continuous-listening behavior unchanged. Added a public **`transcribe(f32_16k_mono)`**
  so non-mic sources (Discord) reuse Whisper without opening the mic. DLL shim must stay first.

### motor_cortex.py — [NEW] the body (avatar output)
- Hosts the avatar web server (aiohttp + WebSocket on `127.0.0.1:8234`) in its own thread;
  serves `avatar/` and streams face/gesture commands to the renderer. See the Avatar section.
- Public API: `start()/stop()`, `set_mood(mood)` (→ `MOOD_MAP` expression), `lipsync(level)`
  (mouth viseme; driven by TTS via the cerebellum), `play_gesture(name)` (→ `GESTURES` VRMA clips),
  `set_expressions({...})`. All non-throwing — if the server isn't up, calls are no-ops, so
  the brain/voice run fine headless.
- **All responses are served `no-store`** (a small aiohttp middleware). Without it Chrome
  aggressively caches the localhost HTML + ES modules and keeps running a **stale `index.html`**
  across reloads/relaunches — so renderer edits silently appear to do nothing. If you ever edit
  the renderer and see no change, hard-reload (`Ctrl+Shift+R`); `__mira.state()` confirms the
  loaded build.
- `motor_cortex` stays the raw output device; the **cerebellum** (below) is now the front door
  the brain uses for movement, and it forwards smoothed/timed commands here.

### cerebellum/coordinator.py — [NEW] the cerebellum (movement coordination)
- Sits between movement *intent* (the brain) and raw output (`motor_cortex`). Jobs:
  (1) **smooths the lip-sync** signal (fast attack / slower decay) before broadcast (`lip()`);
  (2) maps her **spoken words → a body gesture** (`gesture_for_speech(text)` against
  `_SPEECH_GESTURES`); (3) forwards **mood → face** (`set_mood`). `speaking_stopped()` closes the
  mouth. No background thread, no random gestures — the living idle is procedural in the renderer.
- Public API: `set_mood(mood)`, `gesture(name)`, `gesture_for_speech(text)`, `speaking_stopped()`,
  `lip(level)`. All forward to `motor_cortex` (safe no-ops headless). `main.py` wires
  `brocas_area.set_lip_callback(cerebellum.lip)`.

### main.py — adapter-routed loop
- Picks LocalAdapter or DiscordAdapter at launch; everything flows through `on_event`.
- `handle_message(event)`: ingests every utterance → detects addressed → **addressed/interrupt
  → think(); else → consider_speaking()** (silent = logged `[heard:quiet]`). Builds a
  time/place/silence "situation" note (per-channel + cross-restart via hypothalamus).
- **[NEW] Avatar:** starts `motor_cortex` (server/browser) at launch and wires
  `brocas_area.set_lip_callback(cerebellum.lip)`. Per turn it calls `cerebellum.set_mood(amygdala.mood)`
  (face follows mood), then `cerebellum.gesture_for_speech(reply)` right before speaking (plays a
  gesture only if her words call for one) and `cerebellum.speaking_stopped()` after. `motor_cortex.stop()`
  on quit. Avatar startup is wrapped — failure is non-fatal.
- PARTIAL display is a single-line rolling caption (truncated to terminal width) so live
  transcription doesn't wrap and stack into repeats.

### discord_adapter.py — Discord I/O (text + voice)  [NEW, the big one]
- **Client built inside its own event-loop thread** (py-cord voice internals bind to the loop
  active at construction; otherwise voice connect fails "Future attached to a different loop").
- **Text:** `on_message` → gating in the brain. Token via `.env` `DISCORD_BOT_TOKEN`
  (gitignored). Message Content Intent ON.
- **Voice:** `mira join` / `mira leave` commands. Receive via `vc.start_recording` + a custom
  `discord.sinks.Sink` whose `write()` hands decoded PCM to the adapter. STT via wernickes
  `transcribe`, TTS out via `vc.play(FFmpegPCMAudio(...))`. `_brain_busy` flag pauses live
  partials while she speaks so Whisper never fights the TTS for the GPU.
- **Endpointing now MIRRORS the local mic (rewritten for responsiveness).** Discord only sends
  packets while you actually speak, so silence never arrives on its own. A 50 ms ticker thread
  (`_voice_worker`) advances each speaker's buffer by either the audio that arrived or an equal
  slice of **silence**, making the stream look just like a mic. Then it runs the SAME in-buffer
  VAD endpointing as `wernickes_area._worker`: a turn ends after `VOICE_END_SILENCE` of trailing
  silence (default **2.0s**; local uses 3.0). This replaced the old wall-clock packet-gap timer,
  which dragged because the receive build delivers audio in late bursts. Constants
  (`VOICE_BLOCK_SEC`/`VOICE_REFRESH_SEC`/`VOICE_END_SILENCE`/`VOICE_MIN_SECONDS`/`VOICE_MAX_SECONDS`)
  sit at the top of `discord_adapter.py` and line up with the local mic's values.
  - **If turns get chopped mid-sentence**, the build is delivering in bursts with gaps > the
    threshold → raise `VOICE_END_SILENCE` toward 3.0. If delivery is steady, lower toward 1.0 for
    even snappier replies.
- **Speaker name resolution** (`_resolve_member`): py-cord may hand the sink a bare
  `discord.Object` (id only). Resolves id → VC channel members → guild cache → user cache.
  If a speaker still shows as `<Object id=…>`, enable the **Server Members Intent** (portal +
  `intents.members = True`) so the member list stays cached.

---

## Running the app
1. Start **Ollama**.
2. From `D:\aiproject\` (venv active):
   - Local: `python main.py` — just talk; pause and she replies. Use headphones.
   - Discord: `python main.py --discord` — then `mira join` in a server text channel while
     you're in a voice channel.
3. No GPT-SoVITS server needed anymore.

---

## Status

### PHASE 3 — VOICE: COMPLETE ✅ (local mic + speakers, unchanged)

### PHASE 4 — ATTENTION & BEHAVIOR: action selection done; two pieces deferred
- ✅ **Action selector (basal ganglia)** — `consider_speaking` autonomous "speak or stay quiet,"
  biased to engage; addressed → always answer. Replaces the old over-eager-to-ignore gate.
- ✅ **(Bonus, not in plan) Discord I/O layer** — text fully working; voice working with caveats.
- ✅ **Grounding / anti-confabulation** — GROUNDING rules + qwen2.5:3b + temp 0.7.
- ⬜ **RAS message prioritization** (spam filter, rank by relevance/subs/keywords) — DEFERRED.
  Matters for busy multi-viewer text chat; build alongside stream/audience work.
- ⬜ **Idle / default-mode behaviors** (ramble/comment when quiet) — DEFERRED. Best built with
  the avatar (it's stream dead-air filler).

### Discord voice — responsiveness (rewritten to match local)
- The voice path now uses the **silence-filled continuous timeline + in-buffer VAD** model
  (see discord_adapter.py above), so endpointing is as snappy as the local mic — a turn ends
  ~`VOICE_END_SILENCE` (2.0s) after you stop, instead of the old wall-clock packet-gap timer
  that dragged to ~6s. This is the main responsiveness fix.
- **Residual upstream caveat:** the pinned #3159 receive build can still deliver audio in late
  bursts with multi-second gaps, and occasionally stale/duplicated audio. A gap *mid-sentence*
  larger than `VOICE_END_SILENCE` can still chop a turn (the silence-fill reads it as a stop).
  If that happens, raise `VOICE_END_SILENCE` toward 3.0. The stale-audio flakiness is upstream
  and unchanged. Revisit when py-cord's DAVE receive matures.
- Barge-in (her yielding when you talk over her) was built then **removed** — an open/hot mic
  triggered it on background noise. Re-add later via push-to-talk or a "Mira stop" keyword if wanted.

### Still deferred from earlier phases (not blockers)
- **Per-person / per-channel memory** (who said what, last-seen) — belongs in hippocampus;
  needs speaker-identity + multi-person testing.
- **Phase 2 memory refinements:** consolidation dedup; a recall **relevance/distance threshold**
  (loosely-related memories can get injected → a vector cause of invention); crash-safe periodic
  session summaries.
- **ToS / Privacy** templates exist in the repo root — fill the placeholders + host them.

---

### PHASE 5 — BODY (avatar): COMPLETE ✅ (pending visual tuning of the rest-pose angles)
Decided: **own browser renderer** (VRM via `@pixiv/three-vrm`), not VTube Studio/VSeeFace.
Built in 5 steps; see the Avatar section for setup. Status:
- ✅ **Step 1 — renderer + server.** `motor_cortex` aiohttp/WS server serves `avatar/index.html`;
  loads `mira.vrm`, faces camera, chroma-green bg, idle blink/breathing. Verified rendering.
- ✅ **Step 2 — expressions ← mood.** `main.py` calls `set_mood(amygdala.mood)` per message;
  `MOOD_MAP` → VRM expression presets (happy/angry/sad/relaxed/surprised). Verified.
- ✅ **(asked for) VRMA gestures.** 11 clips load via `@pixiv/three-vrm-animation`; `play_gesture()`
  one-shots blend in over the procedural rest pose and ease back (see Step 5).
- ✅ **Step 3 — cerebellum smoothing/timing.** New `brain/hindbrain/cerebellum/coordinator.py`.
  Smooths the lip-sync signal (fast attack / slower decay) before it hits the socket, times the
  deliberate gestures, and gates idle behavior while she's busy. The brain now talks to the
  cerebellum for movement; it forwards to `motor_cortex`. (Face mood/lip still lerp in the browser.)
- ✅ **Step 4 — lip sync from TTS.** `brocas_area` computes a real-time RMS amplitude envelope and
  streams it (via `set_lip_callback` → cerebellum → `motor_cortex.lipsync`) timed to playback —
  **both** local (`_play`) and Discord voice (`_play_wav_in_vc` via `lip_drive_bytes`). Mouth `aa`
  viseme was already wired; this supplies the audio-driven values.
- ✅ **Step 5 — idle behaviors + rest pose + behavioral wiring.** Idle is **no longer a looping
  VRMA clip** (it snapped on each loop and tilted her head back). Idle is a **living procedural
  pose** in `index.html`: the model's captured **bind pose** (upright, facing the camera) with the
  arms overridden down at the sides, plus **breathing + slow head drift + faint arm sway** (layered
  unrelated sines, so non-repetitive) and blink. A `gestureWeight` blends the **whole body** between
  that rest pose and the active gesture clip with an **asymmetric** rate — fast in (`GESTURE_IN`
  0.12s) so the clip's full arm motion plays instead of being fought by the rest pose, slow out
  (`GESTURE_OUT` 0.5s) so the return doesn't snap. The full bind pose is the blend target, so no
  bones freeze after a gesture.
  - **Gestures are driven by her SPOKEN words, not a timer.** When she starts a line,
    `cerebellum.gesture_for_speech(reply)` scans it against `_SPEECH_GESTURES` (greeting→wave,
    surprise→surprised, …) and plays at most one; otherwise she just stands and talks facing the
    camera. The old random idle-gesture scheduler and the thinking/greet gestures were **removed**
    at the user's request.
  - **⚠ Rest-pose arm angles need a visual pass on the real model.** The four arm Eulers in `POSE`
    (`index.html`, ~68° down) depend on the rig. Tune live in the browser console:
    `__mira.setRest('leftUpperArm', 0, 0, -1.2)`; press `p` to toggle the rest pose; `__mira.state()`
    shows `gestureWeight`/`restEnabled`.

---

### PHASE 6 — SENSES: NOT STARTED (this is next)
Goal: Mira can **see** what's on screen (the game) and **hear** non-speech audio events
(alerts/donations/sounds), and react to both — grounded, so she only references what the senses
actually reported (same anti-confabulation rule as everything else).

Anatomy / where the code goes (new brain regions, mirror the existing tree):
- **Visual cortex** → `brain/forebrain/cerebrum/occipital_lobe/visual_cortex.py` (new `occipital_lobe/`).
- **Auditory cortex** (non-speech sound) → `brain/forebrain/cerebrum/temporal_lobe/auditory_cortex.py`
  (sits beside `wernickes_area.py`, which stays the *language* path).
- Both feed the brain through the existing relay: emit an `InputEvent` (new `channel`, e.g.
  `"vision"` / `"event"`) or fold into the `describe_situation` note in `main.py` — no new brain
  branching needed.

Steps:
- ⬜ **Step 1 — screen capture (frame source).** Grab a screen region (game window / a monitor) at a
  LOW rate (~0.5–1 fps) with **`mss`** (fast, local, free). Config: which region, which fps. Keep the
  latest frame in memory; don't save to disk. → `visual_cortex.py`.
- ⬜ **Step 2 — see the frame (local VLM).** Describe the frame with a **small local vision model via
  Ollama** (e.g. `qwen2.5vl:3b` / `moondream` / `llava`), producing a one-line scene description.
  ⚠ **VRAM is the wall:** a VLM + chat LLM + Whisper + TTS will NOT co-reside in 6GB. Plan to
  time-share — describe only occasionally (every N seconds or on demand), and/or pause STT while the
  VLM runs, the same squeeze pattern as the rest of the project. Set the VLM model in ONE constant.
- ⬜ **Step 3 — wire vision into context.** Inject the latest scene description into the `situation`
  note (or as a low-priority `InputEvent`) so `think()`/`consider_speaking()` can comment on the
  game. Grounding: she may only mention what the description contains — if vision is off/empty, she
  doesn't pretend to see.
- ⬜ **Step 4 — auditory cortex (events).** Detect non-speech events. Two sources, easiest first:
  (a) **platform/event webhooks** (Twitch/YouTube/StreamElements donation·sub·raid) — reliable, and
  ties into the deferred Phase-4 RAS prioritization; (b) **system-audio event detection** (alert
  sound onset/template match) for sounds without an API. Map events → an `amygdala` mood bump
  (donation → `excited`) + a one-shot utterance trigger ("thank the donor by name", grounded in the
  event payload). → `auditory_cortex.py`.
- ⬜ **Step 5 — react with the body.** Reuse Phase 5: an event/scene can set mood (face) and, via the
  speech→gesture map, play a fitting gesture (e.g. donation → `clapping`/`jump`). No new avatar work.

First concrete task for the next session: **Step 1** — add `occipital_lobe/visual_cortex.py` with an
`mss`-based capture loop exposing `latest_frame()` / `start()/stop()`, region+fps constants, behind a
master `USE_VISION` switch (off by default so the base app is unaffected). Then Step 2 on top.

### PHASE 7 — HOMEOSTASIS (polish): NOT STARTED
- ⬜ **Energy/mood decay over stream time + a true scheduler** → extends `hypothalamus.py` (it already
  persists the clock but does no background processing yet). Mood drifts toward baseline over time;
  scheduled behaviors fire on a timer.
- ⬜ **Self-correction** → *anterior cingulate*: detect bad/off outputs (refusals, format leaks,
  repetition) and retry/filter before they reach TTS.

## Working style for next session
- Step by step, one step at a time, not verbose.
- Stay local / free unless a fully-free fast option is explicitly wanted.
- Before any `pip install -U`, remember the py-cord pin.