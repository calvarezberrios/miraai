# Mira — Local AI VTuber

Mira is a fully local AI VTuber, architected so each software module maps to a region of the
human brain. She runs entirely on your machine (no cloud LLMs): a local LLM for thinking,
local neural TTS (with optional RVC voice conversion) for speech, and local speech-to-text for
listening. She can run in two modes — a local mic/speaker conversation, or as a Discord bot in
text and voice channels.

> **This is a personal/hobby project.** Everything runs locally; you supply your own models,
> and (for Discord) your own bot token.

---

## 1. Prerequisites

- **OS:** Windows 10/11 (commands below are PowerShell). Adaptable to Linux with minor changes.
- **Python 3.11** (3.10–3.12 OK). Use a dedicated virtual environment.
- **NVIDIA GPU + recent driver.** See the **GPU notes** section — settings differ by card.
- **Git.**
- **ffmpeg** on your PATH (required for Discord voice playback).
  Windows: `winget install Gyan.FFmpeg` (then reopen the terminal), or download from ffmpeg.org.
- **Ollama** (the LLM runtime) — installed separately (next section).
- **TTS** is **Kokoro** (hexgrad/Kokoro-82M) by default. Mira's voice is the Kokoro **"Jessica"**
  voice (`af_jessica`); weights auto-download from Hugging Face on first run. Kokoro can't install
  on this project's Python 3.14 (it pins numpy and pulls spacy), so it runs in a **dedicated
  Python 3.10 venv (`.venv-kokoro`) as a subprocess** — the same pattern RVC uses. espeak-ng is
  bundled by a pip dep, so there's nothing to install on PATH. **Piper + optional RVC** remains
  available as an alternate engine (`MIRA_TTS=piper`) — see section 3b. RVC reuses the **GPT-SoVITS
  bundled runtime's Python** only as an interpreter that has the right deps; you do NOT run the
  GPT-SoVITS server.

---

## 2. Install Ollama + pull the models

1. Install Ollama from https://ollama.com and make sure it's running (tray app or `ollama serve`).
2. Pull the models:
   ```powershell
   ollama pull qwen2.5:3b          # chat model (see GPU notes for bigger options)
   ollama pull nomic-embed-text    # embeddings for long-term memory
   ```

---

## 3. Voice (TTS): Kokoro (default), or Piper + optional RVC

Pick the engine with the `MIRA_TTS` env var: `kokoro` (default), `piper`, or `gptsovits`.

### 3a. Kokoro (default)
Kokoro pins numpy and pulls `misaki[en]`→`spacy`, none of which have Python 3.14 wheels, so it
**cannot** live in the main venv. Instead it runs in a dedicated **Python 3.10 venv** as a
persistent subprocess (`kokoro_tts\kokoro_infer.py`), which `brocas_area.py` launches and keeps
hot — the same approach as RVC. One-time setup:

```powershell
py -3.10 -m venv .venv-kokoro
.\.venv-kokoro\Scripts\python.exe -m pip install kokoro soundfile
.\.venv-kokoro\Scripts\python.exe -m spacy download en_core_web_sm
```

> ⚠️ **GPU torch for Kokoro.** `pip install kokoro` pulls the **CPU-only** PyTorch wheel, so
> Kokoro will synthesize on the CPU (no NVIDIA GPU usage) even though it auto-selects CUDA when
> available. After the install above, replace torch with a CUDA build that matches your card.
> For an RTX 50-series (Blackwell, `sm_120`) the CUDA 13 wheel works:
> ```powershell
> .\.venv-kokoro\Scripts\python.exe -m pip install "torch==2.12.1+cu130" --index-url https://download.pytorch.org/whl/cu130
> ```
> For older cards a cu128 build (`--index-url https://download.pytorch.org/whl/cu128`) is fine.
> Verify with `... -c "import torch; print(torch.cuda.is_available())"` → `True`. On startup
> `kokoro_infer.py` logs `device=cuda:0` (and warns if it fell back to CPU).

The Kokoro-82M weights and the **"Jessica"** voice (`af_jessica`) auto-download on first run — no
files to place. (The `en_core_web_sm` line above is what misaki's English g2p needs; if you skip
it, the very first launch will install it mid-run and fail once, then work — so just pre-install.) **espeak-ng** is bundled by the
`espeakng-loader` dep, so nothing goes on PATH. Tunables live at the top of `brocas_area.py`:
`KOKORO_VOICE`, `KOKORO_LANG` (`a` = American English), `KOKORO_SPEED`, and `KOKORO_PYTHON`
(path to the 3.10 venv's `python.exe`) — or the matching `MIRA_KOKORO_*` env vars.

### 3b. Piper + optional RVC voice conversion (alternate engine, `MIRA_TTS=piper`)
The Piper pipeline is **Piper → (optional) RVC → playback**. Piper is a fast neural TTS that
produces a clean female voice; RVC optionally converts that timbre to a character voice.

**Piper:** installed as a pip package in section 4. The voice model (`hfc_female`) is downloaded
once from Hugging Face into `piper_voices\`. Paths live at the top of `brocas_area.py`
(`PIPER_MODEL`). With RVC off, Mira speaks with this clean female voice — no extra setup.

**RVC voice conversion (optional — for a character voice):**
RVC turns Piper's voice into a trained character voice (`.pth` + `.index` model files). It needs
`fairseq`/`faiss`/`torch+cuda`, which won't install on Python 3.14 — so RVC inference runs as a
subprocess under the **GPT-SoVITS bundled runtime's Python** (Python 3.9, which already has those).
You do **not** run the GPT-SoVITS server; it's only borrowed as an interpreter.

1. Have the GPT-SoVITS integrated package extracted somewhere (e.g.
   `D:\GPT-SoVITS-v3lora-20250228\`). Point `RVC_PYTHON` at its `runtime\python.exe`
   (top of `brocas_area.py`).
2. Install the RVC library into that runtime (`--no-deps` so it can't disturb the existing
   torch/faiss/fairseq):
   ```powershell
   & "D:\GPT-SoVITS-v3lora-20250228\runtime\python.exe" -m pip install rvc-python torchcrepe torchfcpe --no-deps
   ```
3. Put a **v2** RVC model at `rvc_models\mira.pth` and its index at `rvc_models\mira.index`.
   (Plenty of free anime-voice models on Hugging Face; e.g. `SmlCoke/rvc-yui`. Prefer v2, 40k,
   with a real index for clean output.) The HuBERT/RMVPE base models auto-download on first run.
4. Enable it: set `USE_RVC = True` at the top of `brocas_area.py`. Tunables there: `PITCH_SHIFT`,
   `INDEX_RATE` (0.5 default). RVC uses **RMVPE** pitch estimation for crisp output.

If `USE_RVC = False` or the model is missing, Mira falls back to the plain Piper voice — no crash.

---

## 4. Set up this project

```powershell
git clone <your-repo-url> aiproject
cd aiproject

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 4a. Core Python packages
```powershell
python -m pip install openai chromadb sounddevice soundfile faster-whisper numpy piper-tts
```
(Or just `python -m pip install -r requirements.txt`.) Kokoro is **not** in this list — it installs
into its own `.venv-kokoro` (section 3a), not the main venv.

### 4b. CUDA libraries for faster-whisper (Windows)
CTranslate2 (under faster-whisper) needs cuBLAS/cuDNN DLLs that aren't on PATH by default:
```powershell
python -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```
(`wernickes_area.py` has a shim at the very top that registers these DLLs at runtime — keep it
first in the file.)

### 4c. Discord support (only if you want Discord mode)
Discord voice **receive** requires DAVE (Discord's mandatory end-to-end encryption). Stable
py-cord does not decrypt received voice yet, so we pin an exact in-progress build:
```powershell
python -m pip install -U "py-cord[voice] @ git+https://github.com/Pycord-Development/pycord.git@820460aa4"
```
> ⚠️ **Do not run `pip install -U py-cord` later** — it will replace this with stable py-cord and
> silently break voice receive. Keep the pinned line above. Re-check the project's voice-receive
> PR periodically; once it ships in a release, you can unpin.
>
> This pinned build has a voice-receive bug that made audio arrive in ~5s bursts; we work around it
> in-app (`_loop_pump`). See **`DISCORD_VOICE_FIX.md`** — it also covers whether/how to report the
> bug upstream, and is worth checking before bumping the pin (the upstream fix may land there).

Also confirm **ffmpeg** is on PATH (`ffmpeg -version`).

---

## 5. Configuration

- **`.env` file** (project root): on startup `env_loader.py` loads `KEY=VALUE` lines from `.env`
  into the environment, so any `MIRA_*` / config var can live there instead of the system
  environment. A real system/shell env var still wins over `.env` (so you can override per-run).
  Example `.env`:
  ```
  DISCORD_BOT_TOKEN=your-bot-token-here
  MIRA_MODEL=mira          # the Ollama model Mira's brain talks to (default: qwen2.5:3b)
  ```
- **LLM provider — Ollama (default) or Gemini.** Mira's chat brain talks to an
  OpenAI-compatible endpoint, so you can point it at Google **Gemini** to compare
  response quality. Set in `.env`:
  ```
  MIRA_LLM_PROVIDER=gemini             # "ollama" (default) or "gemini"
  GEMINI_API_KEY=your-gemini-api-key
  MIRA_GEMINI_MODEL=gemini-2.0-flash   # optional; e.g. gemini-2.5-flash, gemini-1.5-pro
  ```
  This routes `think` / `consider_speaking` / `wander`, the subconscious, the note-taking
  scribe, **and** memory consolidation through Gemini. **Ollama is still required for
  long-term memory**: embeddings stay on local `nomic-embed-text` because the Chroma store
  is built with them (they can't switch providers without rebuilding the store). If Ollama
  isn't running, chat still works on Gemini — she just won't recall or store memories that
  run. Set `MIRA_LLM_PROVIDER=ollama` (or remove it) to go back to fully local.
- **Discord bot token** (Discord mode only): set `DISCORD_BOT_TOKEN` in `.env` as above.
  In the Discord Developer Portal, enable the **Message Content Intent** (and, for reliable
  speaker names in voice, the **Server Members Intent**). Invite the bot to your server.
- `.gitignore` already excludes secrets and local state: `.venv/`, `__pycache__/`,
  `memory_store/`, `clock.json`, `.env`, `test_out.wav`. A fresh machine starts with **no
  memories** (Chroma store is created on first run).

---

## 6. GPU notes (IMPORTANT when moving to a better GPU)

The old machine was a GTX 1660 Super (6GB, Turing TU116) which has a **broken fp16 path** and
tight VRAM, forcing several conservative settings. On a newer/larger card you should change them:

| Setting | Old (1660 Super) | Better GPU (RTX 20-series+ / 12GB+) |
|---|---|---|
| faster-whisper `COMPUTE_TYPE` (top of `wernickes_area.py`) | `int8` | `float16` — faster/cleaner if VRAM allows |
| faster-whisper `MODEL_SIZE` | `small` | `distil-large-v3` (current default) — large-v3 accuracy at ~3x the speed, English-only. Use `large-v3` if you need multilingual; `medium` for less VRAM |
| Chat `MODEL` (top of `prefrontal_cortex.py`) | `qwen2.5:3b` | `qwen2.5:7b` or `llama3.1:8b` — better grounding, less invention |

Kokoro and Piper TTS are both light and run fine on any GPU (CPU is fine too). With more
VRAM the engines (LLM + Whisper + TTS) coexist comfortably, so the day-to-day VRAM squeeze
goes away. Note: `VOICE_END_SILENCE` in `discord_adapter.py` (default 3.0s, env
`MIRA_VOICE_END_SILENCE`) controls how long after you stop talking her Discord turn ends; the
local mic's equivalent is `END_SILENCE_SEC` (2.25s, env `MIRA_END_SILENCE_SEC`). Lower for
snappier turn-taking; raise if she cuts you off during a pause.

**Getting cut off / chopped into chunks in Discord voice?** Her turn ends only on a gap in your
**transmission** — `discord_adapter._endpoint_speaker` watches `now - last_voice` (last_voice is
bumped on every 20ms voice packet / Discord's `is_speaking`), NOT silence inside the audio buffer,
and it never finalizes a new turn while she's mid-reply (`_brain_busy`). **The old ~5s "stutter /
cut off every few words" problem was a bug in the pinned py-cord voice build (received audio was
processed only when the event loop's ~5s heartbeat woke it, so it arrived in 5-second bursts). It's
fixed** by a `_loop_pump` thread that keeps the loop awake during voice — see **`DISCORD_VOICE_FIX.md`**
for the full root cause + the upstream-PR assessment. Tuning: `MIRA_VOICE_END_SILENCE` (0.7s gap
after you stop — snappy now that delivery is real-time), `MIRA_VOICE_MAX_SECONDS` (60s safety cap).
Diagnostics (all default-off): `MIRA_VOICE_DEBUG=1` (prints `delivery gap`/`finalized` lines — a
return of steady ~5s `delivery gap`s means the pump isn't keeping up), `MIRA_VOICE_DUMP=1` (writes
each utterance to `voice_debug/*.wav` so you can *listen* to what was captured), `MIRA_VOICE_LOG=1`
(py-cord DAVE decrypt log).

---

## 7. Running

Start order: **Ollama** running → then the app. (No TTS server to start — Kokoro/Piper run inline,
and RVC, if the Piper engine is enabled, is launched automatically as a subprocess.)

- **Local mode** (mic + speakers — the smooth, reliable real-time path):
  ```powershell
  python main.py
  ```
  Just talk; pause ~3s and she replies. Typing in the console also works. `quit` / `exit` to
  close (runs memory consolidation + a session summary on the way out). **Use headphones** —
  on speakers her voice can bleed into the mic.

- **Discord mode** (text + voice channels):
  ```powershell
  python main.py --discord
  ```
  Then, in a server text channel (with the bot present), type `mira join` while you're in a
  voice channel; `mira leave` to disconnect. Text chat works anywhere she can see.
  (On the split laptop/desktop setup just run **`start_discord.bat`**, which sets the LAN brain
  endpoints and launches `--discord --draft`. Voice speaker names need **Server Members Intent**
  enabled in the Developer Portal.)

- **Faster replies — pre-drafting** (`--draft`, or implied by `--subconscious`): she drafts a reply
  *while you're still talking* and speaks it the instant you stop, instead of transcribe-then-think.
  Identity-aware, no chime-ins or daydreams. `--subconscious` adds the full background mind
  (autonomous chime-ins + mind-wandering).

- **Knowing who's talking:** in Discord she identifies speakers by their **display name** and treats
  an unfamiliar name as someone new until told who they are ("I'm GameRaiderX" is remembered across
  sessions). Needs Server Members Intent (above).

First run downloads the Whisper model (one-time).

### Note-taking
Mira can take notes of whatever she hears (local mic or a Discord voice channel) — but
**only when asked**, and while she's taking notes she stays **silent and just listens**
(her subconscious is paused too). Say or type:

- **Start:** `take notes` (optionally `take notes about <topic>`). For a tabletop game,
  `take TTRPG notes` / `take D&D notes` / `take game notes` → notes get organized **by
  player and their character**. In Discord voice each speaker is recorded by name; you can
  also tell her `Mira, Alice plays Lyra` to label a character.
- **Recap:** `recap` / `summarize` / `what do you have so far` → she posts an organized
  recap (to the console / Discord text channel — never spoken).
- **Stop:** `stop taking notes` → she finalizes the file with topic- (or player/character-)
  organized notes plus a summary.

Notes are saved as one `.txt` per session under `notes\` (gitignored), named
`<main-topic>_<timestamp>.txt`. The raw transcript is written live (so nothing is lost if
the app crashes); quitting mid-session also finalizes the file.

---

## 8. Troubleshooting

- **Mira sounds garbled / noisy with RVC on** → make sure the RVC library is installed in the
  GPT-SoVITS runtime (3b) and you're using a **v2** model with a real `.index`. A bad/over-trained
  model will sound rough no matter what; try a different one, or set `USE_RVC = False` for the
  clean Piper voice. Tune `INDEX_RATE` (lower if it warbles) and `PITCH_SHIFT`.
- **`RuntimeError: cublas64_12.dll not found`** → run the 4b install; confirm the DLL shim is
  still at the top of `wernickes_area.py`. Install with `python -m pip` so wheels land in `.venv`.
- **Kokoro TTS: `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` at synth time** (model loads on
  `cuda:0`, then the first conv dies) → a cuDNN clash between the two venvs. faster-whisper's main
  `.venv` ships **CUDA-12 cuDNN** (`nvidia-cudnn-cu12`) and prepends it to `PATH`; the Kokoro
  subprocess inherits that `PATH` but its torch (cu130) bundles **CUDA-13 cuDNN** in `torch/lib`.
  Both files are named `cudnn64_9.dll`, so Kokoro loads the wrong one. `kokoro_infer.py` has a
  `_isolate_cuda_dlls()` shim at the very top (before torch import) that strips foreign CUDA dirs
  from `PATH` and puts its own `torch/lib` first — keep it there. (Only matters when STT and TTS
  share a process tree, e.g. the real app; a standalone Kokoro run won't hit it.)
- **Discord voice: noise / no audio received, or it "hears" nothing** → you're on stable py-cord.
  Reinstall the pinned build from 4c and confirm with
  `python -c "import discord; print(discord.__version__)"` (should be a dev build, not 2.8.x).
- **Discord voice splits your speech into chunks** → the experimental build delivers audio in
  bursts; raise `VOICE_END_SILENCE` (top of `discord_adapter.py`) toward 3.0. Local mode avoids
  this entirely.
- **A Discord speaker shows as `<Object id=…>`** → enable the **Server Members Intent** in the
  Developer Portal and add `intents.members = True` with the other intents in `discord_adapter.py`.
- **No voice in Discord at all** → ensure `ffmpeg` is on PATH.

---

## Project layout (high level)

```
aiproject\
├── main.py                          # entry point; routes I/O through the active adapter
├── peripheral_nervous_system\       # swappable I/O: local (mic/speakers) and Discord adapters
└── brain\forebrain\
    ├── cerebrum\frontal_lobe\       # prefrontal_cortex (LLM+persona), brocas_area (TTS)
    ├── cerebrum\temporal_lobe\      # wernickes_area (STT)
    └── subcortical_structures\      # basal_ganglia, hypothalamus, limbic_system (amygdala, hippocampus)
```

See `handoff.md` for architecture details and current build status.
