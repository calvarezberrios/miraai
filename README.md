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
- **TTS** is **Piper** (a pip package, installed into this venv). Mira's voice is the Piper
  `hfc_female` voice by default. An optional **RVC** stage can convert that to a character voice
  (e.g. an anime girl) — see section 3b. RVC reuses the **GPT-SoVITS bundled runtime's Python**
  only as an interpreter that has the right deps; you do NOT run the GPT-SoVITS server.

---

## 2. Install Ollama + pull the models

1. Install Ollama from https://ollama.com and make sure it's running (tray app or `ollama serve`).
2. Pull the models:
   ```powershell
   ollama pull qwen2.5:3b          # chat model (see GPU notes for bigger options)
   ollama pull nomic-embed-text    # embeddings for long-term memory
   ```

---

## 3. Voice (TTS): Piper, with optional RVC voice conversion

The voice pipeline is **Piper → (optional) RVC → playback**. Piper is a fast neural TTS that
produces a clean female voice; RVC optionally converts that timbre to a character voice.

### 3a. Piper (required, easy)
Piper is installed as a pip package in section 4. The voice model (`hfc_female`) is downloaded
once from Hugging Face into `piper_voices\`. Paths live at the top of `brocas_area.py`
(`PIPER_MODEL`). With RVC off, Mira speaks with this clean female voice — no extra setup.

### 3b. RVC voice conversion (optional — for a character voice)
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
(Or just `python -m pip install -r requirements.txt`.)

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

Also confirm **ffmpeg** is on PATH (`ffmpeg -version`).

---

## 5. Configuration

- **Discord bot token** (Discord mode only): create a file named `.env` in the project root:
  ```
  DISCORD_BOT_TOKEN=your-bot-token-here
  ```
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
| faster-whisper `MODEL_SIZE` | `small` | `medium` / `large-v3` for better accuracy |
| Chat `MODEL` (top of `prefrontal_cortex.py`) | `qwen2.5:3b` | `qwen2.5:7b` or `llama3.1:8b` — better grounding, less invention |

Piper TTS is light and runs fine on any GPU (it auto-detects the 1660 and forces fp32). With more
VRAM the engines (LLM + Whisper + Piper/RVC) coexist comfortably, so the day-to-day VRAM squeeze
goes away. Note: `VOICE_END_SILENCE` in `discord_adapter.py` (default 2.0s) controls Discord voice
endpointing — raise it toward 3.0 only if the experimental py-cord build's bursty delivery chops
your turns; that's a *library* limitation, not a GPU one.

---

## 7. Running

Start order: **Ollama** running → then the app. (No TTS server to start — Piper runs inline and
RVC, if enabled, is launched automatically as a subprocess.)

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

First run downloads the Whisper model (one-time).

---

## 8. Troubleshooting

- **Mira sounds garbled / noisy with RVC on** → make sure the RVC library is installed in the
  GPT-SoVITS runtime (3b) and you're using a **v2** model with a real `.index`. A bad/over-trained
  model will sound rough no matter what; try a different one, or set `USE_RVC = False` for the
  clean Piper voice. Tune `INDEX_RATE` (lower if it warbles) and `PITCH_SHIFT`.
- **`RuntimeError: cublas64_12.dll not found`** → run the 4b install; confirm the DLL shim is
  still at the top of `wernickes_area.py`. Install with `python -m pip` so wheels land in `.venv`.
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
