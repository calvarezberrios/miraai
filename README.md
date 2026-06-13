# Mira — Local AI VTuber

Mira is a fully local AI VTuber, architected so each software module maps to a region of the
human brain. She runs entirely on your machine (no cloud LLMs): a local LLM for thinking,
local voice cloning for speech, and local speech-to-text for listening. She can run in two
modes — a local mic/speaker conversation, or as a Discord bot in text and voice channels.

> **This is a personal/hobby project.** Everything runs locally; you supply your own models,
> a voice reference clip, and (for Discord) your own bot token.

---

## 1. Prerequisites

- **OS:** Windows 10/11 (commands below are PowerShell). Adaptable to Linux with minor changes.
- **Python 3.11** (3.10–3.12 OK). Use a dedicated virtual environment.
- **NVIDIA GPU + recent driver.** See the **GPU notes** section — settings differ by card.
- **Git.**
- **ffmpeg** on your PATH (required for Discord voice playback).
  Windows: `winget install Gyan.FFmpeg` (then reopen the terminal), or download from ffmpeg.org.
- Two background services you install separately: **Ollama** (the LLM runtime) and
  **GPT-SoVITS** (the voice/TTS engine).

---

## 2. Install Ollama + pull the models

1. Install Ollama from https://ollama.com and make sure it's running (tray app or `ollama serve`).
2. Pull the models:
   ```powershell
   ollama pull qwen2.5:3b          # chat model (see GPU notes for bigger options)
   ollama pull nomic-embed-text    # embeddings for long-term memory
   ```

---

## 3. Install GPT-SoVITS (the voice engine) — separate from this repo

GPT-SoVITS is a large standalone package with its own bundled Python; **do not** install it
into this project's venv.

1. Download the Windows **integrated package** for GPT-SoVITS and extract it (e.g. to
   `D:\GPT-SoVITS\`).
2. **Precision setting** — edit `GPT_SoVITS\configs\tts_infer.yaml`:
   - Keep `device: cuda`.
   - Set `is_half:` per your GPU (see **GPU notes**). On cards with a working fp16 path use
     `true` (faster); on GTX 16-series use `false` (fp16 is broken there → silent output).
3. **Reference voice:** Mira's voice is cloned from a 3–10s clean, single-speaker clip. The clip
   ships in this repo at `brain\forebrain\cerebrum\frontal_lobe\voice\reference.wav`, and its
   transcript is in the `VOICE` dict in `brocas_area.py`. To change her voice, swap the clip and
   update that transcript.
4. Start the API server (leave it running in its own terminal):
   ```powershell
   cd D:\GPT-SoVITS
   .\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880
   ```
   Wait for the models-loaded message. The app talks to it at `http://127.0.0.1:9880`.

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
python -m pip install openai chromadb requests sounddevice soundfile faster-whisper numpy
```

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
| GPT-SoVITS `is_half` (in `tts_infer.yaml`) | `false` (fp16 broken) | **`true`** — much faster synthesis |
| faster-whisper `COMPUTE_TYPE` (top of `wernickes_area.py`) | `int8` | `float16` — faster/cleaner if VRAM allows |
| faster-whisper `MODEL_SIZE` | `small` | `medium` / `large-v3` for better accuracy |
| Chat `MODEL` (top of `prefrontal_cortex.py`) | `qwen2.5:3b` | `qwen2.5:7b` or `llama3.1:8b` — better grounding, less invention |

With more VRAM all three engines (LLM + Whisper + GPT-SoVITS) coexist comfortably, so the
day-to-day VRAM squeeze goes away. Note: `VOICE_END_SILENCE` in `discord_adapter.py` is set high
(6.0s) to work around the experimental py-cord build's bursty audio delivery — that's a *library*
limitation, not a GPU one, so leave it until py-cord's voice receive matures.

---

## 7. Running

Start order matters: **Ollama** running → **GPT-SoVITS** API running → then the app.

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

- **TTS produces a full-length but SILENT wav** → `is_half: true` on a GPU with a broken fp16
  path. Set `is_half: false` in `tts_infer.yaml` and restart GPT-SoVITS. (Not needed on a
  capable GPU.)
- **`RuntimeError: cublas64_12.dll not found`** → run the 4b install; confirm the DLL shim is
  still at the top of `wernickes_area.py`. Install with `python -m pip` so wheels land in `.venv`.
- **Discord voice: noise / no audio received, or it "hears" nothing** → you're on stable py-cord.
  Reinstall the pinned build from 4c and confirm with
  `python -c "import discord; print(discord.__version__)"` (should be a dev build, not 2.8.x).
- **Discord voice splits your speech into chunks** → the experimental build delivers audio in
  bursts; `VOICE_END_SILENCE = 6.0` mitigates it. Local mode avoids this entirely.
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
