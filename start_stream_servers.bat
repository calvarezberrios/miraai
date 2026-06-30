@echo off
setlocal
REM ============================================================================
REM  start_stream_servers.bat  --  Mira in FULL STREAM mode, ALL on THIS laptop.
REM  (branch: laptop_run)
REM
REM  One box (RTX 5050, 8 GB) runs EVERYTHING for a live stream:
REM    - BRAIN + EYES : Qwen2.5-VL-7B (llama.cpp in Docker) on :8080  ~6.9 GB VRAM
REM                     (one model is both her chat brain AND her vision — see VISION_SETUP.md)
REM    - STT          : Whisper small.en on the CPU  (the GPU is full with the VL model;
REM                     the CPU is otherwise idle, so STT runs there with no contention)
REM    - TTS          : Piper (+ optional RVC) — CPU/light
REM    - MEMORY       : local Ollama nomic-embed-text on :11434
REM    - Mira         : Discord VC + Twitch chat + HOST + VISION + GAME AUDIO
REM
REM  Why CPU Whisper here (vs GPU in start_discord.bat): the 7B vision model nearly fills the
REM  8 GB card, so there's no room for GPU STT. The LLM is fully GPU-offloaded, so the CPU is
REM  free and runs small.en fine. (Alternative: VL-3B + GPU Whisper — edit -Size 3b below and
REM  set WHISPER_DEVICE=cuda. Weaker vision, faster STT.)
REM
REM  ONE-TIME PREREQS:
REM    - Docker Desktop running; turboquant binary built once (build-turboquant.ps1)
REM    - VL model + mmproj in C:\models  (already present: Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
REM      + mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf; else see llm-test\download-vision.ps1)
REM    - Piper voice in C:\models\piper  (see start_discord.bat)
REM    - Ollama with nomic-embed-text
REM    - .env filled in: DISCORD_BOT_TOKEN, TWITCH_CHANNEL, TWITCH_CLIENT_ID/SECRET
REM
REM  SCREEN + GAME AUDIO come from the DESKTOP (that's where the game/stream is), NOT this
REM  laptop. On the DESKTOP, run the senses companion in the repo venv:
REM      python tools\desktop_senses.py
REM  It captures the desktop's screen + game-audio loopback and serves them on :8200. Set
REM  DESKTOP_IP below to the desktop's LAN IP (the companion prints it on start). On the
REM  desktop: set MIRA_GAME_AUDIO_DEVICE (find it with `python tools\check_capture.py devices`),
REM  use headphones (so game audio doesn't bleed into your mic), and open TCP 8200 inbound.
REM  Twitch chat needs nothing extra here — the laptop reads Twitch's API directly.
REM
REM  After it starts: open your OBS Browser Source on the avatar port, then say "mira join"
REM  in the Discord VC. Say "mira host"/"mira let me talk" to hand the floor back and forth.
REM ============================================================================
cd /d "%~dp0"

REM EDIT each session: the DESKTOP's current LAN IP (desktop_senses.py prints it; DHCP).
REM This is where Mira pulls the screen frames + transcribed game dialogue from.
set DESKTOP_IP=192.168.12.151

REM --- 1. Local BRAIN + EYES: Qwen2.5-VL-7B on :8080 --------------------------
REM run-mira-vision.ps1 auto-detects this no-D: laptop (C:\models, CUDA 12.8 image), loads the
REM model + mmproj with q8_0 KV (turbo KV shreds image tokens), and polls /health.
echo [1/2] Bringing up the local vision brain (Qwen2.5-VL-7B, Docker)...
powershell -ExecutionPolicy Bypass -File "%~dp0llm-test\run-mira-vision.ps1" -Size 7b -CtxSize 16384

echo       Checking brain health on :8080 ...
curl.exe -s -m 5 http://localhost:8080/health | findstr /i "ok" >nul
if errorlevel 1 (
  echo.
  echo [stream] Vision brain is not healthy on :8080.
  echo   - Is Docker Desktop running?  Try:  docker logs llama-turbo
  echo   - If it OOMed on the 8 GB GPU, try the 3B:  powershell -File llm-test\run-mira-vision.ps1 -Size 3b
  echo.
  pause
  exit /b 1
)
echo       Brain + vision OK.

REM --- 2. Memory embeddings: local Ollama nomic-embed-text -------------------
ollama list | findstr /i "nomic-embed-text" >nul || ollama pull nomic-embed-text

echo [2/2] Starting Mira in STREAM mode (Discord VC + Twitch + host + vision + game audio)...
call "%~dp0.venv\Scripts\activate"

REM --- Everything points at THIS box ----------------------------------------
REM Chat brain AND vision both hit the one local VL server. llama-server serves whatever GGUF
REM is loaded regardless of the model name, so MIRA_MODEL=turbo is fine; vision works via the
REM image_url content + the loaded --mmproj.
set OLLAMA_BASE_URL=http://localhost:8080/v1
set MIRA_VISION_BASE_URL=http://localhost:8080/v1
set MIRA_EMBED_BASE_URL=http://localhost:11434/v1
set MIRA_MODEL=turbo
set MIRA_VISION_MODEL=qwen2.5-vl
set MIRA_NO_THINK=1
REM Match the scribe's note-chunking budget to the VL server's served context (-CtxSize 16384).
set MIRA_NOTES_CTX=16384

REM --- Senses come from the DESKTOP companion (tools\desktop_senses.py on :8200) ---
REM Vision pulls SCREEN FRAMES from the desktop (her local VL still does the captioning), and
REM game audio pulls TRANSCRIBED dialogue lines (the desktop runs that Whisper). Unset either
REM to fall back to LOCAL capture on this laptop.
set MIRA_VISION_FRAME_URL=http://%DESKTOP_IP%:8200/frame
set MIRA_GAME_AUDIO_URL=http://%DESKTOP_IP%:8200/game-audio

REM --- TTS: Piper (+ RVC) — same as start_discord.bat -----------------------
set MIRA_TTS=piper
set MIRA_PIPER_MODEL=C:\models\piper\en_US-hfc_female-medium.onnx
set USE_RVC=1
set MIRA_RVC_MODEL=C:\models\rvc_models\mira.pth
set MIRA_RVC_INDEX=C:\models\rvc_models\mira.index
set MIRA_RVC_PYTHON=%~dp0.venv-rvc\Scripts\python.exe

REM --- STT: Whisper on the CPU (GPU is full with the 7B vision model) --------
REM This laptop Whisper now ONLY handles YOUR mic (your Discord voice) — the game-audio STT
REM runs on the desktop companion. int8 on the CPU is plenty for one mic stream, and the CPU is
REM otherwise idle (LLM is on the GPU). For VL-3B + GPU mic STT: WHISPER_DEVICE=cuda, float16.
set WHISPER_DEVICE=cpu
set WHISPER_MODEL_SIZE=small.en
set WHISPER_COMPUTE_TYPE=int8

REM --- Run Mira: full stream loadout ----------------------------------------
python "%~dp0main.py" --discord --twitch --host --vision --game-audio

endlocal
