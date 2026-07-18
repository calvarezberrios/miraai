@echo off
setlocal
REM ============================================================================
REM  start_stream_servers.bat — Mira in FULL STREAM mode on the NEW MODEL.
REM  (branch: new-model — Hermes-3 on native llama.cpp, NO vision model)
REM
REM  This laptop runs:
REM    - BRAIN : Hermes-3-Llama-3.1-8B on native llama.cpp (:1234, ~5.6 GB VRAM)
REM    - STT   : Whisper on the laptop GPU (your mic via Discord VC)
REM    - TTS   : Piper (+ optional RVC)
REM    - MEMORY: local Ollama nomic-embed-text (:11434)
REM    - Mira  : Discord VC + Twitch chat + HOST autonomy + GAME AUDIO
REM
REM  VISION IS OFF in this setup (no VL model — it was the OOM culprit and is
REM  retired for now). GAME AUDIO still works: the DESKTOP runs the senses
REM  companion (python tools\desktop_senses.py) which transcribes the game's
REM  dialogue and serves it on :8200 — Mira pulls the text over the LAN.
REM
REM  Personality: the DANDERE persona in mira_live\persona.txt (single source of
REM  truth for both stacks — edit that file to tune her).
REM
REM  PREREQS: llama.cpp + Hermes-3 installed (start_llm_server.bat), Piper voice,
REM  Ollama + nomic-embed-text, .env: DISCORD_BOT_TOKEN, TWITCH_CHANNEL,
REM  TWITCH_CLIENT_ID/SECRET. Desktop: senses companion running + TCP 8200 open;
REM  use headphones so game audio doesn't bleed into your mic.
REM
REM  After it starts: OBS Browser Source on the avatar port, then say "mira join"
REM  in the Discord VC. "mira host"/"mira let me talk" hands the floor around.
REM ============================================================================
cd /d "%~dp0"

REM EDIT each session if it moved: the DESKTOP's LAN IP (desktop_senses.py prints
REM it; DHCP). If it goes stale, game-audio auto-discovers the companion on the
REM LAN after a few failed polls, so a wrong IP self-heals in ~20-30s.
set DESKTOP_IP=192.168.12.151

REM --- 1. Local BRAIN: Hermes-3 on native llama.cpp (:1234) -------------------
curl.exe -s -m 3 -o nul http://localhost:1234/v1/models
if not errorlevel 1 goto llm_ok
echo [stream] Starting the Hermes-3 LLM server (its own window)...
start "Mira LLM (Hermes-3)" "%~dp0start_llm_server.bat"
echo [stream] Waiting for the model to load (~15-25s)...
set _t=0
:waitllm
timeout /t 3 /nobreak >nul
curl.exe -s -m 3 -o nul http://localhost:1234/v1/models
if not errorlevel 1 goto llm_ok
set /a _t+=1
if %_t% lss 40 goto waitllm
echo [stream] LLM server did not come up on :1234 — check its window.
pause
exit /b 1
:llm_ok
echo [stream] Brain OK on :1234.

REM --- 2. Memory embeddings: local Ollama nomic-embed-text --------------------
ollama list | findstr /i "nomic-embed-text" >nul || ollama pull nomic-embed-text

echo [stream] Starting Mira in STREAM mode (Discord VC + Twitch + host + game audio)...
call "%~dp0.venv\Scripts\activate"

REM --- Everything points at THIS box ------------------------------------------
set OLLAMA_BASE_URL=http://localhost:1234/v1
set MIRA_EMBED_BASE_URL=http://localhost:11434/v1
set MIRA_MODEL=hermes-3
set MIRA_NO_THINK=0
REM Match the scribe's note-chunking budget to the served context (-c 8192).
set MIRA_NOTES_CTX=16384

REM --- GAME AUDIO from the DESKTOP companion (NO vision) ----------------------
REM The desktop transcribes the game dialogue and serves text on :8200; Mira
REM polls it. (MIRA_VISION_FRAME_URL is deliberately NOT set and --vision is
REM deliberately NOT passed — the vision model is retired.)
set MIRA_GAME_AUDIO_URL=http://%DESKTOP_IP%:8200/game-audio

REM --- TTS: Piper (+ RVC) ------------------------------------------------------
set MIRA_TTS=piper
set MIRA_PIPER_MODEL=C:\models\piper\en_US-hfc_female-medium.onnx
set USE_RVC=1
set MIRA_RVC_MODEL=C:\models\rvc_models\mira.pth
set MIRA_RVC_INDEX=C:\models\rvc_models\mira.index
set MIRA_RVC_PYTHON=%~dp0.venv-rvc\Scripts\python.exe

REM --- STT: Whisper on the laptop GPU (mic only; game audio is on the desktop) --
REM distil-large-v3 (int8_float16): near large-v3 accuracy at ~small speed — fixes the
REM "Mira" -> "Mia" mishears small.en made (verified: ~0.3s/utterance, 1.25 GB VRAM free
REM next to the 16k Hermes). Name-biased decoding via hotwords (MIRA_STT_HOTWORDS adds more).
set WHISPER_DEVICE=cuda
set WHISPER_MODEL_SIZE=distil-large-v3
set WHISPER_COMPUTE_TYPE=int8_float16

REM --- Run Mira: stream loadout (NO --vision) ----------------------------------
python "%~dp0main.py" --discord --twitch --host --game-audio

endlocal
