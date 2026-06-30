@echo off
setlocal
REM ============================================================================
REM  Mira — ALL-LOCAL on THIS laptop  (branch: laptop_run)
REM  One click brings up EVERYTHING on the RTX 5050 (8 GB):
REM    - BRAIN : Qwen3-4B turbo (llama.cpp in Docker) on :8080
REM    - STT   : Whisper on the laptop GPU
REM    - TTS   : Piper (+ optional RVC voice conversion to Mira's timbre)
REM    - MEMORY: local Ollama nomic-embed-text on :11434
REM  Then runs Mira in DISCORD mode with HOST autonomy (she fills lulls / can host).
REM
REM  Why 4B (not 8B): the brain and Whisper share the one 8 GB GPU. The 8B turbo alone
REM  eats ~7.5 GB and leaves no room for GPU STT. The 4B (~3.5 GB w/ 16k ctx) keeps the
REM  same Qwen3 turbo pipeline + prompt tuning while leaving ~2-3 GB for Whisper. Smaller
REM  model = a little less personality depth — the documented tradeoff for one-box running.
REM
REM  ONE-TIME PREREQS (see LAPTOP_SETUP.md "All-on-the-laptop"):
REM    - Docker Desktop running (WSL2 + GPU passthrough), turboquant binary built once
REM    - 4B model:   powershell -File llm-test\download-small.ps1 -Size 4b -ModelDir C:\models
REM    - Piper voice: C:\models\piper\en_US-hfc_female-medium.onnx (+ .json)
REM    - Ollama with nomic-embed-text  (memory embeddings)
REM    - py-cord voice build + ffmpeg on PATH, and .env has DISCORD_BOT_TOKEN=...
REM    - (optional) Mira's RVC voice: drop mira.pth/mira.index in C:\models\rvc_models and
REM      build the .venv-rvc runtime — see "TTS" block below. Without it she uses raw Piper.
REM ============================================================================

REM --- 1. Local BRAIN: Qwen3-4B turbo on :8080 --------------------------------
REM run-mira-small.ps1 auto-detects this no-D: laptop -> C:\models, C:\llama-cpp-turboquant,
REM CUDA 12.8 image. It launches the container detached and polls /health before returning.
echo [start_discord] Bringing up the local 4B brain (Docker)...
powershell -ExecutionPolicy Bypass -File "%~dp0llm-test\run-mira-small.ps1" -Size 4b -CtxSize 16384

REM Confirm the brain is actually answering before we hand the mic to Mira.
echo [start_discord] Checking brain health on :8080 ...
curl.exe -s -m 5 http://localhost:8080/health | findstr /i "ok" >nul
if errorlevel 1 (
  echo.
  echo [start_discord] Brain server is not healthy on :8080.
  echo   - Is Docker Desktop running?  Try:  docker logs llama-turbo
  echo   - If it OOMed, lower ctx:  powershell -File llm-test\run-mira-small.ps1 -Size 4b -CtxSize 8192
  echo.
  pause
  exit /b 1
)
echo [start_discord] Brain OK.

REM --- 2. Memory embeddings: local Ollama nomic-embed-text --------------------
ollama list | findstr /i "nomic-embed-text" >nul || ollama pull nomic-embed-text

call "%~dp0.venv\Scripts\activate"

REM --- 3. Everything points at THIS box --------------------------------------
set OLLAMA_BASE_URL=http://localhost:8080/v1
set MIRA_EMBED_BASE_URL=http://localhost:11434/v1
set MIRA_MODEL=turbo
set MIRA_NO_THINK=1
REM Keep the scribe's note-chunking budget in sync with the served context (-CtxSize above).
set MIRA_NOTES_CTX=16384

REM --- TTS: Piper (+ RVC) — the pre-GPT-SoVITS local voice -------------------
REM Piper alone gives a clean English-female voice with ~no GPU cost. RVC then converts it
REM to Mira's cloned timbre IF her model is present: drop mira.pth/mira.index in
REM C:\models\rvc_models and create .venv-rvc (py -3.11 -m venv .venv-rvc &&
REM .venv-rvc\Scripts\pip install rvc-python). Until then this falls back to raw Piper
REM automatically (no error) — set USE_RVC=0 to silence the RVC path entirely.
set MIRA_TTS=piper
set MIRA_PIPER_MODEL=C:\models\piper\en_US-hfc_female-medium.onnx
set USE_RVC=1
set MIRA_RVC_MODEL=C:\models\rvc_models\mira.pth
set MIRA_RVC_INDEX=C:\models\rvc_models\mira.index
set MIRA_RVC_PYTHON=%~dp0.venv-rvc\Scripts\python.exe

REM --- STT: Whisper on the laptop GPU ----------------------------------------
REM RTX 5050 (Blackwell) has a WORKING fp16 path, so float16 small.en is fast + accurate.
REM If you ever OOM against the brain, drop to int8_float16, or set WHISPER_DEVICE=cpu.
set WHISPER_DEVICE=cuda
set WHISPER_MODEL_SIZE=small.en
set WHISPER_COMPUTE_TYPE=float16

REM --- 4. Run Mira: Discord + HOST autonomy ----------------------------------
REM --host turns on the subconscious AND makes her fill lulls / host. Live toggles still work:
REM say (or type) "mira host" / "mira take over" to hand her the floor, "mira let me talk" /
REM "mira quiet" to take it back. Works in plain Discord now (no Twitch/stream needed).
python "%~dp0main.py" --discord --host

endlocal
