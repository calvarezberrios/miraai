@echo off
setlocal
REM ============================================================================
REM  Mira — ALL-LOCAL on THIS laptop, on the NEW MODEL (branch: new-model)
REM  One click brings up EVERYTHING:
REM    - BRAIN : Hermes-3-Llama-3.1-8B on NATIVE llama.cpp (CUDA 13) on :1234
REM              (no Docker/WSL -> none of the old OOM crashes; ~5.6 GB VRAM)
REM    - STT   : Whisper on the laptop GPU (fits alongside: ~2.5 GB free)
REM    - TTS   : Piper (+ optional RVC voice conversion to Mira's timbre)
REM    - MEMORY: local Ollama nomic-embed-text on :11434
REM  Then runs Mira in DISCORD mode with HOST autonomy.
REM
REM  Personality: the DANDERE persona in mira_live\persona.txt (single source of
REM  truth — prefrontal_cortex loads that file; edit it to tune her).
REM
REM  ONE-TIME PREREQS:
REM    - llama.cpp + model installed (already done): C:\llama-cpp\llama-server.exe
REM      + C:\models\Hermes-3-Llama-3.1-8B-Q4_K_M.gguf  (see start_llm_server.bat)
REM    - Piper voice: C:\models\piper\en_US-hfc_female-medium.onnx (+ .json)
REM    - Ollama with nomic-embed-text  (memory embeddings)
REM    - py-cord voice build + ffmpeg on PATH, and .env has DISCORD_BOT_TOKEN=...
REM    - (optional) RVC voice: mira.pth/mira.index in C:\models\rvc_models + .venv-rvc
REM ============================================================================
cd /d "%~dp0"

REM --- 1. Local BRAIN: Hermes-3 on native llama.cpp (:1234) -------------------
curl.exe -s -m 3 -o nul http://localhost:1234/v1/models
if not errorlevel 1 goto llm_ok
echo [start_discord] Starting the Hermes-3 LLM server (its own window)...
start "Mira LLM (Hermes-3)" "%~dp0start_llm_server.bat"
echo [start_discord] Waiting for the model to load (~15-25s)...
set _t=0
:waitllm
timeout /t 3 /nobreak >nul
curl.exe -s -m 3 -o nul http://localhost:1234/v1/models
if not errorlevel 1 goto llm_ok
set /a _t+=1
if %_t% lss 40 goto waitllm
echo [start_discord] LLM server did not come up on :1234 — check its window.
pause
exit /b 1
:llm_ok
echo [start_discord] Brain OK on :1234.

REM --- 2. Memory embeddings: local Ollama nomic-embed-text --------------------
ollama list | findstr /i "nomic-embed-text" >nul || ollama pull nomic-embed-text

call "%~dp0.venv\Scripts\activate"

REM --- 3. Everything points at THIS box ---------------------------------------
set OLLAMA_BASE_URL=http://localhost:1234/v1
set MIRA_EMBED_BASE_URL=http://localhost:11434/v1
REM llama.cpp serves whatever GGUF is loaded regardless of the model name.
set MIRA_MODEL=hermes-3
REM Hermes-3 is NOT a reasoning model — no think-block suppression needed.
set MIRA_NO_THINK=0
REM Keep the scribe's note-chunking budget in sync with the served context
REM (start_llm_server.bat -c 8192).
set MIRA_NOTES_CTX=16384

REM --- TTS: Piper (+ RVC) -----------------------------------------------------
set MIRA_TTS=piper
set MIRA_PIPER_MODEL=C:\models\piper\en_US-hfc_female-medium.onnx
set USE_RVC=1
set MIRA_RVC_MODEL=C:\models\rvc_models\mira.pth
set MIRA_RVC_INDEX=C:\models\rvc_models\mira.index
set MIRA_RVC_PYTHON=%~dp0.venv-rvc\Scripts\python.exe

REM --- STT: Whisper on the laptop GPU ------------------------------------------
REM distil-large-v3 (int8_float16): near large-v3 accuracy at ~small speed — fixes the
REM "Mira" -> "Mia" mishears small.en made (verified on this GPU: ~0.3s/utterance,
REM 1.25 GB VRAM free next to the 16k Hermes). Decoding is also biased toward her name
REM (hotwords; add more via MIRA_STT_HOTWORDS). fp16 fits but leaves <300 MB free — don't.
set WHISPER_DEVICE=cuda
set WHISPER_MODEL_SIZE=distil-large-v3
set WHISPER_COMPUTE_TYPE=int8_float16

REM --- 4. Run Mira: Discord + HOST autonomy ------------------------------------
REM Live toggles: "mira host"/"mira take over" hands her the floor,
REM "mira let me talk"/"mira quiet" takes it back.
python "%~dp0main.py" --discord --host

endlocal
