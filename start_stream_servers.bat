@echo off
REM ============================================================================
REM  start_stream_servers.bat  --  bring Mira up in STREAM mode on the DESKTOP.
REM
REM  Starts the GPT-SoVITS TTS server (its own window), waits for it to load, then
REM  launches Mira for a live stream:
REM      Discord VC (you talk to her; her voice -> VC -> OBS) + Twitch chat,
REM      with --host autonomy, --vision (screen), and --game-audio (she hears the game).
REM
REM  PREREQS (see VISION_SETUP.md / README):
REM   - LAPTOP serving Qwen2.5-VL:  llm-test\run-mira-vision.ps1   (brain AND eyes, :8080)
REM   - local Ollama running (nomic-embed-text) for memory embeddings
REM   - .env filled in: DISCORD_BOT_TOKEN, TWITCH_CHANNEL, TWITCH_CLIENT_ID/SECRET,
REM     MIRA_VISION_MODEL, MIRA_GAME_AUDIO_DEVICE=CABLE Output
REM   - headphones (so game audio doesn't bleed into your mic)
REM
REM  After it starts: open your OBS Browser Source on the avatar port, then say
REM  "mira join" in the Discord VC so her voice reaches the call. Ctrl+C / close
REM  windows to stop (or use stop_servers.bat for the SoVITS window).
REM ============================================================================
setlocal
cd /d "%~dp0"

REM EDIT each session: the LAPTOP's current LAN IP (run-mira-vision.ps1 prints it; DHCP).
set BRAIN_IP=192.168.12.116

echo [1/2] Starting GPT-SoVITS server (its own window)...
start "Mira SoVITS Server" "%~dp0start_sovits_server.bat"

echo       Waiting for GPT-SoVITS to load models (~30-60s on the GTX 1660)...
set /a _tries=0
:waitloop
REM curl exits 0 as soon as the server answers ANY HTTP response = port bound = models loaded.
curl.exe -s -m 3 -o nul "http://127.0.0.1:9880/tts?text=x&text_lang=en&ref_audio_path=x&prompt_lang=en" 2>nul
if not errorlevel 1 goto ready
set /a _tries+=1
if %_tries% geq 40 (
    echo       WARNING: GPT-SoVITS still not responding after ~120s. Check its window.
    echo       Starting Mira anyway -- if her voice fails, fix the server and restart.
    goto launch
)
timeout /t 3 /nobreak >nul
goto waitloop

:ready
echo       GPT-SoVITS is up.

:launch
echo [2/2] Starting Mira in STREAM mode (Discord VC + Twitch + host + vision + game audio)...
call .\.venv\Scripts\activate

REM Chat brain AND eyes -> the LAPTOP's llama.cpp server, which in stream mode runs Qwen2.5-VL
REM (run-mira-vision.ps1) as both. Both endpoints follow the single BRAIN_IP above.
REM The MODEL names (MIRA_MODEL / MIRA_VISION_MODEL = the GGUF the server serves) come from
REM .env, so there's one place to keep them in sync with /v1/models.
set OLLAMA_BASE_URL=http://%BRAIN_IP%:8080/v1
set MIRA_VISION_BASE_URL=http://%BRAIN_IP%:8080/v1

REM Memory embeddings -> THIS desktop's own Ollama nomic-embed-text (:11434), NOT the laptop.
set MIRA_EMBED_BASE_URL=http://localhost:11434/v1
set MIRA_NO_THINK=1
REM Match the scribe's note-chunking budget to the VL server's served context (run-mira-vision.ps1 -c 16384).
set MIRA_NOTES_CTX=16384

REM Voice: GPT-SoVITS (started above). Fall back any time with: set MIRA_TTS=kokoro
set MIRA_TTS=gptsovits

REM STT on the desktop GPU. GTX 1660 / Turing has a BROKEN fp16 path, so use int8_float16.
set WHISPER_DEVICE=cuda
set WHISPER_MODEL_SIZE=small.en
set WHISPER_COMPUTE_TYPE=int8_float16

python main.py --discord --twitch --host --vision --game-audio

endlocal
