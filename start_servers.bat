@echo off
REM ============================================================================
REM  start_servers.bat  --  bring Mira fully up on the DESKTOP.
REM
REM  Order matters: the GPT-SoVITS TTS server must be loaded BEFORE Mira starts
REM  (her voice warmup calls it). This launches SoVITS in its own window, waits
REM  for it to finish loading the models, then starts Mira on Discord here.
REM
REM  The turbo LLM runs on the LAPTOP -- make sure run-mira-small.ps1 is up there
REM  first (start_discord.bat points at the laptop's BRAIN_IP). Ollama (embeddings)
REM  must also be running locally.
REM
REM  Stop everything cleanly with stop_servers.bat.
REM ============================================================================
setlocal

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
echo [2/2] Starting Mira (Discord) in this window...
call "%~dp0start_discord.bat"

endlocal
