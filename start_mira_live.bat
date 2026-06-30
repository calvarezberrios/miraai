@echo off
setlocal
cd /d "%~dp0"
REM ============================================================================
REM  Mira Live (branch: new-model) — web-UI chat on the laptop. ONE CLICK:
REM  starts the local LLM server (Hermes-3 on native llama.cpp) if it isn't
REM  already running, then launches the web UI.
REM
REM  No LM Studio, no Docker/WSL (so no OOM). Open http://localhost:8900 after.
REM  Phase 1 = text chat + sessions + context meter. STT/TTS/avatar/Discord next.
REM ============================================================================

REM --- 1. Ensure the local LLM server (Hermes-3) is up -----------------------
curl.exe -s -m 3 -o nul http://localhost:1234/v1/models
if not errorlevel 1 goto llm_ok
echo [mira_live] Starting the local LLM server (Hermes-3, llama.cpp) in its own window...
start "Mira LLM (Hermes-3)" "%~dp0start_llm_server.bat"
echo [mira_live] Waiting for the model to load (~15-25s)...
set _t=0
:waitllm
timeout /t 3 /nobreak >nul
curl.exe -s -m 3 -o nul http://localhost:1234/v1/models
if not errorlevel 1 goto llm_ok
set /a _t+=1
if %_t% lss 40 goto waitllm
echo [mira_live] WARNING: LLM server not responding yet - check its window (docker-free llama.cpp).
:llm_ok
echo [mira_live] LLM server is up on :1234.

REM --- 2. Web UI ------------------------------------------------------------
call ".\.venv\Scripts\activate"
set MIRA_LLM_BASE_URL=http://localhost:1234/v1
set MIRA_LLM_MODEL=
set MIRA_CONTEXT_LIMIT=8192
set MIRA_WEB_HOST=0.0.0.0
set MIRA_WEB_PORT=8900
python -m mira_live.server
endlocal
