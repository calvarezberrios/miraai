@echo off
setlocal
cd /d "%~dp0"
REM ============================================================================
REM  Mira Live (branch: new-model) — web-UI chat on the laptop.
REM
REM  PREREQS:
REM   1. LM Studio installed, with Hermes-3-Llama-3.1-8B (Q4_K_M) downloaded.
REM   2. In LM Studio: load the model, then Developer tab -> Start Server.
REM        - GPU Offload: MAX (all layers)
REM        - Context Length: 8192   (match MIRA_CONTEXT_LIMIT below)
REM        - Flash Attention: ON
REM      The server listens on http://localhost:1234/v1 (OpenAI-compatible).
REM
REM  Then run this. Open http://localhost:8900 in your browser.
REM  (Phase 1 = text chat + sessions + context meter. STT/TTS/avatar/Discord/Twitch next.)
REM ============================================================================
call ".\.venv\Scripts\activate"

REM --- LLM: LM Studio (OpenAI-compatible). Leave MODEL blank to auto-detect the loaded model. ---
set MIRA_LLM_BASE_URL=http://localhost:1234/v1
set MIRA_LLM_MODEL=
set MIRA_CONTEXT_LIMIT=8192

REM --- Web UI (0.0.0.0 so the avatar stage is reachable on the LAN for OBS) ---
set MIRA_WEB_HOST=0.0.0.0
set MIRA_WEB_PORT=8900

python -m mira_live.server
endlocal
