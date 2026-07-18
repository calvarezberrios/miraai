@echo off
setlocal
REM ============================================================================
REM  Local LLM server for Mira Live — Hermes-3 on NATIVE llama.cpp (CUDA 13).
REM
REM  Serves an OpenAI-compatible API on http://localhost:1234/v1 for mira_live.
REM  Runs straight on Windows (no Docker/WSL) -> none of the WSL OOM crashes.
REM  Settings: all layers on GPU (-ngl 99), 8192 context, flash attention ON.
REM
REM  Leave this window open while you use Mira. Ctrl+C to stop.
REM  (start_mira_live.bat launches this automatically if it isn't already running.)
REM ============================================================================
set LLAMA=C:\llama-cpp\llama-server.exe
set MODEL=C:\models\Hermes-3-Llama-3.1-8B-Q4_K_M.gguf

if not exist "%LLAMA%" (
  echo llama-server not found: %LLAMA%
  pause & exit /b 1
)
if not exist "%MODEL%" (
  echo Model not found: %MODEL%
  pause & exit /b 1
)

REM 16384 ctx (a long session overflowed 8192) with q8_0 KV cache: the quantized KV
REM keeps 16k at the same VRAM as 8k-f16 was (~5.6 GB) — q8 KV is fine for TEXT models
REM (it only broke vision image tokens, and there's no vision model here).
echo Starting Hermes-3 on llama.cpp (CUDA, all layers on GPU, 16384 ctx, flash attention)...
"%LLAMA%" -m "%MODEL%" --host 0.0.0.0 --port 1234 -ngl 99 -c 16384 -fa on ^
  --cache-type-k q8_0 --cache-type-v q8_0 --threads-http 4
endlocal
