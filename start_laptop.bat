@echo off
REM ===== DEPRECATED (2026-06-27): topology REVERSED. =====
REM The LAPTOP now runs the BRAIN (turbo LLM) via ./llm-test/run-mira-small.ps1, and MIRA
REM runs on the DESKTOP via start_discord.bat. This script is the OLD path (Mira on the
REM laptop, brain on the desktop) — kept only in case you flip back. If you do, set PC_IP
REM to whichever machine is running the brain server.
REM
REM ===== Mira on the LAPTOP — brain runs on the desktop PC over the LAN =====
REM EDIT PC_IP to your desktop's current LAN IP. run-turbo-server.ps1 on the PC prints it.
set PC_IP=192.168.12.151

call ./.venv/Scripts/activate
REM chat brain -> PC's llama.cpp turbo server (:8080)
set OLLAMA_BASE_URL=http://%PC_IP%:8080/v1
REM memory embeddings -> PC's Ollama nomic-embed-text (:11434), NOT the turbo server
set MIRA_EMBED_BASE_URL=http://%PC_IP%:11434/v1
set MIRA_MODEL=turbo
set MIRA_NO_THINK=1
REM laptop has a free 8GB NVIDIA GPU now (no local LLM) -> Whisper on CUDA, fast STT.
REM float16 is faster + more accurate than int8 on a modern GPU (int8 was an old-PC
REM workaround). small.en is the English-only model: faster and more accurate than "small"
REM for English. For even faster STT use base.en (a bit less accurate); if fp16 errors on
REM your GPU, set WHISPER_COMPUTE_TYPE=int8_float16.
set WHISPER_DEVICE=cuda
set WHISPER_MODEL_SIZE=small.en
set WHISPER_COMPUTE_TYPE=float16
python main.py
