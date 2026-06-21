@echo off
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
REM laptop has a free 8GB NVIDIA GPU now (no local LLM) -> Whisper on CUDA, fast STT
set WHISPER_DEVICE=cuda
set WHISPER_MODEL_SIZE=small
python main.py
