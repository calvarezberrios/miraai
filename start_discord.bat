@echo off
REM ===== Mira in DISCORD mode on the DESKTOP — brain (turbo LLM) on the LAPTOP over the LAN =====
REM Topology FLIPPED (2026-06-27): the turbo llama.cpp model now runs on the LAPTOP
REM (./llm-test/run-mira-small.ps1 on :8080); Mira (STT + TTS + avatar + Discord) runs HERE
REM on the desktop. This is the reverse of the old LAPTOP_SETUP.md layout.
REM One-time prereqs (see README 4c / LAPTOP_SETUP.md):
REM   - py-cord voice build:  python -m pip install -U "py-cord[voice] @ git+https://github.com/Pycord-Development/pycord.git@820460aa4"
REM   - ffmpeg on PATH:       winget install Gyan.FFmpeg   (reopen terminal)
REM   - .env has DISCORD_BOT_TOKEN=...   (same bot)
REM   - local Ollama running with nomic-embed-text  (for long-term memory embeddings)
REM EDIT BRAIN_IP to the LAPTOP's current LAN IP (run-mira-small.ps1 prints it each session).
set BRAIN_IP=192.168.12.116

call ./.venv/Scripts/activate
REM chat brain -> LAPTOP's llama.cpp turbo server (:8080)
set OLLAMA_BASE_URL=http://%BRAIN_IP%:8080/v1
REM memory embeddings -> THIS desktop's own Ollama nomic-embed-text (:11434), NOT the laptop
set MIRA_EMBED_BASE_URL=http://localhost:11434/v1
set MIRA_MODEL=turbo
set MIRA_NO_THINK=1
REM STT on the desktop GPU. GTX 1660 / Turing has a BROKEN fp16 path (-> NaN), so use
REM int8_float16. If that still errors, drop to WHISPER_COMPUTE_TYPE=int8. Only switch to
REM distil-large-v3 + float16 on a GPU with a working fp16 path (e.g. the RTX 5050 laptop).
set WHISPER_DEVICE=cuda
set WHISPER_MODEL_SIZE=small.en
set WHISPER_COMPUTE_TYPE=int8_float16
python main.py --discord --draft
