@echo off
REM ===== Mira in DISCORD mode on the LAPTOP — brain on the desktop PC over the LAN =====
REM One-time prereqs (see README 4c / LAPTOP_SETUP.md):
REM   - py-cord voice build:  python -m pip install -U "py-cord[voice] @ git+https://github.com/Pycord-Development/pycord.git@820460aa4"
REM   - ffmpeg on PATH:       winget install Gyan.FFmpeg   (reopen terminal)
REM   - .env has DISCORD_BOT_TOKEN=...   (copy it from the desktop's .env; same bot)
REM EDIT PC_IP to your desktop's current LAN IP.
set PC_IP=192.168.12.151

call ./.venv/Scripts/activate
set OLLAMA_BASE_URL=http://%PC_IP%:8080/v1
set MIRA_EMBED_BASE_URL=http://%PC_IP%:11434/v1
set MIRA_MODEL=turbo
set MIRA_NO_THINK=1
REM distil-large-v3 + float16 on the RTX 5050: ~large-v3 accuracy at ~340ms/utterance, English-only.
REM For lower latency drop to small.en or base.en; if fp16 ever errors, use int8_float16.
set WHISPER_DEVICE=cuda
set WHISPER_MODEL_SIZE=distil-large-v3
set WHISPER_COMPUTE_TYPE=float16
python main.py --discord --draft
