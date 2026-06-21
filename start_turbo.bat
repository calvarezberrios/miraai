@echo off
REM Run Mira on the Qwen3.6-35B turbo model (Docker llama-server on :8080) with voice.
REM Prereq: start the brain server first ->  powershell -File llm-test\run-turbo-mira.ps1
call ./.venv/Scripts/activate
set OLLAMA_BASE_URL=http://localhost:8080/v1
set MIRA_MODEL=turbo
set MIRA_NO_THINK=1
set WHISPER_DEVICE=cpu
REM RAM is tight (the 35B fills most of it) -> use a smaller Whisper on CPU. If STT
REM accuracy suffers, bump to "small". Close Discord/extra browser tabs before running.
set WHISPER_MODEL_SIZE=base.en
python main.py
