@echo off
REM Run Mira on Groq cloud (fast, free for dev). Put your key in .env: GROQ_API_KEY=gsk_...
REM Get a free key at https://console.groq.com/keys
REM The LLM runs in the cloud, so your local GPU/CPU is free -> Whisper runs on the GPU
REM (fast STT, no CPU fight) and there's no turbo container to babysit.
call ./.venv/Scripts/activate
set MIRA_LLM_PROVIDER=groq
set MIRA_GROQ_MODEL=llama-3.3-70b-versatile
python main.py
