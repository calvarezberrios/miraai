@echo off
setlocal
REM ============================================================================
REM  start_desktop_senses.bat  --  run this on the DESKTOP (the streaming machine).
REM
REM  Mira runs on the LAPTOP, but the game/stream/screen/game-audio are HERE on the desktop.
REM  This companion captures the desktop's screen + game-audio loopback and serves them on
REM  :8200 so laptop-Mira can SEE and HEAR the stream (separate from the Discord VC).
REM    - screen      -> served raw at /frame      (the laptop's VL does the captioning)
REM    - game audio  -> transcribed HERE (its own Whisper) and served at /game-audio
REM  Twitch chat is NOT here — the laptop reads Twitch directly.
REM
REM  PREREQS on this desktop (it already has the repo + venv from when Mira ran here):
REM    - .env has MIRA_GAME_AUDIO_DEVICE set (find it: python tools\check_capture.py devices)
REM    - headphones (so game audio doesn't bleed into your mic)
REM    - open TCP 8200 inbound in the firewall so the laptop can reach it:
REM        New-NetFirewallRule -DisplayName "Mira senses 8200" -Direction Inbound -Protocol TCP -LocalPort 8200 -Action Allow
REM
REM  It prints this desktop's LAN IP on start — put that in DESKTOP_IP in the laptop's
REM  start_stream_servers.bat. Ctrl+C to stop.
REM ============================================================================
cd /d "%~dp0"
call "%~dp0.venv\Scripts\activate"

REM Game-audio STT runs HERE. The desktop GTX 1660 (Turing) has a BROKEN fp16 path, so use
REM int8_float16 (drop to WHISPER_DEVICE=cpu / int8 if the GPU is busy with the game).
set WHISPER_DEVICE=cuda
set WHISPER_MODEL_SIZE=small.en
set WHISPER_COMPUTE_TYPE=int8_float16

REM Screen capture cadence (seconds). 2s keeps Mira's view fresh; it's cheap.
set MIRA_VISION_EVERY=2

python "%~dp0tools\desktop_senses.py"

endlocal
