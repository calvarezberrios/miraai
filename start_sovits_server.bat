@echo off
REM ===== GPT-SoVITS TTS server for Mira (desktop GPU) =====
REM Start this BEFORE launching Mira with MIRA_TTS=gptsovits (start_discord.bat).
REM Uses the 'custom' block in tts_infer.yaml: device=cuda, is_half=false (the GTX 1660
REM is fp32-only). Mira clones her voice zero-shot from voice\sample_clip.mp3, so no
REM trained model is needed. First synth warms slowly (~15s); warm sentences run ~0.5x
REM real-time. Leave this window open while Mira runs; Ctrl+C to stop.
set GSV_DIR=D:\GPT-SoVITS-v3lora-20250228
set PYTHONIOENCODING=utf-8
cd /d "%GSV_DIR%"
"%GSV_DIR%\runtime\python.exe" api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS\configs\tts_infer.yaml
