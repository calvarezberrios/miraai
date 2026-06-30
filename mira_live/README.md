# Mira Live (branch: `new-model`)

A from-scratch rebuild: **LM Studio** serves the model, a **web UI** replaces the terminal, and
Mira keeps her personality across Discord + Twitch + a custom avatar stage — all on the laptop.
No Docker/WSL (so no more OOM crashes), no long-term memory yet (session-only).

## Status — phased build

- [x] **Phase 1 — core chat UI:** web chatroom, LM Studio (Hermes-3) streaming, persona,
  sessions sidebar (browse old chats), "New chat", context meter.
- [ ] Phase 2 — voice: TTS (Piper/Kokoro) playback + STT (Whisper) mic input in the UI.
- [ ] Phase 3 — avatar stage: mount the VRM, move/zoom/rotate, lip-sync; OBS browser source by LAN IP.
- [ ] Phase 4 — Discord (VC + text, only when addressed with "Mira"/"@Mira") + Twitch chat (live).
- [ ] Phase 5 — natural autonomy polish (varied solo hosting; best with live chat).

## Model server — already set up (native llama.cpp, no LM Studio)

The LLM runs on **native llama.cpp (CUDA 13)** — straight on Windows, **no Docker/WSL, so no OOM**.
Already installed for you:
- `C:\llama-cpp\llama-server.exe` (+ CUDA 13 DLLs) — llama.cpp build b9851.
- `C:\models\Hermes-3-Llama-3.1-8B-Q4_K_M.gguf` (~4.9 GB).
- `start_llm_server.bat` runs it with: all layers on GPU (`-ngl 99`), **8192 context** (`-c 8192`),
  **flash attention on** (`-fa on`), OpenAI API on `http://localhost:1234/v1`. Uses ~5.6 GB VRAM,
  leaving room for CPU Whisper/Piper later.

You don't run that by hand — `start_mira_live.bat` starts it automatically if it isn't up.

> Want LM Studio's GUI instead someday? Install it, Start Server with Hermes-3 + the same
> settings, and it serves the same `:1234/v1` API — nothing else changes.

## Run

```
start_mira_live.bat
```
This starts the LLM server (in its own window) if needed, waits for the model to load, then opens
the web UI. Go to **http://localhost:8900** (LAN/OBS later: `http://<laptop-ip>:8900`).

- **New chat** (sidebar) starts a fresh context with the persona re-applied.
- The **context meter** under the stage shows tokens used vs. the 8192 limit — when it's near
  full, click **New chat**.
- Old sessions are listed in the sidebar; click one to read it back. They're stored as JSON in
  `mira_live/sessions/` (git-ignored).

## Persona

Edit `mira_live/persona.txt`. It's the system prompt prepended to every turn (kept short to save
tokens). Restart the server to reload.

## Config (env, set in `start_mira_live.bat`)

| Var | Default | Meaning |
|---|---|---|
| `MIRA_LLM_BASE_URL` | `http://localhost:1234/v1` | LM Studio server |
| `MIRA_LLM_MODEL` | _(blank)_ | Pin a model id, or blank = auto-detect the loaded one |
| `MIRA_CONTEXT_LIMIT` | `8192` | Context window (for the meter; match LM Studio) |
| `MIRA_TEMPERATURE` | `0.85` | Higher = more varied |
| `MIRA_WEB_PORT` | `8900` | Web UI port |
