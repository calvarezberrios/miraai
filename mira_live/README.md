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

## One-time setup

1. **Install LM Studio** → https://lmstudio.ai (native Windows app, no Docker).
2. In LM Studio, **download** `Hermes-3-Llama-3.1-8B` (Q4_K_M) — search the model browser.
3. **Load** it, then open the **Developer** tab → **Start Server**. Set:
   - **GPU Offload:** Max (all layers) — the 8B fits the 8 GB 5050.
   - **Context Length:** 8192 (match `MIRA_CONTEXT_LIMIT` in `start_mira_live.bat`).
   - **Flash Attention:** ON (less VRAM, faster long chats).
   - Server URL should be `http://localhost:1234/v1`.

## Run

```
start_mira_live.bat
```
Open **http://localhost:8900**. On the LAN (e.g. for the desktop/OBS later):
`http://<laptop-ip>:8900`.

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
