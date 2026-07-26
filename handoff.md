# Mira — Handoff (branch `new-model`)

Last commit at handoff: **`98370e6`** (D&D Beyond import + action economy). Working tree clean.
This doc is the from-scratch rebuild on the **`new-model`** branch. The old `main`/`laptop_run`
architecture (Docker/Qwen3 brain, senses bridge, etc.) is parked — see the bottom section only if
you need it.

Repo: `C:\Users\manni\Downloads\aiproject` · GitHub `calvarezberrios/miraai` · Windows 11 laptop,
RTX 5050 (Blackwell, 8 GB), Python 3.11 venv at `.venv`. Bash + PowerShell both available.

---

## 0. TL;DR of what she is now

**Mira** = a shy, human, dandere anime-girl VTuber. Runs entirely on the laptop:
- **Brain:** Hermes-3-Llama-3.1-8B (Q4_K_M) on **native llama.cpp (CUDA 13)** — no Docker/WSL, no OOM.
- **Voice:** Piper (+ optional RVC). **Ears:** Whisper `distil-large-v3` int8_float16 on GPU.
- **Two front-ends share one brain + persona:**
  1. **The full Mira** (`main.py`) — Discord voice/text + Twitch + avatar + memory + subconscious +
     note-taking + games (D&D). Launched by `start_discord.bat` / `start_stream_servers.bat`.
  2. **`mira_live/`** — a new web-UI chat (FastAPI) replacing the terminal. Launched by
     `start_mira_live.bat`. Session-only memory, sessions sidebar, context meter.
- **Persona is ONE file:** `mira_live/persona.txt`, loaded by BOTH stacks (prefrontal_cortex reads it).

---

## 1. How to run it

**LLM server (shared by everything)** — `start_llm_server.bat` runs
`C:\llama-cpp\llama-server.exe` with `C:\models\Hermes-3-Llama-3.1-8B-Q4_K_M.gguf`,
`-ngl 99 -c 16384 -fa on --cache-type-k/v q8_0`, OpenAI API on `:1234`. ~5.7 GB VRAM.
The `start_*.bat` files auto-start it in its own window if `:1234` is down.

- **Discord + host:** `start_discord.bat`
- **Full stream (Discord VC + Twitch + host + game audio, NO vision):** `start_stream_servers.bat`
  (edit `DESKTOP_IP` for the game-audio senses companion; auto-discovers if it moves)
- **Web UI:** `start_mira_live.bat` → http://localhost:8900 (LAN-reachable for OBS)
- **Avatar stage:** served by motor_cortex on `:8234` (VRM `avatar/shiori.vrm`, loaded by default;
  `?model=/mira.vrm` to switch). OBS browser source points here.

**⚠ CRITICAL GOTCHAS when testing:**
- **Single-instance lock:** `main.py` binds `127.0.0.1:8235` for its lifetime. Before killing any
  `llama-server` process, run: `python -c "import socket;s=socket.socket();s.bind(('127.0.0.1',8235))"`
  — if it raises OSError, **Mira is live**; killing the LLM breaks her session. (I did this twice.)
- **Stale servers stack:** the LLM window OUTLIVES Mira's console. `Get-Process llama-server` and
  kill leftovers if VRAM looks short (two servers = 8 GB exhausted).
- **Console screenshots time out** on the WebGL avatar / claude.ai. Use headless Edge for shots:
  `& msedge --headless --disable-gpu --screenshot=out.png --window-size=1280,720 <url>`.
- **Emoji to the Windows console** breaks Python prints — set `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`.

---

## 2. Persona & personality (all in `mira_live/persona.txt`)

Dandere archetype done right: **safety-dependent switch** — near-silent/deflecting with strangers
or an audience (silence = shyness, never dislike) ↔ warm/talkative/initiating when ALONE with
someone whose trust is EARNED (staged: few words → dry jokes → real warmth; partly losable). Hard
rules kept: **no end-questions**, strictly PG/ToS, ≤3 sentences, one `*wrapped action*` max.

**Output filters (`prefrontal_cortex.py` + `mira_live/clean.py` + `main.py`):**
- **No-questions:** `_drop_trailing_questions` / `_no_trailing_question` strip trailing question
  sentences (mid-reply Qs and whole-reply Qs survive). Trailing `*actions*` are transparent (silent).
- **Actions:** she may emit ONE `*wrapped action*` (shown as body language, drives avatar gestures,
  **TTS skips it**). Bare (asterisk-less) beats — including comma'd ("blinks in confusion, then...")
  and speech-fused ("...to herself To have...") — are auto-WRAPPED via `_split_bare_action` (uses the
  tell: beats are lowercase, speech resumes at a Capital). THREE word-length caps exist (prefrontal
  regex, clean.py regex, `main.py _is_tts_action_or_gesture`) — keep them consistent if you touch this.
- **Hosting stays dandere:** the `--host` directive in `main.py describe_situation` + `host_patter`
  angles were rewritten so hosting = "willing to speak unaddressed, quietly, as herself" (was turning
  her bubbly/attention-seeking). LESSON: per-turn situation directives can override the persona — any
  injected instruction must say "stay in character".

---

## 3. D&D player mode (the big recent feature)

Files: `brain/forebrain/cerebrum/frontal_lobe/games/` → `dnd_engine.py` (TRUTH: dice + sheet),
`dnd_player.py` (the mode/commands), `dnd_beyond.py` (DDB import). Wired in `main.py`
`handle_message` (intercepted BEFORE the scribe; situation note added; scrub applied to replies).
Design mirrors the existing MTG `game_master.py`/`deep_iq_engine.py`: **engine owns every number,
LLM only narrates.** Characters saved per-name under `games/dnd_characters/` (gitignored).

**Commands (all name-addressed "mira ..." from voice/text/local; Twitch/game-audio blocked):**
- Mode: `mira let's play dnd` / `join the party` (enters mode, **no auto-create**), `mira leave the party`.
- Create (step-by-step): `mira create a character` → `mira be an elf wizard` / `name yourself Lyra` /
  `level 3` / `roll your stats` / `standard array` / `set int 16` → `mira finish the character`.
- Load: `mira use character <name>`, `mira list characters`.
- **D&D Beyond:** `mira use your dndbeyond character <url>`, `mira sync your character`.
- Rolls (natural speech is SEARCHED, not just exact commands): `mira roll perception`, `roll a dex
  save`, `roll initiative`, `roll 2d6+3`, `attack` / `attack with my shortsword` / `shoot the goblin`,
  `cast magic missile`, `take 5 damage`, `heal 3`, `long rest`, `show your sheet`. (+"with advantage".)
- **Turn/action economy:** `mira start your turn`, `what can you do` / `options`, `move 20 feet`,
  `use your reaction to ...`, `dash`/`dodge`/`disengage`, `end your turn`. `turn_options()` lists what's
  available per action/bonus/reaction/movement; over-spend is flagged; bonus-action spells (healing
  word) don't burn the action.

**Dice discipline (this was the user's main complaint — she was inventing rolls):** fixed 3 ways —
(1) natural-speech roll intent routes to the engine, (2) situation note forbids stating any number,
(3) `scrub_stream`/`scrub_text` strip fabricated rolls from normal replies (engine narration is a
separate path and keeps real numbers). Narration rules: STATE the total, never adjudicate hit/miss
(DM's call). The `[dice] ...` line is always posted to the table via `notify` regardless.

**⚠ PENDING / KNOWN LIMITS:**
- **DDB importer is MOCK-tested only, NOT against a real character yet.** Ask the user to set a
  character Public and run `mira use your dndbeyond character <url>`, then compare the printed sheet
  to DDB. Expect tuning on **AC** (armor edge cases), **weapon proficiency** (currently assumes prof),
  and **spell slots**. GOTCHA already found: DDB modifier subTypes use FULL ability names
  (`dexterity-score`, not `dex-score`). Endpoint: `https://character-service.dndbeyond.com/character/v5/character/{id}`.
- Engine tracks **only 1st-level spell slots** (5e-lite) — higher slots approximated.
- **DM mode** not built (assessed: engine-backed referee feasible, human-quality story not on an 8B).
- Multiattack, concentration, conditions, opportunity attacks vs others not modeled.

---

## 4. Other systems (state)

- **STT:** `wernickes_area.py` — `distil-large-v3` int8_float16 on GPU, + **hotword biasing** to her
  name (`MIRA_STT_HOTWORDS`, defaults to `action_selector.NAME`). Fixed "Mira"→"Mia" mishears. CUDA
  DLLs come from `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` wheels installed in `.venv`.
- **Name:** configurable via `MIRA_NAME`/`MIRA_NAME_ALIASES` (`action_selector.py`). Currently **Mira**
  (was briefly Shiori; "shiori" kept as alias). Commands + display all follow it.
- **Memory:** ChromaDB at `memory_store/` (nomic-embed-text via Ollama). Was **wiped** for a fresh
  awakening (she starts as a stranger). Recall hardened to never kill a turn. If "hnsw segment reader:
  Nothing found on disk" recurs: salvage+rebuild recipe is in memory `new-model-rebuild.md`.
- **Note-taking** (`dorsolateral_prefrontal_cortex.py`): start/stop are **text-only** (voice can't
  trigger); transcript echoes to terminal (`[notes HH:MM:SS] Speaker: text`); empty session warns.
- **Discord `[heard]` logs** show source: `[heard #general @ ServerName] Name: ...`. NOTE: she reads
  EVERY text channel in EVERY server the bot is in (`ONLY_CHANNELS` empty) — a channel/guild allowlist
  was offered but NOT built. Watch for cross-server context bleed.
- **Web UI (`mira_live/`):** sessions created lazily on first message (no empty-chat spam), sidebar
  with delete (confirm), per-chat context meter (16384), New chat. STT/TTS/avatar in the web UI are
  NOT built yet (Phase 2/3 — it's text-only so far).
- **Avatar:** her VRoid model `avatar/shiori.vrm` renders on the stage. Move/zoom/rotate + OBS by IP
  work via the existing avatar pipeline.

---

## 5. Roadmap / likely next asks

- **Verify + tune DDB import against the user's real character** (highest priority — it's untested live).
- **mira_live Phase 2+:** wire Piper/Kokoro TTS playback + Whisper STT mic into the web UI; then the
  avatar stage (mount VRM, move/zoom/rotate), then Discord/Twitch from the web app.
- Autonomy polish (solo hosting is thin without live chat — inherent to a small model).
- Optional: channel/guild allowlist for Discord; higher-level spell slots; more D&D combat rules.

---

## 6. Working style notes (from this run)

- User tests live and reports specific transcripts — reproduce the exact failing case, fix, re-verify
  live on Hermes, then commit. Commit per logical change with detailed messages.
- Verify against reality (nvidia-smi, real endpoints, seeded dice) — don't assert "should work".
- Persona/voice tuning is the user's domain; propose, don't unilaterally rewrite the persona.
- Memory lives at `~/.claude/projects/.../memory/` — `new-model-rebuild.md` has the detailed
  chronological log with every commit hash and gotcha; read it alongside this.

---

## 7. Old architecture (parked, `laptop_run`/`main` branches) — only if needed

`LAPTOP_SETUP.md`, `DESKTOP_SENSES_SETUP.md` describe the previous all-local Docker/Qwen3 setup with
the desktop "senses bridge" (vision frames + game-audio over LAN). Vision was retired (OOM). The
`tools/desktop_senses.py` companion is still used ONLY for **game audio** in stream mode now.
