# Mira AI VTuber — Project Handoff

## Purpose of this doc
Primer for continuing the build in a new chat session. Paste this in (along with the
human brain reference + build plan docs) to get back up to speed.

---

## Project overview
Building an AI VTuber named **Mira**, architected around the human brain. Each brain
region maps to a software module. Working through a phased build plan.

## Key decisions / constraints
- **Fully local, no cloud LLMs.** Running models via **Ollama**.
- The code uses the `openai` Python package against Ollama's OpenAI-compatible
  endpoint, so swapping providers later is just a base_url/model change. Same client is
  reused for chat and embeddings.
- **Chat model:** `qwen2.5:3b` (changed from `llama3.2:3b`). On the 6GB GPU running
  STT + LLM + TTS together, a 3B is the sweet spot; qwen2.5:3b follows instructions /
  grounding rules and invents far less than llama3.2:3b while still fitting alongside
  Whisper and GPT-SoVITS. Set in ONE place: the `MODEL` constant in `prefrontal_cortex.py`.
  Bigger (`qwen2.5:7b` / `llama3.1:8b`) grounds even better but crowds the GPU.
- **Embedding model:** `nomic-embed-text` (local, for long-term memory).
- **TTS engine:** **GPT-SoVITS** (local voice cloning) — young-adult-female anime timbre,
  cloned zero-shot from a short reference clip.
- **STT engine:** **Whisper** via **faster-whisper** (accent-robust).
- **OS:** Windows (PowerShell). Ollama at `D:\Ollama\`, project at `D:\aiproject\`.
- **GPU: NVIDIA GTX 1660 Super, 6GB VRAM, FULL-PRECISION ONLY.** Turing TU116 has a broken
  fp16 path (half precision → NaN/silent). HARDWARE, not config; do not chase fp16 fixes.
  GPT-SoVITS must run fp32 (`is_half: false`). VRAM is the day-to-day squeeze. A future GPU
  with working fp16 + 12GB+ is the only real speedup.
- Ollama server must be running before the app works.
- The GPT-SoVITS API server must ALSO be running before voice works.

## NEW: Mira now runs in two modes
- `python main.py`            → **local** mode (mic + speakers; unchanged, rock-solid).
- `python main.py --discord`  → **Discord** mode (bot in text channels + voice channels).
- Modes are mutually exclusive, chosen at launch. Same brain/persona/memory in both.
- Local voice is the **reliable** real-time path. Discord voice works but rides an
  experimental library (see Discord section) — use local when you want smooth conversation.

---

## Environment
- Project root: `D:\aiproject\`. Git repo, `.gitignore`, `.venv` present.
- `.gitignore` covers: `.venv/`, `__pycache__/`, `memory_store/`, `test_out.wav`, `.env`,
  `clock.json`.
- Installed: `openai`, `chromadb`, `requests`, `sounddevice`, `soundfile`,
  `faster-whisper`, `numpy`, and for Discord: **`py-cord[voice]` (PINNED — see below)** +
  **`ffmpeg` on PATH** (for voice playback).
- **CUDA DLL fix (Windows + faster-whisper):** `pip install nvidia-cublas-cu12
  "nvidia-cudnn-cu12==9.*"` AND the DLL-registration shim at the top of `wernickes_area.py`.
  Both required. Install with `python -m pip` so wheels land in the venv, not conda base.
- **GPT-SoVITS** installed SEPARATELY at `D:\GPT-SoVITS\` (own bundled Python). Not in git.

### ⚠️ py-cord is PINNED to an experimental commit — DO NOT casually upgrade
Discord voice **receive** requires DAVE (Discord's mandatory E2EE, enforced since Mar 2026).
Stable py-cord (2.8.x) does NOT decrypt received voice; an in-progress PR build does. We are
pinned to that build:
```
py-cord[voice] @ git+https://github.com/Pycord-Development/pycord.git@820460aa4
```
A plain `pip install -U py-cord` or a fresh venv will pull stable and **silently break voice
receive** — and the symptom looks identical to the bugs we already chased (noise / no audio).
Keep that exact line in requirements. Re-check the PR every few weeks; when py-cord ships
steady DAVE receive (or it merges to stable), unpin and lower `VOICE_END_SILENCE` back to ~2s.

---

## GPT-SoVITS setup (read before touching voice)
- Windows integrated package at `D:\GPT-SoVITS\`. Start in API mode (not WebUI):
  ```powershell
  cd D:\GPT-SoVITS
  .\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880
  ```
- **CRITICAL — half precision:** in `GPT_SoVITS\configs\tts_infer.yaml` set `is_half: false`
  (keep `device: cuda`), restart. With `is_half: true` on this GPU synthesis returns a
  full-size but ZERO-amplitude (silent) wav.
- **Reference voice:** clip at `brain\forebrain\cerebrum\frontal_lobe\voice\reference.wav`;
  transcript in the `VOICE` dict in `brocas_area.py`. Must be 3–10s, clean, single speaker.

---

## Folder structure (current)
```
D:\aiproject\
├── main.py                 # entry point — routes ALL I/O through the active adapter
├── memory_store\           # Chroma vector store (auto-created, gitignored)
├── clock.json              # persistent last-active timestamp (gitignored)
├── PRIVACY_POLICY.md       # template (placeholders to fill: operator, contact, date, jurisdiction)
├── TERMS_OF_SERVICE.md     # template (same placeholders)
├── peripheral_nervous_system\          # [NEW] swappable I/O substrate (NOT a brain region)
│   ├── io_adapter.py        # IOAdapter base + InputEvent dataclass + FINAL/PARTIAL/INTERRUPT
│   ├── local_adapter.py     # mic + speakers (wraps wernickes + brocas); local mode
│   └── discord_adapter.py   # Discord bot: text + voice (py-cord); the big one
└── brain\
    └── forebrain\
        ├── cerebrum\
        │   ├── frontal_lobe\
        │   │   ├── prefrontal_cortex.py   # persona + GROUNDING + think() + consider_speaking()
        │   │   ├── brocas_area.py         # TTS: GPT-SoVITS pipeline
        │   │   └── voice\reference.wav
        │   └── temporal_lobe\
        │       └── wernickes_area.py      # STT: faster-whisper (+ public transcribe())
        └── subcortical_structures\
            ├── basal_ganglia\
            │   └── action_selector.py     # should_respond() addressed-detection + engagement window
            ├── hypothalamus.py            # [NEW] persistent clock (last_active across restarts)
            └── limbic_system\
                ├── amygdala.py            # emotion state that colors responses
                └── hippocampus.py         # long-term memory + consolidation
```
Every package dir has `__init__.py`. Brain core is I/O-agnostic; the PNS adapters feed it.

---

## Module summary (changes since Phase 3 handoff)

### prefrontal_cortex.py — "the CEO"
- `MODEL` constant (currently `qwen2.5:3b`) — used by `think()`, `consider_speaking()`, and
  the legacy `judge_relevance()`.
- **`GROUNDING` block** — hard anti-confabulation rules kept separate from persona: she may
  only reference people / events / past conversations that are in the current chat or in her
  recalled memories; otherwise she says she doesn't know rather than invent. There's also an
  explicit "no memories surfacing right now" guard. `temperature` lowered 0.9 → 0.7.
- `think(history, mood_flavor, memories, situation)` — generates a reply; used when she's
  **directly addressed** (name / @ / reply / DM) or interrupting. Always speaks.
- **`consider_speaking(...)`** — the autonomous floor decision for chatter she was NOT
  addressed in. ONE model call both decides AND writes: returns her line, or `[QUIET]` →
  stay silent. Biased toward joining in. This replaced the old rigid two-gate (should_respond
  "consider" + judge_relevance), which made her ignore too much.
- `_build_system(...)` — shared prompt assembly for both paths.

### action_selector.py (basal ganglia)
- `should_respond(text, *, mentioned, reply_to_her, channel, now)` → Decision. Now used by
  main.py mainly to detect **addressed** (name/@/reply/DM). Un-addressed chatter goes to
  `consider_speaking`. `mark_engaged()` / window logic still present but no longer gates.

### hypothalamus.py — [NEW] persistent clock
- Persists `last_active` to `memory_store/clock.json`. `touch()`, `last_active()`,
  `time_since_last_active()`. Read at startup so Mira knows the downtime gap (used in the
  situation note). No background processing yet (true scheduler is Phase 7).

### wernickes_area.py — STT
- Local continuous-listening behavior unchanged. Added a public **`transcribe(f32_16k_mono)`**
  so non-mic sources (Discord) reuse Whisper without opening the mic. DLL shim must stay first.

### main.py — adapter-routed loop
- Picks LocalAdapter or DiscordAdapter at launch; everything flows through `on_event`.
- `handle_message(event)`: ingests every utterance → detects addressed → **addressed/interrupt
  → think(); else → consider_speaking()** (silent = logged `[heard:quiet]`). Builds a
  time/place/silence "situation" note (per-channel + cross-restart via hypothalamus).
- PARTIAL display is a single-line rolling caption (truncated to terminal width) so live
  transcription doesn't wrap and stack into repeats.

### discord_adapter.py — Discord I/O (text + voice)  [NEW, the big one]
- **Client built inside its own event-loop thread** (py-cord voice internals bind to the loop
  active at construction; otherwise voice connect fails "Future attached to a different loop").
- **Text:** `on_message` → gating in the brain. Token via `.env` `DISCORD_BOT_TOKEN`
  (gitignored). Message Content Intent ON.
- **Voice:** `mira join` / `mira leave` commands. Receive via `vc.start_recording` + a custom
  `discord.sinks.Sink` whose `write()` buffers PCM per speaker. STT via wernickes `transcribe`,
  TTS out via `vc.play(FFmpegPCMAudio(...))`. `_brain_busy` flag pauses live partials while she
  speaks so Whisper never fights GPT-SoVITS for the GPU.
- **Endpointing = wall-clock packet gap, NOT in-buffer silence.** Discord (voice-activity mode)
  stops sending packets when you're quiet, so silence never enters the buffer — we time the
  gap since the last packet arrived. `VOICE_END_SILENCE` currently **6.0s** (see limitation).
- **Speaker name resolution** (`_resolve_member`): py-cord may hand the sink a bare
  `discord.Object` (id only). Resolves id → VC channel members → guild cache → user cache.
  If a speaker still shows as `<Object id=…>`, enable the **Server Members Intent** (portal +
  `intents.members = True`) so the member list stays cached.

---

## Running the app
1. Start **Ollama**.
2. Start **GPT-SoVITS** API: `cd D:\GPT-SoVITS; .\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880`.
3. From `D:\aiproject\` (venv active):
   - Local: `python main.py` — just talk; pause and she replies. Use headphones.
   - Discord: `python main.py --discord` — then `mira join` in a server text channel while
     you're in a voice channel.

---

## Status

### PHASE 3 — VOICE: COMPLETE ✅ (local mic + speakers, unchanged)

### PHASE 4 — ATTENTION & BEHAVIOR: action selection done; two pieces deferred
- ✅ **Action selector (basal ganglia)** — `consider_speaking` autonomous "speak or stay quiet,"
  biased to engage; addressed → always answer. Replaces the old over-eager-to-ignore gate.
- ✅ **(Bonus, not in plan) Discord I/O layer** — text fully working; voice working with caveats.
- ✅ **Grounding / anti-confabulation** — GROUNDING rules + qwen2.5:3b + temp 0.7.
- ⬜ **RAS message prioritization** (spam filter, rank by relevance/subs/keywords) — DEFERRED.
  Matters for busy multi-viewer text chat; build alongside stream/audience work.
- ⬜ **Idle / default-mode behaviors** (ramble/comment when quiet) — DEFERRED. Best built with
  the avatar (it's stream dead-air filler).

### Discord voice — known limitations (experimental-build bound)
- The pinned #3159 receive build delivers audio **late, in bursts with multi-second gaps**
  (~4s seen), and occasionally **stale/duplicated** audio. `VOICE_END_SILENCE = 6.0` rides over
  the delivery gaps so one turn isn't chopped into many — at the cost of ~6s latency after you
  finish. No adapter logic fully fixes the stale-audio flakiness; it's upstream. **Local voice
  is the smooth path** for real conversation. Revisit when py-cord's DAVE receive matures.
- Barge-in (her yielding when you talk over her) was built then **removed** — an open/hot mic
  triggered it on background noise. Re-add later via push-to-talk or a "Mira stop" keyword if wanted.

### Still deferred from earlier phases (not blockers)
- **Per-person / per-channel memory** (who said what, last-seen) — belongs in hippocampus;
  needs speaker-identity + multi-person testing.
- **Phase 2 memory refinements:** consolidation dedup; a recall **relevance/distance threshold**
  (loosely-related memories can get injected → a vector cause of invention); crash-safe periodic
  session summaries.
- **ToS / Privacy** templates exist in the repo root — fill the placeholders + host them.

---

## NEXT: Phase 5 — Body (avatar)
Per the build plan:
- **VTube Studio API or VRM control** — lip sync + expressions tied to the amygdala mood state
  → *motor cortex*.
- **Animation smoothing / timing** → *cerebellum*.
- Tie-in: the emotion state (`amygdala.mood`) already exists and colors replies — drive
  expressions from it. Lip sync can hang off the TTS playback in `brocas_area` / the adapter's
  voice-out path. (Then loop back for the deferred Phase 4 idle behaviors once she's visible.)

## Working style for next session
- Step by step, one step at a time, not verbose.
- Stay local / free unless a fully-free fast option is explicitly wanted.
- Before any `pip install -U`, remember the py-cord pin.