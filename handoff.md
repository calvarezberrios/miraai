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
- **TTS engine:** **Piper** (replaced GPT-SoVITS), with optional **RVC** voice conversion.
  Piper generates fast English-female speech. RVC (timbre conversion to a trained `.pth`/`.index`
  voice) is wired up but **currently DISABLED** (`USE_RVC = False` in `brocas_area.py`) because the
  available RVC model output wasn't clean enough. Flip `USE_RVC = True` to re-enable once a
  better-sounding model is available. With RVC off, no subprocess is launched.
- **STT engine:** **Whisper** via **faster-whisper** (accent-robust).
- **OS:** Windows (PowerShell). Ollama at `D:\Ollama\`, project at `D:\aiproject\`.
- **GPU: NVIDIA GTX 1660 Super, 6GB VRAM, FULL-PRECISION ONLY.** Turing TU116 has a broken
  fp16 path (half precision → NaN/silent). HARDWARE, not config; do not chase fp16 fixes.
  GPT-SoVITS must run fp32 (`is_half: false`). VRAM is the day-to-day squeeze. A future GPU
  with working fp16 + 12GB+ is the only real speedup.
- Ollama server must be running before the app works.
- No separate TTS server needed — Piper runs inline, RVC via subprocess.

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

## TTS setup — Piper + RVC (replaced GPT-SoVITS)
No server to start. Piper is a pip package; RVC runs as a subprocess.

### File layout
```
D:\aiproject\
├── piper_voices\en\en_US\hfc_female\medium\
│   ├── en_US-hfc_female-medium.onnx        ← already downloaded
│   └── en_US-hfc_female-medium.onnx.json
├── rvc_models\
│   ├── hubert_base.pt                       ← already downloaded (from lj1995/VoiceConversionWebUI)
│   ├── mira.pth                             ← DROP YOUR RVC MODEL HERE
│   └── mira.index                           ← DROP YOUR RVC INDEX HERE (optional)
└── rvc\
    └── rvc_infer.py                         ← standalone inference script
```

### Enabling / placing an RVC model
`USE_RVC` at the top of `brocas_area.py` toggles it (currently ON). Drop a **v2** `.pth` as
`D:\aiproject\rvc_models\mira.pth` and its `.index` as `mira.index`. Current voice = **Yui
(K-On!)**, a v2 40k model from HF `SmlCoke/rvc-yui`. Tunables at top of `brocas_area.py`:
`PITCH_SHIFT`, `INDEX_RATE` (0.5 default; ↑ for more character, ↓ if it warbles).

### How RVC runs — uses the proven `rvc_python` library (NOT a hand-rolled model)
An earlier version reimplemented the RVC network from scratch; it produced intelligible but
noisy/garbled audio. **Replaced with the `rvc_python` package + RMVPE F0** — far crisper.
`rvc_python` (and `torchcrepe`, `torchfcpe`) are installed into the **GPT-SoVITS bundled Python**
(`D:\GPT-SoVITS-v3lora-20250228\runtime\python.exe`, Python 3.9 with torch+cuda+faiss) — the main
venv (Python 3.14) can't install them. Installed there with `pip install <pkg> --no-deps` so the
existing torch/faiss/fairseq weren't disturbed.

`brocas_area.py` launches `rvc_infer.py --serve` ONCE as a **persistent subprocess**. It loads the
HuBERT + RMVPE + RVC model once (rvc_python auto-downloads HuBERT/RMVPE base models to its own
`base_models/`), prints `READY`, then takes `input|output|pitch` requests on stdin. Hot conversions
are **~0.7s**; cold load is ~10s, so `warmup()` pre-starts it. rvc_python auto-detects the GTX 1660
and forces fp32 (matches this GPU's broken-fp16 constraint).

### F0 method (crispness knob)
`rvc_infer.py` defaults `--f0method rmvpe` — the modern, cleanest pitch estimator. harvest (the old
hand-rolled default) sounds warbly by comparison. Other options: `pm` (fast/rough), `crepe`.

### Fallback behavior
- If `USE_RVC=False` or `mira.pth` is missing → uses raw Piper output (clean female voice).
- If RVC subprocess fails → same Piper fallback, no crash.
- Pitch shift: `PITCH_SHIFT` constant in `brocas_area.py` (0 = no shift, positive = higher).

### Running
1. Start **Ollama**.
2. From `D:\aiproject\` (venv active):
   - Local: `python main.py`
   - Discord: `python main.py --discord`

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
  `discord.sinks.Sink` whose `write()` hands decoded PCM to the adapter. STT via wernickes
  `transcribe`, TTS out via `vc.play(FFmpegPCMAudio(...))`. `_brain_busy` flag pauses live
  partials while she speaks so Whisper never fights the TTS for the GPU.
- **Endpointing now MIRRORS the local mic (rewritten for responsiveness).** Discord only sends
  packets while you actually speak, so silence never arrives on its own. A 50 ms ticker thread
  (`_voice_worker`) advances each speaker's buffer by either the audio that arrived or an equal
  slice of **silence**, making the stream look just like a mic. Then it runs the SAME in-buffer
  VAD endpointing as `wernickes_area._worker`: a turn ends after `VOICE_END_SILENCE` of trailing
  silence (default **2.0s**; local uses 3.0). This replaced the old wall-clock packet-gap timer,
  which dragged because the receive build delivers audio in late bursts. Constants
  (`VOICE_BLOCK_SEC`/`VOICE_REFRESH_SEC`/`VOICE_END_SILENCE`/`VOICE_MIN_SECONDS`/`VOICE_MAX_SECONDS`)
  sit at the top of `discord_adapter.py` and line up with the local mic's values.
  - **If turns get chopped mid-sentence**, the build is delivering in bursts with gaps > the
    threshold → raise `VOICE_END_SILENCE` toward 3.0. If delivery is steady, lower toward 1.0 for
    even snappier replies.
- **Speaker name resolution** (`_resolve_member`): py-cord may hand the sink a bare
  `discord.Object` (id only). Resolves id → VC channel members → guild cache → user cache.
  If a speaker still shows as `<Object id=…>`, enable the **Server Members Intent** (portal +
  `intents.members = True`) so the member list stays cached.

---

## Running the app
1. Start **Ollama**.
2. From `D:\aiproject\` (venv active):
   - Local: `python main.py` — just talk; pause and she replies. Use headphones.
   - Discord: `python main.py --discord` — then `mira join` in a server text channel while
     you're in a voice channel.
3. No GPT-SoVITS server needed anymore.

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

### Discord voice — responsiveness (rewritten to match local)
- The voice path now uses the **silence-filled continuous timeline + in-buffer VAD** model
  (see discord_adapter.py above), so endpointing is as snappy as the local mic — a turn ends
  ~`VOICE_END_SILENCE` (2.0s) after you stop, instead of the old wall-clock packet-gap timer
  that dragged to ~6s. This is the main responsiveness fix.
- **Residual upstream caveat:** the pinned #3159 receive build can still deliver audio in late
  bursts with multi-second gaps, and occasionally stale/duplicated audio. A gap *mid-sentence*
  larger than `VOICE_END_SILENCE` can still chop a turn (the silence-fill reads it as a stop).
  If that happens, raise `VOICE_END_SILENCE` toward 3.0. The stale-audio flakiness is upstream
  and unchanged. Revisit when py-cord's DAVE receive matures.
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