# Speech-to-Speech (S2S) feasibility — replacing the LLM + TTS stack

Branch: `llm-s2s`. Question: can we swap Mira's current **STT → LLM → TTS** pipeline for a
single **speech-to-speech** model (e.g. NVIDIA **PersonaPlex**) so we drop the separate TTS
stage?

**Short answer:** Conceptually a great fit for Mira (full-duplex, persona + voice control,
no separate TTS). **But it is not runnable on the current dev GPU**, and it would bypass most
of Mira's "brain" (memory, games, text channels). Recommended path: treat S2S as a *cloud /
future-hardware* mode, prototype with a small Omni model first, and keep the text brain as the
source of truth. Details below.

---

## 1. What PersonaPlex actually is

- NVIDIA `personaplex-7b-v1` — real-time, **full-duplex** speech-to-speech. One 7B model does
  streaming speech understanding **and** speech generation in a single pass (predicts text
  *and* audio tokens autoregressively over a neural codec). Handles barge-in, overlaps,
  backchannels ("uh-huh"), fast turn-taking.
- Persona control via **text role-prompt** + **audio voice-conditioning** clip — i.e. you can
  give it Mira's persona text and a reference voice sample. That maps cleanly onto our
  `PERSONA` string + the GPT-SoVITS reference clip we already have.
- Built on the **Moshi** architecture (Kyutai), LLM backbone is **Helium**.
- **Gated** weights (must accept NVIDIA license on HF). Checkpoint ~16.7 GB.
- **VRAM: ~20–24 GB for comfortable inference.**

Sources:
[NVIDIA ADLR](https://research.nvidia.com/labs/adlr/personaplex/) ·
[HF model card](https://huggingface.co/nvidia/personaplex-7b-v1) ·
[GitHub](https://github.com/NVIDIA/personaplex) ·
[KDnuggets walkthrough](https://www.kdnuggets.com/run-a-real-time-speech-to-speech-ai-model-locally)

## 2. The hard blocker: hardware

| | VRAM needed | Fits current dev GPU (GTX 1660 Super, **6 GB**)? |
|---|---|---|
| PersonaPlex-7B | ~20–24 GB | ❌ no (3–4× over) |
| Moshi | 16–20 GB | ❌ no |
| GLM-4-Voice 9B | ~18 GB | ❌ no |
| Hertz-dev | 8–12 GB | ❌ no |
| Sesame CSM-1B | 6–8 GB | ⚠️ borderline, and it's voice-gen, not a full S2S brain |
| Qwen2.5-Omni-3B (and 4-bit 7B) | ~6–8 GB quantized | ⚠️ maybe, tight |
| Mini-Omni2 (~1.5B) | ~4 GB | ✅ yes, but noticeably lower quality |

The current box reports **6144 MiB total, ~4.5 GB free**, and Whisper STT + Kokoro TTS already
share it. PersonaPlex on this machine is a non-starter **locally**. It is only realistic via:
- a **cloud GPU** (RunPod/etc. — we already have a RunPod path for GPT-SoVITS), or
- the **other machine** referenced in the code (`start_laptop.bat` mentions a free 8 GB GPU;
  `wernickes_area.py` comments mention an RTX 5050) — still under PersonaPlex's 20 GB ask.

Sources:
[Spheron S2S GPU guide](https://www.spheron.network/blog/speech-to-speech-gpu-cloud-moshi-sesame-csm-hertz-dev/) ·
[Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) ·
[Qwen2.5-Omni](https://github.com/qwenlm/qwen2.5-omni)

## 3. The architectural cost: S2S bypasses most of Mira's brain

Today the pipeline is **text in the middle**, and a lot hangs off that text:

```
mic → wernickes_area (Whisper STT) → text
        → handle_message (main.py)
            → hippocampus.recall()        # RAG memory injected into the prompt
            → _recall_for() documents     # PDF/rulebook RAG
            → game_master.intercept()     # Deep IQ / MTG game mode
            → prefrontal_cortex.think_stream()  # persona + memory + mood → sentences
            → _sanitize()                 # strips leaked role labels / fake turns
        → brocas_area (Kokoro/Piper/GPT-SoVITS TTS) → speaker
            → cerebellum.lip (avatar lip-sync from audio envelope)
        → hippocampus.observe()/consolidate()  # writes new memories
   (also: Discord text + voice adapters, text-chat mode, --subconscious mind)
```

A pure S2S model collapses STT+LLM+TTS into one box. What we **lose or must re-plumb**:

- **Memory / RAG** (`hippocampus`, Chroma + `nomic-embed-text`): S2S has no place to inject
  recalled memories or document passages mid-stream. This is the biggest loss — grounding and
  "remember when…" depend on it.
- **Output hygiene** (`_sanitize`, the streaming label-peeling): can't filter audio tokens the
  way we filter text. Whatever the model says, it says.
- **Games** (`game_master`, Deep IQ / MTG): pure text logic, expects to intercept text turns.
- **Text channels**: Discord text chat and local text-chat mode have no audio at all.
- **Speaker identity** (`_identity_block`, "who is talking", Senpai gating): some S2S models do
  speaker turns, but our name-based recognition is text-driven.
- **Subconscious** (`--subconscious`: pre-drafting, chime-ins, daydreams): all text-model calls.

What largely **survives**: persona (→ role prompt), voice identity (→ audio conditioning clip),
avatar lip-sync (can still be derived from the model's **output** audio envelope), barge-in
(S2S does this *better* than our VAD-based interrupt).

**Implication:** S2S is not a drop-in for `prefrontal_cortex`. It replaces the *voice loop* but
not the *brain*. A faithful integration needs the text brain to still run (for memory, games,
text channels) — which means S2S becomes a **mode**, not a wholesale replacement.

## 4. Integration shape (if/when hardware allows)

The codebase is already provider-pluggable (`MIRA_LLM_PROVIDER` ollama/gemini/groq;
`MIRA_TTS` kokoro/piper/gptsovits), so a clean way to add S2S is a **fourth runtime mode**
rather than editing the existing text path:

- Add a `peripheral_nervous_system` / adapter (mirrors the Discord voice adapter) that owns the
  full-duplex audio stream to a PersonaPlex server (local or RunPod, same as GPT-SoVITS).
- Feed it the `PERSONA` text as the role prompt and the existing reference clip
  (`voice/sample_clip.mp3`) as the voice conditioning.
- **Hybrid option (recommended)** to keep the brain: run RAG/memory/games as a pre-step that
  injects a short context string into the role prompt each turn, and still call
  `hippocampus.observe()` on the recognized transcript PersonaPlex emits (it predicts text
  tokens too, so we can tap those for memory writes + Discord text mirroring).
- Drive `cerebellum.lip` from the S2S output audio envelope (reuse `brocas_area._rms_envelope`).
- Gate it behind `MIRA_VOICE_MODE=s2s` + a `start_s2s.bat`, pointing at a cloud/2nd-GPU host.

## 5. Recommendation

1. **Do not target PersonaPlex on the 6 GB GTX 1660.** It physically won't load.
2. **Cheapest way to actually try S2S now:** prototype **Qwen2.5-Omni-3B** (aligns with our
   existing Qwen usage and OpenAI-compatible tooling) or **Mini-Omni2** locally to feel out the
   UX and the brain-bypass tradeoffs, accepting lower quality.
3. **Best-quality path:** run **PersonaPlex on a cloud GPU** (RunPod, ≥24 GB — reuse the
   pattern we already have for GPT-SoVITS) or on a future ≥24 GB local card, behind a new
   `s2s` voice mode, **hybridized** with the text brain so memory/games/text channels survive.
4. Keep the current STT→LLM→TTS pipeline as the default; S2S is an opt-in mode.

**Net:** Possible and attractive, but it's a *new mode requiring new hardware*, not a swap that
saves work on the current machine. The win (true full-duplex, no TTS stage) is real; the cost
is VRAM and re-plumbing the brain so it isn't bypassed.
