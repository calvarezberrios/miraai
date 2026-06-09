# AI VTuber Build Plan
*Phased roadmap mapped to human brain regions. Build each phase end-to-end before moving on.*

---

## Phase 1 — Text Brain (core loop)
- LLM with a persona system prompt → *prefrontal cortex*
- Chat input handler (Twitch/YouTube/console) → *thalamus*
- Short-term context window (recent messages) → *working memory*
- Basic emotion state variable (happy/annoyed/excited) that colors responses → *amygdala*

## Phase 2 — Memory
- Vector DB (Chroma/FAISS) for long-term recall of facts, viewers, past streams → *hippocampus*
- Memory write rules: what's worth saving vs. discarding
- Periodic summarization of chat history → *memory consolidation (sleep)*

## Phase 3 — Voice
- Speech output: TTS (ElevenLabs, Piper, or local RVC) → *Broca's area*
- Speech input (optional): Whisper STT for collab/voice chat → *Wernicke's area*
- Queue system so audio doesn't overlap → *motor sequencing*

## Phase 4 — Attention & Behavior
- Message prioritization: filter spam, rank by relevance/subs/keywords → *reticular activating system*
- Action selector: respond / ignore / react / start topic → *basal ganglia*
- Idle behaviors when chat is quiet (ramble, hum, comment on game) → *default mode network*

## Phase 5 — Body (avatar)
- VTube Studio API or VRM control: lip sync, expressions tied to emotion state → *motor cortex*
- Animation smoothing/timing → *cerebellum*

## Phase 6 — Senses (advanced)
- Screen capture + vision model to "see" the game → *visual cortex*
- Sound event detection (alerts, donations) → *auditory cortex*

## Phase 7 — Homeostasis (polish)
- Energy/mood decay over stream time, scheduled behaviors → *hypothalamus*
- Self-correction: detect bad outputs, retry/filter → *anterior cingulate*

---

## Recommended Starting Stack
- **Language:** Python
- **LLM:** OpenAI / Anthropic API
- **Memory:** Chroma
- **TTS:** Piper (free, local) or ElevenLabs (paid, higher quality)
- **Avatar:** VTube Studio API

> **Tip:** A working Phase 1 text loop beats a half-built everything.
