# Stream vision setup — Mira can see the screen

Mira's eyes are **offloaded**, the same way her brain is. A local vision model won't fit the
desktop's 6 GB GTX 1660 next to Whisper + TTS + a game, so the desktop only grabs frames and
ships them to a **Qwen2.5-VL** model on the **laptop** (8 GB). Because the 8 GB laptop GPU
can't hold a separate text brain *and* a VL model at once, the VL model **replaces** the text
turbo — one model is both her brain and her eyes, on the same `:8080` endpoint she already uses.

```
Desktop (Mira, GTX 1660)                         Laptop (8 GB)
  python main.py ... --vision                       llama.cpp + Qwen2.5-VL  :8080
  grabs a frame every ~8s, shrinks to 768px  ──img──►  returns a 1-sentence caption
  caption -> ambient context -> her situation  ◄──────  (also serves normal chat)
```

## 1. Laptop: get the model + projector

Qwen2.5-VL needs **two** GGUF files: the model and its `mmproj` (vision encoder). Put both in
your model dir (`C:\models` on the laptop). 7B is best; 3B is safer on 8 GB.

From Hugging Face (`ggml-org/Qwen2.5-VL-7B-Instruct-GGUF`, or unsloth/bartooski equivalents):

```powershell
# 7B (default)
huggingface-cli download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF `
  Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf `
  --local-dir C:\models
# or 3B
huggingface-cli download ggml-org/Qwen2.5-VL-3B-Instruct-GGUF `
  Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf `
  --local-dir C:\models
```

If your files are named differently, edit the `$models` table at the top of
`llm-test/run-mira-vision.ps1` to match.

## 2. Laptop: serve it

```powershell
cd llm-test
./run-mira-vision.ps1            # 7B, or:  ./run-mira-vision.ps1 -Size 3b
```

This reuses the existing turboquant build + Docker setup; it just adds `--mmproj` so
`llama-server` accepts images. It listens on `:8080` (replacing `run-mira-small.ps1`).

## 3. Desktop: point Mira at it and turn on `--vision`

In `.env` (the IP is the laptop's — same as `OLLAMA_BASE_URL`/`BRAIN_IP` today):

```
OLLAMA_BASE_URL=http://192.168.12.116:8080/v1
MIRA_MODEL=qwen2.5-vl-7b           # chat model = whatever the server reports
MIRA_VISION_MODEL=qwen2.5-vl-7b    # caption model (same one)
# MIRA_VISION_BASE_URL=...         # optional; defaults to OLLAMA_BASE_URL
# MIRA_VISION_EVERY=8              # seconds between frames (raise to cut load/cost)
# MIRA_VISION_MONITOR=1            # which monitor (1 = primary)
# MIRA_VISION_MAX_DIM=768          # longest edge sent to the model
```

Then run:

```
python main.py --discord --twitch --host --vision --game-audio
```

Needs `pip install mss pillow` on the desktop (screen grab + image encode — tiny, CPU only).

## Notes / tuning

- **Cost = frames, not video.** One caption every `MIRA_VISION_EVERY` seconds. Raise it to
  8–15s for slower/cheaper; the caption is injected as ambient context, so she references
  what's on screen when she talks rather than narrating every frame.
- **Verify the model name.** Whatever you set `MIRA_MODEL`/`MIRA_VISION_MODEL` to must match
  what the server advertises at `http://<laptop>:8080/v1/models`.
- **If vision stalls** (laptop endpoint down), `stream_vision.summary()` returns nothing after
  ~30s rather than feeding a stale caption — she just won't mention the screen.
- **Want to keep the text brain separate instead?** You can't on one 8 GB GPU (no room for
  both). Options: run the VL only when you want sight, or run vision against a **cloud**
  multimodal endpoint by setting `MIRA_VISION_BASE_URL`/`MIRA_VISION_MODEL` to it and leaving
  the text brain on `:8080`.
