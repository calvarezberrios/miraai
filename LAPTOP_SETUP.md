# Mira setup

> **★ ALL-ON-THE-LAPTOP as of 2026-06-29 (branch `laptop_run`).** Everything runs on this
> one laptop (RTX 5050, 8 GB) — no second machine, no LAN. Launch it all with
> **`start_discord.bat`**.
>
> ```
> Laptop (RTX 5050 / 8 GB)  —  one box, one click
>   BRAIN : Qwen3-4B turbo (llama.cpp in Docker)   :8080   ~3.1 GB VRAM
>   STT   : Whisper small.en  (CUDA, float16)              ~2 GB VRAM
>   TTS   : Piper (+ optional RVC -> Mira's timbre)        CPU / light
>   MEMORY: Ollama nomic-embed-text                        :11434
>   Mira  : Discord voice + HOST autonomy                  python main.py --discord --host
> ```
>
> - **Why 4B, not 8B:** the brain and Whisper share the single 8 GB GPU. The 8B turbo alone
>   eats ~7.5 GB (no room for GPU STT). The 4B keeps the same Qwen3 turbo pipeline + prompt
>   tuning at ~3.1 GB (16k ctx), leaving ~5 GB for Whisper. Smaller model = a little less
>   personality depth — the accepted tradeoff for running everything on one box.
> - **Run it:** `start_discord.bat`. It (1) brings up the 4B brain in Docker via
>   `llm-test/run-mira-small.ps1 -Size 4b -CtxSize 16384` and health-checks `:8080`,
>   (2) ensures Ollama has `nomic-embed-text`, (3) points everything at `localhost`, sets
>   `MIRA_TTS=piper`, GPU Whisper, then (4) runs `python main.py --discord --host`.
> - **HOST in Discord:** `--host` makes her fill lulls / host the voice channel (works in plain
>   Discord now, not just stream mode). Live toggles, spoken or typed in Discord:
>   `mira host` / `mira take over` hands her the floor, `mira let me talk` / `mira quiet`
>   takes it back.
> - **Full stream mode (vision + Twitch + game audio):** run **`start_stream_servers.bat`**
>   instead. It swaps the 4B brain for **Qwen2.5-VL-7B** (brain AND eyes, ~6.9 GB VRAM via
>   `run-mira-vision.ps1`), so Whisper moves to the **CPU** (the GPU is full; the CPU is idle
>   since the LLM is fully offloaded). Adds `--twitch --vision --game-audio`.
>
>   **Senses live on the DESKTOP, not the laptop.** When you stream on the desktop but Mira runs
>   on the laptop, her local `--vision`/`--game-audio` would capture the *laptop* (blank screen,
>   silence). A companion bridges them over the LAN — `tools/desktop_senses.py`, launched by
>   `start_desktop_senses.bat`:
>   ```
>   DESKTOP (the stream)                           LAPTOP (Mira)
>     start_desktop_senses.bat                        start_stream_servers.bat
>       screen     -> :8200/frame ......LAN.......> --vision  (laptop VL captions the frame)
>       game audio -> Whisper -> :8200/game-audio .> --game-audio (ingests dialogue text)
>                                                    --twitch reads Twitch's API directly
>   ```
>   - On the **desktop**: run `start_desktop_senses.bat` (captures its screen + game-audio
>     loopback, transcribes the audio THERE, serves both on `:8200`). Set `MIRA_GAME_AUDIO_DEVICE`
>     in `.env` (find it: `python tools\check_capture.py devices`), use headphones (so game audio
>     doesn't bleed into the mic), and open TCP 8200 inbound. It prints the desktop's LAN IP.
>   - On the **laptop**: set `DESKTOP_IP=<that IP>` at the top of `start_stream_servers.bat`
>     (it wires `MIRA_VISION_FRAME_URL` / `MIRA_GAME_AUDIO_URL` to the companion). Unset those two
>     to fall back to capturing this laptop instead. Twitch chat needs nothing extra — the laptop
>     reads it directly. Needs `mss` + `pillow` (installed both sides).
>   - **Full desktop walkthrough:** see **`DESKTOP_SENSES_SETUP.md`** (deps, picking the
>     game-audio device, firewall, running it, verifying the link, config + troubleshooting).
> - **Solo hosting with no inputs:** `--host` no longer needs chat/vision/game audio to talk —
>   she fills lulls from her own head (opinions, callbacks to memory, bits, fresh topics), one
>   rotating "move" per beat, and never parrots her own last line.
> - **Voice (Piper → RVC):** Piper alone gives a clean English-female voice. To get Mira's
>   cloned timbre, drop `mira.pth` / `mira.index` in `C:\models\rvc_models\` and build the
>   `.venv-rvc` runtime (`py -3.11 -m venv .venv-rvc && .venv-rvc\Scripts\pip install
>   rvc-python`). Until then it falls back to raw Piper automatically (no error). The RVC
>   model isn't in git — copy it from the desktop.
> - **One-time prereqs:** Docker Desktop + WSL2 + GPU passthrough, the turboquant binary built
>   once (`llm-test/build-turboquant.ps1`), the 4B model
>   (`llm-test/download-small.ps1 -Size 4b -ModelDir C:\models`), the Piper voice in
>   `C:\models\piper\`, Ollama with `nomic-embed-text`, py-cord voice build + ffmpeg, and
>   `.env` with `DISCORD_BOT_TOKEN`.
>
> The sections below describe earlier **split** layouts (brain and Mira on separate machines).
> Keep them for reference if you ever go back to two boxes.

---

# Split setup (brain ⇄ Mira over the LAN)

> **⚠ REVERSED as of 2026-06-27.** The turbo LLM now runs on the **laptop** (RTX 5050,
> 8 GB Blackwell — a fast dense model fits entirely in VRAM) and **Mira runs on the
> desktop**. This is the opposite of the "Original layout" documented below.
>
> ```
> Laptop (192.168.12.116)                     Desktop (GTX 1660 / 24 GB)
>   llama.cpp turbo  :8080  <----LAN chat----  Mira: STT + TTS + avatar + Discord + brain calls
>                                              Ollama (embeds) :11434  (LOCAL on the desktop)
> ```
>
> - **Laptop (the brain):** set up via `llm-test/` — Docker Desktop + WSL2, then
>   `./llm-test/run-mira-small.ps1 -Size 8b -ModelDir C:\models -LlamaDir C:\llama-cpp-turboquant -Image nvidia/cuda:12.8.0-devel-ubuntu22.04`.
>   Serves an 8B dense Qwen3 fully on the GPU (~54 tok/s). Open TCP 8080 in the firewall.
>   The turboquant binary is built for Blackwell (`-CudaArch 120`, CUDA 12.8).
> - **Desktop (Mira):** `git pull` then `start_discord.bat`. It already points at the laptop
>   (`BRAIN_IP=192.168.12.116`), keeps embeddings on the desktop's **own** `localhost:11434`
>   (the Chroma store was built with `nomic-embed-text`, so Ollama stays required here), and
>   runs Whisper `cuda + int8_float16 + small.en` (the 1660's fp16 path is broken).
> - **IP is DHCP** — re-check `192.168.12.116` on the laptop each session and update
>   `BRAIN_IP` if it moved (a router reservation keeps it fixed).
>
> The sections below describe the ORIGINAL layout (brain on the desktop, Mira on the
> laptop). Keep them for reference, or if you flip back — swap the IPs accordingly.

---

## Original layout: brain on the desktop PC, Mira on the laptop

Run the heavy LLM on the desktop (full CPU/RAM/GPU, nothing competing) and run Mira —
mic/STT, voice/TTS, avatar — on the laptop (its 8 GB NVIDIA GPU handles Whisper fast,
no LLM stealing cycles). This is what fixes the STT-hang you hit running both on one box.

```
Desktop PC (192.168.12.151)                 Laptop (16GB / 8GB NVIDIA)
  llama.cpp turbo  :8080  <----LAN chat----  Mira: STT + TTS + avatar + brain calls
  Ollama (embeds)  :11434 <----LAN embed---- (Whisper on the laptop GPU)
```

---

## A. Desktop PC (the brain) — one-time

1. **Apply the RAM bump** (Mira's no longer on the PC, so the LLM gets the whole box):
   ```powershell
   wsl --shutdown      # applies .wslconfig memory=22GB, then relaunch Docker Desktop
   ```
2. **Open the firewall** (run in an ELEVATED PowerShell):
   ```powershell
   New-NetFirewallRule -DisplayName "Mira LLM 8080"   -Direction Inbound -Protocol TCP -LocalPort 8080  -Action Allow
   New-NetFirewallRule -DisplayName "Mira embed 11434" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow
   ```
3. **Expose Ollama on the LAN** (for memory embeddings), then restart Ollama:
   ```powershell
   setx OLLAMA_HOST "0.0.0.0:11434"
   # quit Ollama from the tray and reopen it (or restart the service)
   ollama pull nomic-embed-text        # if not already present
   ```
4. **Start the brain server:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File llm-test\run-turbo-server.ps1
   ```
   It prints the address to use on the laptop, e.g. `http://192.168.12.151:8080/v1`.

> **IP note:** `192.168.12.151` is this PC's current Wi-Fi IP — DHCP can change it. Set a
> router reservation (static lease) so it stays fixed, or re-check it each session with
> `ipconfig`.

---

## B. Laptop (Mira) — one-time install

Mirrors this PC's setup (it's Windows + NVIDIA), but **lighter** — no Ollama model, no
GPT-SoVITS/RVC, no Discord needed. Just STT + Kokoro TTS + the brain call.

1. **Install prerequisites:** Git, **Python 3.11** (`winget install Python.Python.3.11`),
   and the NVIDIA driver (you have the GPU; make sure the driver is current).

2. **Clone the repo** (after you've pushed it from the PC — see section D):
   ```powershell
   git clone <your-repo-url> mira
   cd mira
   git checkout llm-test
   ```

3. **Main venv + deps** (Python 3.11):
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install openai chromadb sounddevice soundfile faster-whisper numpy piper-tts
   python -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"   # GPU STT on Windows
   ```
   (Skip the `py-cord` Discord line — local voice doesn't need it.)

4. **Kokoro TTS venv** (separate Python 3.10, same pattern as the PC):
   ```powershell
   py -3.10 -m venv .venv-kokoro
   .\.venv-kokoro\Scripts\python.exe -m pip install kokoro soundfile
   .\.venv-kokoro\Scripts\python.exe -m spacy download en_core_web_sm
   ```
   Kokoro weights + the `af_jessica` voice auto-download from Hugging Face on first run.

5. **Point Mira at the PC:** edit `set PC_IP=...` at the top of `start_laptop.bat` to the
   desktop's LAN IP (from section A.4). The script already sets everything else:
   chat -> PC `:8080`, embeddings -> PC `:11434`, Whisper on the laptop GPU, reasoning off.

6. *(Optional)* **Bring her memories over:** the long-term memory store isn't in git. To
   keep her existing memories, copy `memory_store\` from the PC to the laptop's repo root.
   Otherwise she starts with a fresh memory and builds it up on the laptop.

---

## C. Run

On the laptop:
```powershell
start_laptop.bat
```
Expect: first reply ~15-20s (PC warming the persona cache), then ~3-8s/turn. STT is now
fast and never freezes — it has the laptop GPU to itself.

**Quick connectivity test** before launching (from the laptop):
```powershell
curl http://192.168.12.151:8080/health        # should return {"status":"ok"}
curl http://192.168.12.151:11434/api/tags      # should list nomic-embed-text
```
If those hang: firewall (section A.2), the PC's IP changed, or both machines aren't on the
same network (Wi-Fi vs Ethernet, or guest network isolation).

---

## D. Getting the code onto the laptop (git)

From this PC, commit and push so the laptop can clone:
```powershell
git add -A
git commit -m "split-host setup: laptop Mira + desktop brain"
git push -u origin llm-test          # needs a remote (GitHub) set up first
```
Then use that repo URL in section B.2.

---

## Honest expectations
- **STT freeze: gone** — the laptop GPU runs Whisper with nothing competing.
- **Reply latency:** still the turbo model's ~3-8s/turn (the desktop's CPU-expert prefill) —
  the split fixes contention, not raw model speed. For ~1-2s/turn, Groq is still the faster
  path; this gives you the local 35B without the crashes.
- **Network:** keep both on the same LAN (wired PC + Wi-Fi laptop is fine). Round-trip is
  negligible next to generation time.

## Where the work runs (and why the laptop GPU "looks idle")
- **Thinking = the desktop.** The LLM runs on the desktop PC, so while Mira "thinks" the **laptop's
  NVIDIA GPU is correctly at 0%** — the laptop just sends a LAN request and waits. Watch the
  *desktop's* GPU/CPU during thinking, and the laptop's *network* tab.
- **Listening (Whisper) + speaking (Kokoro) = the laptop NVIDIA GPU.** Verified: loading Whisper
  uses ~2 GB VRAM and transcription hits ~100% GPU-util; Kokoro loads on `cuda:0`.
- **Task Manager hides this.** CUDA compute shows under the **"Cuda"** engine, which Task Manager's
  GPU graphs DON'T show by default (they show 3D/Copy/Video → read ~0). To see it: Performance →
  GPU (NVIDIA) → change a graph's dropdown to **"Cuda"**, and watch **Dedicated GPU memory**. The
  work is also short/bursty (STT fires ~0.3-0.5s when you stop talking), so it spikes, not sustains.
  The **Intel UHD** usage you see is just the display/compositor + the avatar browser window.
- **Easiest way to actually watch it:** `nvidia-smi -l 1` in a laptop terminal while you talk.
