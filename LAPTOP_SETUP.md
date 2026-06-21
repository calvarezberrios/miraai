# Split setup: brain on the desktop PC, Mira on the laptop

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
