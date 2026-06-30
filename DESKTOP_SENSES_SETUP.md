# Desktop senses setup — give laptop-Mira eyes + game-audio ears

Mira runs on the **laptop**, but you stream on the **desktop**. Her `--vision` and
`--game-audio` capture *the machine she runs on*, so on their own they'd see a blank laptop
screen and hear silence. The Discord VC only carries your **voice**, not the screen or the game
sound.

This guide sets up the **senses companion** (`tools/desktop_senses.py`, launched by
`start_desktop_senses.bat`) that runs on the **desktop** and ships its screen + game audio to
Mira over your LAN — a side channel completely separate from the Discord VC.

```
 DESKTOP (the stream)                              LAPTOP (Mira)
   start_desktop_senses.bat                          start_stream_servers.bat
     screen      ──► :8200/frame ───── LAN ────►   --vision      (laptop VL captions the frame)
     game audio ─► Whisper ─► :8200/game-audio ─►  --game-audio  (ingests dialogue text)
                                                    --twitch      (reads Twitch's API directly)
```

**Who does what:** the desktop grabs the screen and transcribes the game audio (its own
Whisper); the laptop does the brain, your-mic STT, TTS, and the actual vision *captioning*. Only
small JPEGs and short text lines cross the network. **Twitch chat needs nothing here** — the
laptop reads Twitch directly.

---

## 1. Prerequisites on the desktop

The desktop already has the repo + venv from when Mira ran there. From the repo folder:

1. **Update the code** (the companion lives on the `laptop_run` branch):
   ```powershell
   git fetch
   git checkout laptop_run
   git pull
   ```

2. **Activate the venv and install the companion's deps** (most are already present):
   ```powershell
   .\.venv\Scripts\activate
   python -m pip install mss pillow sounddevice faster-whisper numpy
   ```

3. **NVIDIA driver / CUDA** for game-audio Whisper. The desktop's GTX 1660 (Turing) has a
   broken fp16 path, so the launcher uses `int8_float16` (already set). No action needed unless
   Whisper fails to load — then fall back to CPU (see Troubleshooting).

---

## 2. Pick the game-audio capture device

The companion captures **what the stream hears** (the desktop's output mix), not your mic. It
needs a *recordable* output — a WASAPI loopback of your speakers/headphones, a "Stereo Mix"
input, a "CABLE Output", or a Voicemeeter "Output" bus.

1. **List the devices:**
   ```powershell
   python tools\check_capture.py devices
   ```
   Note the **name or index** of the output you want to capture (the one your game plays to).

2. **Set it in `.env`** (in the repo root) — name substring or index both work:
   ```
   MIRA_GAME_AUDIO_DEVICE=CABLE Output
   ```
   Leave it unset to auto-pick the **default output** via WASAPI loopback (works on most setups
   with `sounddevice >= 0.5.0`).

3. **Use headphones.** If game sound comes out of speakers it bleeds into your mic and reaches
   Mira under *your* name. Headphones keep the game audio on the loopback only.

> Tip: `python tools\check_capture.py audio` plays-through a few seconds and measures the level,
> so you can confirm the device actually carries the game audio before going live.

---

## 3. Pick which monitor she sees (multi-monitor only)

By default the companion captures the **primary** monitor (the one anchored at top-left 0,0).
To watch a different screen:

1. **List monitors:**
   ```powershell
   python tools\check_capture.py monitors
   ```
2. **Set the index** in `.env` (1 = first physical monitor; never 0 = all-screens stitched):
   ```
   MIRA_VISION_MONITOR=2
   ```

---

## 4. Open the firewall (one time)

So the laptop can reach the companion, in an **elevated** PowerShell:
```powershell
New-NetFirewallRule -DisplayName "Mira senses 8200" -Direction Inbound -Protocol TCP -LocalPort 8200 -Action Allow
```

---

## 5. Run it

```powershell
start_desktop_senses.bat
```

On start it prints something like:
```
============================================================
  Desktop senses serving on http://192.168.12.151:8200
    screen : ON  -> /frame
    audio  : ON  -> /game-audio
  On the LAPTOP set DESKTOP_IP=192.168.12.151 in start_stream_servers.bat
============================================================
```

Note that **IP**. When game dialogue is heard it also prints lines like `[senses/audio] game> ...`
so you can see it working. Leave this window open while you stream. `Ctrl+C` stops it.

---

## 6. Point the laptop at the desktop

On the **laptop**, edit the top of `start_stream_servers.bat`:
```bat
set DESKTOP_IP=192.168.12.151
```
Use the IP the companion printed. Then launch Mira on the laptop as usual:
```
start_stream_servers.bat
```
It auto-wires `MIRA_VISION_FRAME_URL` / `MIRA_GAME_AUDIO_URL` to the companion. (Unsetting those
two makes the laptop capture itself instead — useful only if everything moves onto one machine.)

> **IPs are DHCP** — recheck the desktop's IP each session (the companion prints it), or set a
> router reservation so it stays fixed.
>
> **Auto-discovery (self-healing):** if the desktop's IP moves and `DESKTOP_IP` goes stale, the
> laptop notices the timeouts, **rescans the LAN for the companion** (a host answering
> `/health` with `ok` on `:8200`), and re-points itself — you'll see `[vision] found it —
> re-pointed to ...` / `[game-audio] found it — ...` in the laptop console, and senses come back
> in ~20–30s without editing anything. Keeping `DESKTOP_IP` current just avoids that initial
> reconnect delay. Disable the scan with `MIRA_SENSES_AUTODISCOVER=0`.

---

## 7. Verify the link (optional but recommended)

**From the desktop** (does the server answer locally?):
```powershell
curl http://localhost:8200/health           # -> ok
curl http://localhost:8200/game-audio        # -> {"last": N, "lines": [...]}
```

**From the laptop** (does it reach across the LAN?):
```powershell
curl http://<DESKTOP_IP>:8200/health         # -> ok
curl -o frame.jpg http://<DESKTOP_IP>:8200/frame   # saves a screenshot of the desktop
```
If `/frame` saves a picture of your desktop and `/health` returns `ok` from the laptop, the
bridge is up. In Mira's laptop console you'll then see `[vision] pulling screen frames from the
desktop senses companion: ...` and, when she captions, `On screen right now: ...` folded into
her context.

---

## Configuration reference (env / `.env`)

| Variable | Default | What it does |
|---|---|---|
| `MIRA_SENSES_PORT` | `8200` | HTTP port the companion serves on |
| `MIRA_SENSES_MAX_DIM` | `1024` | Longest edge of the served screen JPEG (bandwidth vs detail) |
| `MIRA_VISION_EVERY` | `2` | Seconds between screen grabs |
| `MIRA_VISION_MONITOR` | primary | mss monitor index to capture |
| `MIRA_GAME_AUDIO_DEVICE` | default output | Loopback/output device name or index |
| `WHISPER_DEVICE` | `cuda` | Game-audio STT device (`cpu` to offload the GPU) |
| `WHISPER_MODEL_SIZE` | `small.en` | Whisper model for game audio |
| `WHISPER_COMPUTE_TYPE` | `int8_float16` | 1660 needs this (no fp16); use `int8` on CPU |
| `MIRA_GAME_END_SILENCE` | `1.0` | Trailing-quiet seconds that end one dialogue line |
| `MIRA_GAME_MIN_SPOKEN` | `0.4` | Ignore speech blips shorter than this |
| `MIRA_GAME_MAX_SEC` | `30` | Force-finalize a long monologue after this |
| `MIRA_GAME_SILENCE_RMS` | `0.006` | Loudness below which audio counts as "quiet" |

`start_desktop_senses.bat` already sets the Whisper + cadence values; the rest come from `.env`
or fall back to the defaults above.

---

## Troubleshooting

- **Laptop can't reach `:8200`** — firewall (step 4), wrong/stale `DESKTOP_IP`, or the machines
  are on different networks (Wi-Fi vs Ethernet, or guest-network isolation). Test with the curl
  in step 7 from the laptop.
- **`audio: off` in the banner / "couldn't open capture"** — the device wasn't recordable. Run
  `python tools\check_capture.py devices`, set `MIRA_GAME_AUDIO_DEVICE` to a loopback/Output
  device, and make sure `sounddevice >= 0.5.0` (`pip install -U sounddevice`).
- **Whisper fails to load on CUDA** — set in this terminal before launching, or edit the bat:
  ```powershell
  $env:WHISPER_DEVICE="cpu"; $env:WHISPER_COMPUTE_TYPE="int8"
  ```
  CPU STT is fine for one game-audio stream.
- **She "hears" the game under your name** — game audio is bleeding into your mic. Use
  headphones, or capture a dedicated virtual output (VB-CABLE / Voicemeeter) the game plays to.
- **`screen: off`** — `pip install mss pillow`. On some multi-GPU laptops mss needs the primary
  display active; pick the monitor explicitly with `MIRA_VISION_MONITOR`.
- **Captions look wrong/garbled** — that's the **laptop** VL, not the companion. The companion
  only ships pixels; check the laptop's `run-mira-vision.ps1` server (it must use `q8_0` KV, not
  the turbo KV, or image tokens get shredded — see `VISION_SETUP.md`).
