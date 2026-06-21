# llm-test — Qwen3.6-35B-A3B on Docker (stock + TurboQuant)

Run `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` (Q4_K_M, 20.75 GB) under llama.cpp in
Docker, on a **GTX 1660 SUPER (6 GB) / 24 GB RAM** box. The MoE design (3B active)
plus CPU expert-offload (`--n-cpu-moe 36`) and the TurboQuant KV cache are what make
a 35B model fit this hardware.

## Verified on this machine
| Check | Result |
|---|---|
| GPU / driver | GTX 1660 SUPER 6 GB, driver 595.97 (≥470 → WSL2 CUDA OK) |
| RAM | 24 GB |
| Disk | D: 462 GB free → models go to `D:\models` |
| Virtualization | **Enabled** (VBS running) |
| Model repo | exists; `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` = 20.75 GB, single file |
| TurboQuant fork | `github.com/TheTom/llama-cpp-turboquant` exists |
| Docker / WSL2 | **NOT installed** → step 1 |

## The catch with automating this
Installing WSL2 + Docker Desktop needs **Administrator** rights and a **reboot**,
which can't be driven from a non-elevated, non-interactive shell. So step 1 is a
script *you* run as admin; everything after is plain (no admin).

## Steps (run in order, from this folder)

```powershell
# 1. PREREQS — run as Administrator (script self-elevates). Installs WSL2 + Docker
#    Desktop, then reboots. After reboot, launch Docker Desktop once.
./01-install-prereqs.ps1

# 2. Give the WSL2 VM enough RAM to pin a 20.75 GB model (default cap is too low),
#    then apply with `wsl --shutdown` and relaunch Docker Desktop.
./set-wslconfig.ps1
wsl --shutdown

# 3. Download the GGUF to D:\models (~21 GB, resumable).
./02-download-model.ps1

# 4a. STOCK run (no TurboQuant) — prebuilt image, nothing to compile. Smoke test.
#     ctx 128k. Open http://localhost:8080
./run-stock.ps1

# 4b. TURBOQUANT run — build the fork once, then serve with turbo4/turbo3 KV @ 256k ctx.
./build-turboquant.ps1     # one-time, ~10–20 min compile inside CUDA container
./run-turboquant.ps1

# 5. In another terminal, smoke-test whichever server is up:
./test-server.ps1
```

## Flag mapping (instructions → these scripts)
The original commands are Linux/Proxmox-oriented. Adaptations:

| Original | Here | Why |
|---|---|---|
| `-v /path/to/models:/models` / `-v /root:/root` | `-v D:\models:/models`, `-v D:\llama-cpp-turboquant:/work` | Windows host paths (Docker Desktop translates them) |
| (no port in stock cmd) | `-p 8080:8080 --host 0.0.0.0` | reach the server from Windows |
| `--ulimit memlock=-1` | kept (+ added to stock) | required for `--mlock` to pin the model |
| `lxc.prlimit.memlock: unlimited` | **dropped** | LXC/Proxmox-only; N/A on Windows |
| pre-built `./build/bin/llama-server` | `build-turboquant.ps1` compiles it | nothing pre-built on this box |

Everything else (`--n-cpu-moe 36`, `-ngl`, `--no-mmap --mlock`, `--jinja`,
`--cache-type-k turbo4 --cache-type-v turbo3`, ctx sizes) is passed through verbatim.

## If it OOMs (likely-tight, 20.75 GB in 24 GB)
This config is at the edge of the hardware. If the model gets killed or Windows thrashes:
- Lower context: `./run-turboquant.ps1 -CtxSize 64000` (or `32000`).
- Close all other apps; the WSL VM needs ~22 GB.
- Drop to a smaller quant (e.g. `Q3_K_M`, ~16 GB) via `02-download-model.ps1 -File Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf` and pass `-Model` to the run scripts.
- Raise `--n-cpu-moe` to push more experts to CPU (less VRAM, slower).

## Validated results (this machine, 2026-06-21)
Both configs ran end-to-end on the GTX 1660 SUPER / 24 GB box:

| Config | Ctx | Load time | Output | Speed | VRAM |
|---|---|---|---|---|---|
| Stock (f16 KV) | 128k | ~225s | `43` for "17+26" ✓ | ~2.7 tok/s | model+KV in 6 GB, experts on CPU |
| **TurboQuant** (turbo4/turbo3 KV) | **256k** | ~200s | `Tokyo` for capital of Japan ✓ | ~0.9 tok/s | **5669/6144 MiB** |

Notes from bring-up (fixes already folded into the scripts):
- The image entrypoint is `/app/llama-server`, so the `llama-server` token in the
  original stock command is a duplicate that errors (`invalid argument`). Removed.
- Building on a Windows bind-mount can't create versioned `.so` symlinks
  (`Operation not permitted`) → static build (`-DBUILD_SHARED_LIBS=OFF`).
- Qwen3.6 is a **reasoning** model: without `--jinja` the `<think>` block isn't
  parsed and short `max_tokens` returns empty `content`. The TurboQuant run uses
  `--jinja`; give it ≥150 `max_tokens`. Reasoning lands in `message.reasoning_content`.
- Throughput is low because experts run on CPU — expected for a 35B MoE on 6 GB VRAM.

## Security note
The `discord.gg` invite from the instructions was **not** used and isn't trusted —
all downloads here come from Hugging Face (`bartowski`) and GitHub (`TheTom`) over HTTPS.
