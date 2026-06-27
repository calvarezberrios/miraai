<#
  run-mira-small.ps1  --  PC-side BRAIN SERVER, lightweight edition.

  Runs a small DENSE Qwen3 *entirely on the 6 GB GPU* so host RAM stays free for
  other programs. This is the opposite tradeoff from the 35B MoE: instead of pinning
  ~20 GB of experts in RAM (--mlock --n-cpu-moe), the whole model lives in VRAM, the
  CPU is idle, and ~18-20 GB of RAM is yours again.

  Same Docker llama-server + the SAME browser UI at  http://localhost:8080  as the 35B.

    -Size 8b  (DEFAULT)  Qwen3-8B Q4_K_S (~4.6 GB)  -- closest personality to llama-turbo
    -Size 4b             Qwen3-4B Q4_K_M (~2.5 GB)  -- faster, even more headroom

  Switch any time:  ./run-mira-small.ps1 -Size 4b   (recreates the container)

  Laptop side is UNCHANGED: Mira still points at  http://<this-pc-ip>:8080/v1
  PREREQS: models downloaded (./download-small.ps1) and the turboquant binary built
  once (./build-turboquant.ps1). Firewall rule for :8080 -- see run-turbo-server.ps1.
#>
param(
    [ValidateSet('8b','4b')][string]$Size = '8b',
    [string]$ModelDir = "D:\models",
    [string]$LlamaDir = "D:\llama-cpp-turboquant",
    [string]$Image    = "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    [int]   $CtxSize  = 32768   # Qwen3-8B native trained context (no YaRN needed). Fits a
                                # ~1h40m session in one pass; longer ones auto-chunk in the
                                # scribe. Raising past 32768 requires --rope-scaling yarn.
)

$models = @{
    '8b' = 'Qwen_Qwen3-8B-Q4_K_S.gguf'
    '4b' = 'Qwen_Qwen3-4B-Q4_K_M.gguf'
}
$Model = $models[$Size]

if (-not (Test-Path (Join-Path $ModelDir $Model))) {
    Write-Warning "Model not found: $(Join-Path $ModelDir $Model)"
    Write-Host  "Download it first:  ./download-small.ps1 -Size $Size" -ForegroundColor Yellow
    return
}

docker rm -f llama-turbo 2>$null | Out-Null
# Dense model, full GPU offload. NOTE vs the 35B script:
#   - dropped --n-cpu-moe  (no experts to offload; it's dense)
#   - dropped --mlock --no-mmap (don't pin RAM; mmap lets the OS reclaim pages)
#   - kept turbo4/turbo3 KV  (saves VRAM headroom on the 6 GB card)
docker run -d --name llama-turbo `
  --gpus all --ulimit memlock=-1 --cap-add=IPC_LOCK `
  -p 8080:8080 `
  -v ${LlamaDir}:/llama `
  -v ${ModelDir}:/models `
  $Image `
  /llama/build/bin/llama-server `
    -m /models/$Model `
    --host 0.0.0.0 --port 8080 `
    --cache-type-k turbo4 --cache-type-v turbo3 `
    -ngl 99 --jinja `
    -c $CtxSize | Out-Null

Write-Host "Loading $Model @ ${CtxSize} ctx on GPU (~20-40s)..." -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 48; $i++) {
    Start-Sleep -Seconds 5
    if (-not (docker ps --filter "name=llama-turbo" --format "{{.Status}}")) {
        Write-Warning "Container exited early -- check: docker logs llama-turbo"
        Write-Host  "If it OOMed on VRAM, try the 4B:  ./run-mira-small.ps1 -Size 4b" -ForegroundColor Yellow
        return
    }
    try { if ((Invoke-RestMethod "http://localhost:8080/health" -TimeoutSec 3).status -eq "ok") { $ready = $true; break } } catch {}
}
if (-not $ready) { Write-Warning "Did not become ready -- check: docker logs llama-turbo"; return }

$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { ($_.InterfaceAlias -like "Wi-Fi*" -or $_.InterfaceAlias -like "Ethernet*") -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -eq "Dhcp" } | Select-Object -First 1).IPAddress
Write-Host "`nBRAIN SERVER READY ($Size)." -ForegroundColor Green
Write-Host "Browser UI:        http://localhost:8080" -ForegroundColor Green
Write-Host "Laptop / Mira ->   http://${ip}:8080/v1" -ForegroundColor Green
Write-Host "(verify the IP each session -- DHCP can change it; a router reservation keeps it fixed)" -ForegroundColor Yellow
