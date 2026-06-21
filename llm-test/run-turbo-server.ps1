<#
  run-turbo-server.ps1  --  PC-side BRAIN SERVER for the split setup.
  This PC runs ONLY the LLM (no Mira, no Whisper competing), so it gets the full box:
  Q4 model (better quality than Q3) in the 22 GB WSL, served on the LAN at :8080.

  The laptop runs Mira and points its brain at  http://<this-pc-ip>:8080/v1.

  PREREQS (one-time, see LAPTOP_SETUP.md):
    1. .wslconfig memory=22GB  (already set) -> then `wsl --shutdown` once to apply.
    2. Firewall (run elevated):
         New-NetFirewallRule -DisplayName "Mira LLM 8080"  -Direction Inbound -Protocol TCP -LocalPort 8080  -Action Allow
         New-NetFirewallRule -DisplayName "Mira embed 11434" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow
    3. Ollama reachable on the LAN for memory embeddings (nomic-embed-text):
         setx OLLAMA_HOST "0.0.0.0:11434"   (then restart Ollama)  -- so the laptop can embed.
#>
param(
    [string]$ModelDir = "D:\models",
    [string]$Model    = "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
    [int]   $CtxSize  = 8192
)
if (-not (Test-Path (Join-Path $ModelDir $Model))) {
    Write-Warning "Model not found: $(Join-Path $ModelDir $Model)"; return
}
docker rm -f llama-turbo 2>$null | Out-Null
docker run -d --name llama-turbo `
  --gpus all --ulimit memlock=-1 --cap-add=IPC_LOCK `
  -p 8080:8080 `
  -v D:\llama-cpp-turboquant:/llama `
  -v ${ModelDir}:/models `
  nvidia/cuda:12.4.1-devel-ubuntu22.04 `
  /llama/build/bin/llama-server `
    -m /models/$Model `
    --host 0.0.0.0 --port 8080 `
    --cache-type-k turbo4 --cache-type-v turbo3 `
    --n-cpu-moe 36 -ngl 99 `
    --no-mmap --mlock --jinja `
    -c $CtxSize | Out-Null

Write-Host "Loading $Model @ ${CtxSize} ctx (mlock ~20 GB, ~3-4 min)..." -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 96; $i++) {
    Start-Sleep -Seconds 5
    if (-not (docker ps --filter "name=llama-turbo" --format "{{.Status}}")) {
        Write-Warning "Container exited early -- check: docker logs llama-turbo"; return
    }
    try { if ((Invoke-RestMethod "http://localhost:8080/health" -TimeoutSec 3).status -eq "ok") { $ready = $true; break } } catch {}
}
if (-not $ready) { Write-Warning "Did not become ready -- check: docker logs llama-turbo"; return }

$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -like "Wi-Fi*" -or $_.InterfaceAlias -like "Ethernet*" } | Select-Object -First 1).IPAddress
Write-Host "`nBRAIN SERVER READY." -ForegroundColor Green
Write-Host "On the laptop, point Mira here:  http://${ip}:8080/v1" -ForegroundColor Green
Write-Host "(verify the IP each session -- DHCP can change it; a router reservation keeps it fixed)" -ForegroundColor Yellow