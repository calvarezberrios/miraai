<#
  run-turbo-mira.ps1  --  start the Qwen3.6-35B turbo model as Mira's BRAIN SERVER.
  Tuned for the voice loop (NOT the 256k text experiment):
    - Q3_K_M quant (~16 GB) so it leaves host RAM for Mira's Whisper + Kokoro + avatar
    - 8k context  -> fast time-to-first-token (~1s), ~9 tok/s generation
    - reasoning is disabled on the MIRA side (start_turbo.bat sets MIRA_NO_THINK=1)
  Runs detached on :8080. Pair with .wslconfig memory=16GB so the host keeps ~8 GB free.
  Then launch Mira with start_turbo.bat.
#>
param(
    [string]$ModelDir = "D:\models",
    [string]$Model    = "Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf",
    [int]   $CtxSize  = 8192
)
if (-not (Test-Path (Join-Path $ModelDir $Model))) {
    Write-Warning "Model not found: $(Join-Path $ModelDir $Model) -- download it first."
    return
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

Write-Host "Loading $Model @ ${CtxSize} ctx (mlock ~16 GB, ~3-4 min)..." -ForegroundColor Cyan
for ($i = 0; $i -lt 96; $i++) {
    Start-Sleep -Seconds 5
    if (-not (docker ps --filter "name=llama-turbo" --format "{{.Status}}")) {
        Write-Warning "Container exited early -- check: docker logs llama-turbo"; return
    }
    try { if ((Invoke-RestMethod "http://localhost:8080/health" -TimeoutSec 3).status -eq "ok") {
        Write-Host "Turbo brain ready on http://localhost:8080  ->  now run start_turbo.bat" -ForegroundColor Green; return
    } } catch {}
}
Write-Warning "Did not become ready in time -- check: docker logs llama-turbo"
