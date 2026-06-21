<#
  run-turboquant.ps1  --  llama.cpp server WITH TurboQuant KV cache
  Runs the binary built by build-turboquant.ps1 inside the CUDA devel container.
  This is the full command from the instructions, adapted from /root paths to Windows mounts.

  Key TurboQuant flags:  --cache-type-k turbo4  --cache-type-v turbo3
  These shrink the KV cache so a 256k context fits in 6 GB VRAM alongside the
  non-expert layers; the experts live in CPU RAM via --n-cpu-moe 36.
#>
param(
    [string]$SrcDir   = "D:\llama-cpp-turboquant",
    [string]$ModelDir = "D:\models",
    [string]$Model    = "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
    [int]   $CtxSize  = 256000,
    [int]   $Port     = 8080
)

if (-not (Test-Path (Join-Path $SrcDir 'build\bin\llama-server'))) {
    Write-Warning "TurboQuant binary missing -- run build-turboquant.ps1 first."
    return
}
if (-not (Test-Path (Join-Path $ModelDir $Model))) {
    Write-Warning "Model not found -- run 02-download-model.ps1 first."
    return
}

Write-Host "Starting TurboQuant server (ctx=$CtxSize, k=turbo4 v=turbo3) on http://localhost:$Port ..." -ForegroundColor Cyan
docker run --rm -it `
  --gpus all `
  --ulimit memlock=-1 `
  --cap-add=IPC_LOCK `
  -p ${Port}:8080 `
  -v ${SrcDir}:/work `
  -v ${ModelDir}:/models `
  -w /work `
  nvidia/cuda:12.4.1-devel-ubuntu22.04 `
  ./build/bin/llama-server `
    -m /models/$Model `
    --host 0.0.0.0 --port 8080 `
    --cache-type-k turbo4 --cache-type-v turbo3 `
    --n-cpu-moe 36 `
    -ngl 99 `
    --no-mmap --mlock `
    --jinja `
    -c $CtxSize
